#!/usr/bin/env python3
"""Built-in SDXL LoRA trainer (Diffusers + PEFT).
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git

Standalone worker executed as a subprocess by TrainingService when no Kohya
sd-scripts installation is available.  It performs genuine gradient descent on
a LoRA adapter attached to the SDXL UNet attention projections:

  1. Loads the base checkpoint (single-file .safetensors or a diffusers folder).
  2. Encodes every training image once with the VAE and every caption once with
     both text encoders (cached; the encoders are then freed).
  3. Trains only the LoRA parameters with noise-prediction MSE loss.
  4. Saves the adapter in Kohya / ComfyUI key format (``lora_unet_*``) with
     ``modelspec`` metadata, so the file loads in ComfyUI, A1111, Forge and
     diffusers ``load_lora_weights``.

Telemetry lines parsed by TrainingService:
    {step}/{total_steps} loss={loss:.4f} avr_loss={avg:.4f}
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diffusers_lora_worker")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def parse_args():
    p = argparse.ArgumentParser(description="Diffusers / PEFT SDXL LoRA trainer")
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--network_dim", type=int, default=16)
    p.add_argument("--network_alpha", type=float, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_steps", type=int, default=0, help="Hard step cap (0 = derive from epochs)")
    p.add_argument("--max_train_epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--caption_extension", default=".txt")
    p.add_argument("--resolution", default="1024,1024")
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--optimizer_type", default="AdamW")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow_cpu", action="store_true", help="Permit CPU training (smoke tests only)")
    p.add_argument("--base_model_id", default="sdxl-1.0")
    # Accepted for CLI compatibility with Kohya-style commands; no-ops here.
    for flag in ("--enable_bucket", "--network_train_unet_only", "--cache_latents",
                 "--cache_latents_to_disk", "--sdpa", "--no_half_vae", "--bucket_no_upscale",
                 "--random_crop", "--persistent_data_loader_workers"):
        p.add_argument(flag, action="store_true")
    return p.parse_known_args()[0]


def load_samples(image_dir: Path, caption_ext: str) -> List[Tuple[Path, str]]:
    samples = []
    if not image_dir.exists():
        return samples
    for item in sorted(image_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in VALID_EXTS:
            cap = item.with_suffix(caption_ext)
            caption = cap.read_text(encoding="utf-8").strip() if cap.exists() else ""
            if caption:
                samples.append((item, caption))
    return samples


def repeats_from_folder(path: Path, root: Path) -> int:
    """Kohya convention: ``<repeats>_<name>`` concept folders."""
    try:
        rel = path.relative_to(root)
        first = rel.parts[0] if len(rel.parts) > 1 else ""
        head = first.split("_", 1)[0]
        return max(1, int(head)) if head.isdigit() else 1
    except Exception:
        return 1


def fit_image(img, resolution: int):
    """Resize preserving aspect ratio then center-crop to a multiple of 64."""
    from PIL import Image

    w, h = img.size
    scale = resolution / min(w, h)
    nw, nh = max(64, round(w * scale)), max(64, round(h * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    tw = min(nw, resolution) - (min(nw, resolution) % 64)
    th = min(nh, resolution) - (min(nh, resolution) % 64)
    tw, th = max(64, tw), max(64, th)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def to_kohya_state_dict(peft_sd: Dict[str, "object"], alpha: float, torch):
    """Convert PEFT UNet keys to Kohya ``lora_unet_*`` keys with alpha tensors."""
    out = {}
    for key, tensor in peft_sd.items():
        if ".lora_A." in key:
            module, _ = key.split(".lora_A.", 1)
            suffix = "lora_down.weight"
        elif ".lora_B." in key:
            module, _ = key.split(".lora_B.", 1)
            suffix = "lora_up.weight"
        else:
            continue
        kohya = "lora_unet_" + module.replace(".", "_")
        out[f"{kohya}.{suffix}"] = tensor.detach().to("cpu", torch.float16).contiguous()
        out[f"{kohya}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return out


def main():
    args = parse_args()
    t0 = time.time()
    logger.info("Diffusers/PEFT SDXL LoRA worker starting")
    logger.info("Base model: %s", args.pretrained_model_name_or_path)

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from safetensors.torch import save_file
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
    except ImportError as exc:
        logger.error("Required package missing: %s. Run install.bat again (needs torch, diffusers, peft, safetensors).", exc)
        sys.exit(2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("Device: %s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024**3)
    elif args.allow_cpu:
        device = torch.device("cpu")
        logger.warning("No CUDA device; training on CPU because --allow_cpu was given.")
    else:
        logger.error("No CUDA GPU available. Real training requires a CUDA device.")
        sys.exit(3)

    if args.mixed_precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        logger.warning("bf16 not supported on this GPU; falling back to fp16")
        args.mixed_precision = "fp16"
    if device.type == "cpu":
        args.mixed_precision = "no"
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)

    train_dir = Path(args.train_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.output_name}.safetensors"
    resolution = int(str(args.resolution).split(",")[0])

    samples = load_samples(train_dir, args.caption_extension)
    if len(samples) < 2:
        logger.error("Need at least 2 captioned images, found %d in %s", len(samples), train_dir)
        sys.exit(1)
    logger.info("Dataset: %d captioned images", len(samples))

    # ---- Load base model ---------------------------------------------------
    model_path = Path(args.pretrained_model_name_or_path)
    logger.info("Loading base model ...")
    if model_path.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(str(model_path), torch_dtype=weight_dtype)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(str(args.pretrained_model_name_or_path), torch_dtype=weight_dtype)
    unet: UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    # ---- Cache text embeddings ---------------------------------------------
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    text_cache: Dict[str, Tuple[object, object]] = {}
    with torch.no_grad():
        for _, caption in samples:
            if caption in text_cache:
                continue
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(
                prompt=caption, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            text_cache[caption] = (prompt_embeds.to("cpu"), pooled.to("cpu"))
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    del pipe.text_encoder, pipe.text_encoder_2
    logger.info("Cached %d caption embeddings", len(text_cache))

    # ---- Cache latents (VAE in fp32 to avoid SDXL fp16 NaNs) ---------------
    vae.to(device, dtype=torch.float32)
    latent_cache: List[Dict[str, object]] = []
    with torch.no_grad():
        for path, caption in samples:
            with Image.open(path) as im:
                im = fit_image(im.convert("RGB"), resolution)
            w, h = im.size
            arr = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
            lat = vae.encode(arr.unsqueeze(0).to(device)).latent_dist.sample() * vae.config.scaling_factor
            latent_cache.append({
                "latent": lat.squeeze(0).to("cpu", torch.float32),
                "caption": caption,
                "size": (h, w),
                "repeats": repeats_from_folder(path, train_dir),
            })
    vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Cached %d latents at up to %dpx", len(latent_cache), resolution)

    # ---- Attach LoRA -------------------------------------------------------
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    rank = max(1, int(args.network_dim))
    alpha = float(args.network_alpha)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=LORA_TARGETS))
    trainable = [p for p in unet.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.to(torch.float32)
    n_params = sum(p.numel() for p in trainable)
    logger.info("LoRA rank %d alpha %g on %s -> %.2fM trainable params", rank, alpha, LORA_TARGETS, n_params / 1e6)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()

    if args.optimizer_type.lower() == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable, lr=args.learning_rate, weight_decay=1e-2)
        except Exception as exc:
            logger.warning("bitsandbytes unavailable (%s); using torch AdamW", exc)
            optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)

    # ---- Schedule ----------------------------------------------------------
    order: List[int] = []
    for idx, entry in enumerate(latent_cache):
        order.extend([idx] * int(entry["repeats"]))
    steps_per_epoch = max(1, math.ceil(len(order) / max(1, args.train_batch_size)))
    total_steps = int(args.max_train_steps) if args.max_train_steps > 0 else steps_per_epoch * max(1, args.max_train_epochs)
    logger.info("Training for %d steps (%d samples/epoch incl. repeats, batch %d)", total_steps, len(order), args.train_batch_size)

    use_scaler = args.mixed_precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    autocast_ctx = (lambda: torch.autocast("cuda", dtype=weight_dtype)) if device.type == "cuda" and weight_dtype != torch.float32 else (lambda: torch.autocast("cpu", enabled=False))

    def batch_for(indices: List[int]):
        first = latent_cache[indices[0]]
        same = [i for i in indices if latent_cache[i]["size"] == first["size"]]
        lat = torch.stack([latent_cache[i]["latent"] for i in same]).to(device, dtype=weight_dtype)
        pe = torch.cat([text_cache[latent_cache[i]["caption"]][0] for i in same]).to(device, dtype=weight_dtype)
        pooled = torch.cat([text_cache[latent_cache[i]["caption"]][1] for i in same]).to(device, dtype=weight_dtype)
        h, w = first["size"]
        time_ids = torch.tensor([[h, w, 0, 0, h, w]] * len(same), device=device, dtype=weight_dtype)
        return lat, pe, pooled, time_ids

    # Fixed evaluation batch: identical latents, noise and timesteps before and
    # after training.  A lower loss afterwards is direct evidence that the
    # adapter learned from this dataset.
    eval_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    eval_items = [latent_cache[i] for i in range(min(4, len(latent_cache)))]
    eval_batch = []
    for entry in eval_items:
        lat = entry["latent"].unsqueeze(0)
        eval_batch.append((
            lat,
            torch.randn(lat.shape, generator=eval_gen),
            torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), generator=eval_gen),
            entry["caption"], entry["size"],
        ))

    def fixed_batch_loss() -> float:
        unet.eval()
        total = 0.0
        with torch.no_grad():
            for lat, noise, t, caption, (h, w) in eval_batch:
                lat_d, noise_d, t_d = lat.to(device, weight_dtype), noise.to(device, weight_dtype), t.to(device)
                pe = text_cache[caption][0].to(device, weight_dtype)
                pooled = text_cache[caption][1].to(device, weight_dtype)
                time_ids = torch.tensor([[h, w, 0, 0, h, w]], device=device, dtype=weight_dtype)
                noisy = noise_scheduler.add_noise(lat_d, noise_d, t_d)
                with autocast_ctx():
                    pred = unet(noisy, t_d, encoder_hidden_states=pe,
                                added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                total += float(F.mse_loss(pred.float(), noise_d.float()).item())
        unet.train()
        return total / max(1, len(eval_batch))

    loss_before = fixed_batch_loss()
    logger.info("Fixed-batch loss before training: %.5f", loss_before)

    step = 0
    avg = None
    losses: List[float] = []
    while step < total_steps:
        random.shuffle(order)
        for b in range(0, len(order), args.train_batch_size):
            if step >= total_steps:
                break
            lat, pe, pooled, time_ids = batch_for(order[b:b + args.train_batch_size])
            noise = torch.randn_like(lat)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (lat.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(lat, noise, timesteps)
            with autocast_ctx():
                pred = unet(noisy, timesteps, encoder_hidden_states=pe,
                            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                loss = F.mse_loss(pred.float(), noise.float())
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            step += 1
            val = float(loss.item())
            if not math.isfinite(val):
                logger.error("Non-finite loss at step %d; aborting.", step)
                sys.exit(4)
            losses.append(val)
            avg = val if avg is None else 0.9 * avg + 0.1 * val
            print(f"{step}/{total_steps} loss={val:.4f} avr_loss={avg:.4f}", flush=True)

    loss_after = fixed_batch_loss()
    logger.info("Fixed-batch loss after training: %.5f (before %.5f)", loss_after, loss_before)
    print(f"fixed_batch_loss_before={loss_before:.6f} fixed_batch_loss_after={loss_after:.6f}", flush=True)

    # ---- Save --------------------------------------------------------------
    peft_sd = get_peft_model_state_dict(unet)
    tensors = to_kohya_state_dict(peft_sd, alpha, torch)
    up_norm = sum(float(v.float().norm()) for k, v in tensors.items() if k.endswith("lora_up.weight"))
    meta = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": "stable-diffusion-xl-v1-base/lora",
        "modelspec.implementation": "diffusers",
        "modelspec.title": args.output_name,
        "modelspec.description": "Trained with Geekatplay LoRA Maker (Diffusers/PEFT engine)",
        "modelspec.author": "Geekatplay Studio - Vladimir Chopine",
        "modelspec.date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modelspec.resolution": f"{resolution}x{resolution}",
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(alpha),
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_sd_model_name": model_path.name,
        "ss_learning_rate": str(args.learning_rate),
        "ss_total_batch_size": str(args.train_batch_size),
        "ss_max_train_steps": str(total_steps),
        "ss_steps": str(total_steps),
        "ss_num_train_images": str(len(samples)),
        "ss_optimizer": type(optimizer).__name__,
        "ss_mixed_precision": args.mixed_precision,
        "ss_training_engine": "Diffusers/PEFT",
        "lm_fixed_batch_loss_before": f"{loss_before:.6f}",
        "lm_fixed_batch_loss_after": f"{loss_after:.6f}",
    }
    save_file(tensors, str(target), metadata=meta)
    size_mb = target.stat().st_size / 1024**2
    logger.info("Saved %s (%.2f MB, %d tensors, %d adapter pairs, up-weight norm %.4f, first loss %.4f, last loss %.4f, %.0fs)",
                target, size_mb, len(tensors), sum(k.endswith("lora_down.weight") for k in tensors),
                up_norm, losses[0], losses[-1], time.time() - t0)
    print(f"Completed saving model to {target}", flush=True)


if __name__ == "__main__":
    main()
