#!/usr/bin/env python3
"""Prove that LoRA training really runs on this machine.

Creates a tiny synthetic captioned dataset, runs the built-in Diffusers/PEFT
trainer for a handful of real optimizer steps, then checks that the output:

  * is a safetensors file in Kohya / ComfyUI key format,
  * has non-zero "up" projection tensors (they start at zero; only gradient
    descent changes them),
  * loads back through diffusers ``load_lora_weights``.

Usage
-----
  python verify_training.py                 # uses your SDXL checkpoint if found, GPU
  python verify_training.py --tiny          # 20 MB test model from Hugging Face, CPU OK
  python verify_training.py --checkpoint "D:/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"

Exit code 0 means training is verified.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
WORKER = ROOT / "app" / "services" / "diffusers_lora_worker.py"
TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


def make_dataset(root: Path, n: int = 4) -> Path:
    from PIL import Image, ImageDraw

    concept = root / "2_zorb character"
    concept.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        im = Image.new("RGB", (640, 480), (30 + 40 * i, 80, 200 - 30 * i))
        d = ImageDraw.Draw(im)
        d.ellipse((120 + 30 * i, 100, 420 + 30 * i, 400), fill=(250, 200, 50))
        d.rectangle((200, 300, 360, 420), fill=(20, 20, 20))
        im.save(concept / f"zorb_{i:02d}.png")
        (concept / f"zorb_{i:02d}.txt").write_text(f"zorb, a yellow blob character, pose {i}", encoding="utf-8")
    return root


def find_sdxl_checkpoint() -> str | None:
    try:
        from app.services.training_service import TrainingService

        found = TrainingService._first_file(TrainingService._TRAINERS["sdxl-1.0"]["checkpoints"])
        return str(found) if found else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="SDXL single-file checkpoint or diffusers folder")
    ap.add_argument("--tiny", action="store_true", help="Use the Hugging Face tiny SDXL test pipeline (CPU friendly)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--keep", action="store_true", help="Keep the temporary dataset/output folder")
    args = ap.parse_args()

    import torch

    cuda = torch.cuda.is_available()
    print(f"[verify] torch {torch.__version__} | CUDA: {cuda}" + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))

    checkpoint = args.checkpoint
    if not checkpoint and not args.tiny:
        checkpoint = find_sdxl_checkpoint()
        if not checkpoint:
            print("[verify] No SDXL checkpoint found in configured model folders; falling back to --tiny test model.")
            args.tiny = True
    if args.tiny:
        checkpoint = TINY_MODEL
    if not cuda and not args.tiny:
        print("[verify] No CUDA GPU: a full SDXL checkpoint cannot be trained on CPU. Use --tiny to verify the pipeline.")
        return 2

    steps = args.steps or (6 if args.tiny else 20)
    resolution = args.resolution or (64 if args.tiny else 512)
    mp = "no" if (args.tiny or not cuda) else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")

    work = Path(tempfile.mkdtemp(prefix="loramaker_verify_"))
    dataset = make_dataset(work / "img")
    out_dir = work / "output"
    name = "verify_sdxl-1.0_lora"
    cmd = [
        sys.executable, str(WORKER),
        f"--pretrained_model_name_or_path={checkpoint}",
        f"--train_data_dir={dataset}",
        f"--output_dir={out_dir}",
        f"--output_name={name}",
        "--network_dim=4", "--network_alpha=4",
        "--learning_rate=1e-4",
        f"--max_train_steps={steps}",
        "--train_batch_size=1",
        f"--resolution={resolution},{resolution}",
        f"--mixed_precision={mp}",
        "--gradient_checkpointing",
        "--allow_cpu",
    ]
    print(f"[verify] model: {checkpoint}")
    print(f"[verify] {steps} steps @ {resolution}px, precision {mp}")
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, cwd=str(ROOT))
    losses = [l for l in proc.stdout.splitlines() if "avr_loss=" in l]
    for line in losses[:3] + (["..."] if len(losses) > 6 else []) + losses[-3:]:
        print("   ", line)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        print(f"[verify] FAILED: trainer exited with code {proc.returncode}")
        return 1
    elapsed = time.time() - t0

    target = out_dir / f"{name}.safetensors"
    from app.services.training_service import TrainingService

    try:
        info = TrainingService._verify(target, "sdxl-1.0")
    except Exception as exc:
        print(f"[verify] FAILED: {exc}")
        return 1
    print(f"[verify] artifact: {target.name} ({target.stat().st_size / 1024:.0f} KB) {json.dumps(info)}")

    # Load back through diffusers to prove ComfyUI/A1111 compatibility of the key format.
    try:
        from diffusers import StableDiffusionXLPipeline

        if Path(checkpoint).is_file():
            pipe = StableDiffusionXLPipeline.from_single_file(checkpoint, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(checkpoint)
        pipe.load_lora_weights(str(target), adapter_name="verify")
        n_mod = sum(1 for n, _ in pipe.unet.named_modules() if n.endswith("lora_A"))
        print(f"[verify] diffusers load_lora_weights OK: {n_mod} UNet modules carry the adapter")
    except Exception as exc:
        print(f"[verify] FAILED: trained LoRA did not load in diffusers: {exc}")
        return 1

    print(f"[verify] PASS: real gradient-descent training verified in {elapsed:.0f}s ({len(losses)} logged steps).")
    if args.keep:
        print(f"[verify] kept: {work}")
    else:
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
