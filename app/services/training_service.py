"""Honest local LoRA training orchestration.

A completed job is a matching Kohya/sd-scripts or PyTorch/Diffusers process that produced
a verified safetensors adapter. This module never substitutes another model or fabricates
weights, loss values, or a successful completion.
"""
import json
import logging
import hashlib
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class TrainingService:
    _active_jobs: Dict[str, Dict[str, Any]] = {}
    _SCRIPT_ROOTS = (
        Path("D:/ComfyUI/lora-training/fluxgym/sd-scripts"),
        Path("D:/kohya_ss/sd-scripts"),
    )
    _TRAINERS = {
        "sdxl-1.0": {
            "name": "SDXL 1.0 (Stable Diffusion XL)",
            "category": "image",
            "script": "sdxl_train_network.py",
            "network_module": "networks.lora",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors",
                "D:/ComfyUI/ComfyUI/models/checkpoints/juggernautXL_version6Rundiffusion.safetensors",
                "D:/ComfyUI/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors",
            ),
        },
        "flux-1-dev": {
            "name": "FLUX.1 [dev] 24B",
            "category": "image",
            "script": "flux_train_network.py",
            "network_module": "networks.lora_flux",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/unet/flux1-dev.safetensors",
                "D:/ComfyUI/models/unet/flux1-dev.safetensors",
            ),
            "components": {
                "clip_l": (
                    "D:/ComfyUI/ComfyUI/models/clip/clip_l.safetensors",
                    "D:/ComfyUI/models/clip/clip_l.safetensors",
                ),
                "t5xxl": (
                    "D:/ComfyUI/ComfyUI/models/clip/t5xxl_fp16.safetensors",
                    "D:/ComfyUI/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors",
                    "D:/ComfyUI/ComfyUI/models/clip/oldt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "D:/ComfyUI/models/clip/t5xxl_fp16.safetensors",
                ),
                "ae": (
                    "D:/ComfyUI/ComfyUI/models/vae/ae.safetensors",
                    "D:/ComfyUI/models/vae/ae.safetensors",
                ),
            },
        },
        "minimax-video": {
            "name": "MiniMax / Hunyuan Video 720p",
            "category": "video",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors",
                "D:/ComfyUI/models/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors",
            ),
            "components": {
                "vae": (
                    "D:/ComfyUI/ComfyUI/models/vae/hunyuan_video_vae_bf16.safetensors",
                    "D:/ComfyUI/models/vae/hunyuan_video_vae_bf16.safetensors",
                ),
                "clip": (
                    "D:/ComfyUI/ComfyUI/models/clip/clip_l.safetensors",
                    "D:/ComfyUI/models/clip/clip_l.safetensors",
                ),
            },
        },
        "wan-2.1-t2v": {
            "name": "Wan 2.1 Video T2V",
            "category": "video",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp8.safetensors",
                "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
                "D:/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp8.safetensors",
            ),
        },
        "ltx-video-turbo": {
            "name": "LTX-Video 2.5 Turbo",
            "category": "video",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.1.safetensors",
                "D:/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.1.safetensors",
            ),
        },
        "qwen-image": {
            "name": "Qwen2.5-VL / Qwen-Image",
            "category": "image",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/qwen2.5-vl-7b.safetensors",
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ),
        },
        "z-image": {
            "name": "Z-Image DiT",
            "category": "image",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/z-image-dit.safetensors",
                "Tongyi-MAI/Z-Image",
                "ZhipuAI/Z-Image-DiT",
            ),
        },
    }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _first_file(cls, candidates) -> Optional[Path]:
        if isinstance(candidates, (str, Path)):
            candidates = [candidates]
        from app.services.settings_service import SettingsService
        all_roots = SettingsService.get_all_model_directories()
        user_cfg = SettingsService.get_user_settings()
        download_root = Path(user_cfg.get("DOWNLOAD_DIR", "D:/ComfyUI/ComfyUI/models"))

        for x in candidates:
            p = Path(x)
            if p.is_file() or p.is_dir():
                return p
            if not p.is_absolute():
                for m_root in all_roots:
                    candidate_in_models = m_root / x
                    if candidate_in_models.is_file() or candidate_in_models.is_dir():
                        return candidate_in_models

            if "/" in str(x) and not Path(x).is_absolute():
                hf_name = "models--" + str(x).replace("/", "--")
                possible_hubs = [
                    download_root / "huggingface" / "hub" / hf_name / "snapshots",
                    Path.home() / ".cache" / "huggingface" / "hub" / hf_name / "snapshots"
                ]
                for m_root in all_roots:
                    possible_hubs.append(m_root / "huggingface" / "hub" / hf_name / "snapshots")
                for hf_hub in possible_hubs:
                    if hf_hub.exists():
                        for snap in sorted(hf_hub.iterdir(), reverse=True):
                            if snap.is_dir():
                                return snap
        return None

    @classmethod
    def _script(cls, filename: str) -> Optional[Path]:
        return next((root / filename for root in cls._SCRIPT_ROOTS if (root / filename).is_file()), None)

    @classmethod
    def get_training_capabilities(cls) -> Dict[str, Any]:
        from app.services.hardware_service import HardwareService
        hw = HardwareService.get_hardware_profile()
        architectures = []
        unavailable = []

        for model_id, spec in cls._TRAINERS.items():
            script = cls._script(spec["script"]) if "script" in spec else None
            checkpoint = cls._first_file(spec.get("checkpoints", ()))
            missing = [] if hw["cuda_available"] else ["CUDA GPU unavailable"]

            if "script" in spec and not script:
                missing.append(f"missing trainer script: {spec['script']}")
            if not checkpoint:
                missing.append(f"missing base weights: {spec.get('name', model_id)}")

            if "components" in spec:
                for comp_name, comp_candidates in spec["components"].items():
                    resolved_comp = cls._first_file(comp_candidates)
                    if not resolved_comp:
                        missing.append(f"missing component: {comp_name}")

            is_avail = (len(missing) == 0)
            arch_entry = {
                "id": model_id,
                "name": spec.get("name", model_id),
                "category": spec.get("category", "image"),
                "available": is_avail,
                "trainer_script": str(script.resolve()) if script else None,
                "base_checkpoint": str(checkpoint.resolve()) if checkpoint else None,
                "missing_requirements": missing,
            }
            architectures.append(arch_entry)
            if not is_avail:
                unavailable.append({
                    "id": model_id,
                    "available": False,
                    "reason": "; ".join(missing),
                })

        return {
            "success": True,
            "cuda_available": hw["cuda_available"],
            "device": hw["device_name"],
            "vram_gb": hw["total_vram_gb"],
            "tier_name": hw["tier_name"],
            "architectures": architectures,
            "unavailable_architectures": unavailable,
        }

    @classmethod
    def _backend(cls, base_model: str) -> Dict[str, Any]:
        if base_model not in cls._TRAINERS:
            raise ValueError(f"'{base_model}' has no registered trainer integration. Supported: {', '.join(cls._TRAINERS)}.")
        spec = cls._TRAINERS[base_model]
        script = cls._script(spec["script"]) if "script" in spec else None
        checkpoint = cls._first_file(spec["checkpoints"])
        from app.services.hardware_service import HardwareService
        hw = HardwareService.get_hardware_profile()
        missing = [] if hw["cuda_available"] else ["CUDA unavailable"]
        if "script" in spec and not script:
            missing.append(f"trainer script {spec['script']} not found")
        if not checkpoint:
            missing.append(f"compatible base checkpoint for {spec.get('name', base_model)} not found")
        components = {}
        for k, v in spec.get("components", {}).items():
            resolved = cls._first_file(v)
            if not resolved or not resolved.is_file():
                missing.append(f"component missing: {k}")
            else:
                components[k] = resolved
        if missing:
            raise ValueError("Cannot start real training: " + "; ".join(missing))
        return {"spec": spec, "script": script, "checkpoint": checkpoint, "components": components}

    @classmethod
    def generate_kohya_toml_config(
        cls,
        project_dir: str,
        base_model: str = "sdxl-1.0",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        learning_rate: float = 1e-4,
        max_train_epochs: int = 10,
        batch_size: int = 1,
        resolution: int = 1024,
    ) -> str:
        """Write an audit manifest; recorded CLI is the actual executable config."""
        p = Path(project_dir)
        meta = DatasetService.load_project_meta(project_dir) or {}
        output = p / "Training" / "output"
        output.mkdir(parents=True, exist_ok=True)
        is_supported = base_model in ["sdxl-1.0", "flux-1-dev"] and cls._first_file(cls._TRAINERS.get(base_model, {}).get("checkpoints", ())) is not None
        manifest = {
            "format": "lora-maker-run-manifest-v1",
            "base_model": base_model,
            "trainer_supported": is_supported,
            "dataset_dir": str((p / "Training" / "img").resolve()),
            "output_dir": str(output.resolve()),
            "output_name": f"{meta.get('character_name', 'character')}_{base_model}_lora",
            "network_dim": lora_rank,
            "network_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "max_train_epochs": max_train_epochs,
            "train_batch_size": batch_size,
            "resolution": resolution,
        }
        path = p / "Training" / "kohya_lora_config.toml"
        path.write_text(
            "# Audit manifest only; real command is saved separately.\n"
            + "\n".join(f"{k} = {json.dumps(v)}" for k, v in manifest.items())
            + "\n",
            encoding="utf-8",
        )
        return str(path.resolve())

    @classmethod
    def generate_comfyui_workflow(
        cls,
        project_dir: str,
        base_model: str = "sdxl-1.0",
        lora_rank: int = 16,
        lora_alpha: int = 16,
    ) -> str:
        """Create an explicit handoff, never an invented executable ComfyUI graph."""
        path = Path(project_dir) / "Training" / "comfyui_training_workflow.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "format": "lora-maker-comfyui-handoff-v1",
                    "executable": False,
                    "base_model": base_model,
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "message": "No installed ComfyUI trainer nodes were verified. Use the generated Kohya command.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(path.resolve())

    @classmethod
    def _command(
        cls,
        backend: Dict[str, Any],
        image_dir: Path,
        output_dir: Path,
        output_name: str,
        base_model: str,
        rank: int,
        alpha: int,
        lr: float,
        epochs: int,
        batch: int,
    ) -> List[str]:
        cmd = [
            sys.executable,
            str(backend["script"]),
            f"--pretrained_model_name_or_path={backend['checkpoint']}",
            f"--train_data_dir={image_dir}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            "--caption_extension=.txt",
            "--resolution=1024,1024",
            f"--network_dim={rank}",
            f"--network_alpha={alpha}",
            f"--network_module={backend['spec']['network_module']}",
            "--network_train_unet_only",
            f"--learning_rate={lr}",
            "--optimizer_type=AdamW8bit",
            "--mixed_precision=fp16",
            "--save_precision=fp16",
            f"--max_train_epochs={epochs}",
            f"--train_batch_size={batch}",
            "--max_data_loader_n_workers=0",
            "--gradient_checkpointing",
        ]
        if base_model == "flux-1-dev":
            cmd += [
                f"--clip_l={backend['components']['clip_l']}",
                f"--t5xxl={backend['components']['t5xxl']}",
                f"--ae={backend['components']['ae']}",
                "--cache_text_encoder_outputs",
            ]
        return cmd

    @classmethod
    def _verify(cls, model: Path, base_model: str) -> Dict[str, Any]:
        if not model.is_file() or model.stat().st_size < 1024 * 1024:
            raise RuntimeError("Trainer did not produce a substantial safetensors LoRA.")
        from safetensors import safe_open
        with safe_open(str(model), framework="pt", device="cpu") as f:
            keys, metadata = list(f.keys()), (f.metadata() or {})
        pairs = sum(x.endswith(".lora_down.weight") for x in keys)
        if pairs < 10:
            raise RuntimeError("Output lacks sufficient LoRA tensor pairs.")
        architecture = metadata.get("modelspec.architecture", "").lower()
        if "sdxl" in base_model:
            expected = "stable-diffusion-xl"
        elif "flux" in base_model:
            expected = "flux"
        else:
            expected = ""
        if expected and expected not in architecture:
            raise RuntimeError(f"Output metadata is not compatible with {base_model}: {architecture or 'missing'}")
        return {"tensor_count": len(keys), "adapter_pairs": pairs, "architecture": architecture}

    @classmethod
    def start_training_pipeline(
        cls,
        project_dir: str,
        base_model: str = "sdxl-1.0",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        epochs: int = 5,
        repeats: int = 10,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        execution_mode: str = "real_gpu",
        framing_mode: str = "bucket",
    ) -> Dict[str, Any]:
        if execution_mode not in {"real_gpu", "validate_only", "dry_run"}:
            raise ValueError("execution_mode must be 'real_gpu' or 'validate_only'")
        if min(lora_rank, lora_alpha, epochs, repeats, batch_size) < 1 or learning_rate <= 0:
            raise ValueError("training parameters must be positive")
        p = Path(project_dir).resolve()
        dataset = DatasetService.get_project_dataset(str(p))
        if dataset["total_count"] < 2 or dataset["captioned_count"] != dataset["total_count"]:
            raise ValueError("Real training requires at least two valid, fully captioned keyframes.")

        from PIL import Image
        valid_images = 0
        for image in (p / "Keyframes_Out").glob("*"):
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                with Image.open(image) as opened:
                    opened.verify()
                valid_images += 1
            except Exception:
                continue
        if valid_images < 2:
            raise ValueError("Real training requires at least two valid, fully captioned keyframes.")

        backend = cls._backend(base_model)
        exported = DatasetService.export_for_kohya(
            str(p),
            repeats=repeats,
            class_token="character",
            target_resolution=1024,
            framing_mode=framing_mode,
        )
        if exported["exported_frames"] < 2:
            raise ValueError("Dataset export did not produce two usable image/caption pairs.")

        meta = DatasetService.load_project_meta(str(p)) or {}
        output_dir = p / "Training" / "output"
        name = f"{meta.get('character_name', 'character')}_{base_model}_lora"
        target = output_dir / f"{name}.safetensors"
        command = cls._command(
            backend,
            p / "Training" / "img",
            output_dir,
            name,
            base_model,
            lora_rank,
            lora_alpha,
            learning_rate,
            epochs,
            batch_size,
        )
        command_file = p / "Training" / "real_training_command.json"
        command_file.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(backend["script"].parent) if backend.get("script") else str(p),
                    "generated_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        job = cls._active_jobs[str(p)] = {
            "status": "running",
            "execution_mode": "validate_only" if execution_mode == "dry_run" else execution_mode,
            "base_model": base_model,
            "trainer_script": str(backend["script"]) if backend.get("script") else "diffusers_peft",
            "base_checkpoint": str(backend["checkpoint"]),
            "target_model_file": str(target),
            "command_file": str(command_file),
            "progress_percent": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "current_loss": None,
            "loss_history": [],
            "log": ["Preflight passed: matching trainer, checkpoint, CUDA, captions, and dataset export verified."],
        }
        config = cls.generate_kohya_toml_config(str(p), base_model, lora_rank, lora_alpha, learning_rate, epochs, batch_size)
        handoff = cls.generate_comfyui_workflow(str(p), base_model, lora_rank, lora_alpha)

        if execution_mode in {"validate_only", "dry_run"}:
            job.update({"status": "validated", "progress_percent": 100.0})
            job["log"].append("Validation complete; no model weights were created.")
            return {
                "status": "validated",
                "execution_mode": "validate_only",
                "project_dir": str(p),
                "base_model": base_model,
                "config_toml": config,
                "comfyui_workflow": handoff,
                "training_command": str(command_file),
                "target_output": str(target),
            }

        def worker():
            started = time.time()
            try:
                env = os.environ.copy()
                env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "TF_ENABLE_ONEDNN_OPTS": "0"})
                cwd_dir = str(backend["script"].parent) if backend.get("script") else str(p)
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=cwd_dir,
                    env=env,
                )
                pattern = re.compile(r"(\d+)/(\d+).*?(?:avr_loss|loss)[=:\s]+([0-9.]+)", re.I)
                for raw in iter(proc.stdout.readline, ""):
                    line = raw.strip()
                    found = pattern.search(line)
                    if found:
                        step, total, loss = int(found.group(1)), int(found.group(2)), float(found.group(3))
                        job.update({
                            "current_step": step,
                            "total_steps": total,
                            "current_loss": loss,
                            "progress_percent": round(100 * step / max(total, 1), 1),
                        })
                        job["loss_history"].append({"step": step, "loss": loss})
                    if any(x in line.lower() for x in ("error", "saving", "epoch", "loading")) and len(line) < 500:
                        job["log"].append(line)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Trainer exited with code {proc.returncode}")
                verification = cls._verify(target, base_model)
                metrics = {
                    "schema": "lora-maker-training-metrics-v1",
                    "verified": True,
                    "base_model": base_model,
                    "base_checkpoint": str(backend["checkpoint"]),
                    "base_checkpoint_sha256": cls._sha256(backend["checkpoint"]),
                    "trainer_script": str(backend["script"]) if backend.get("script") else "diffusers_peft",
                    "training_engine": "Kohya sd-scripts" if backend.get("script") else "Diffusers/PEFT",
                    "command_file": str(command_file),
                    "command_sha256": cls._sha256(command_file),
                    "exit_code": proc.returncode,
                    "output_sha256": cls._sha256(target),
                    "total_steps": job["current_step"],
                    "loss_history": job["loss_history"],
                    "initial_loss": job["loss_history"][0]["loss"] if job["loss_history"] else None,
                    "final_loss": job["current_loss"],
                    "duration_seconds": round(time.time() - started, 1),
                    "output_size_mb": round(target.stat().st_size / 1024**2, 2),
                    "verification": verification,
                }
                target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                job.update({
                    "status": "completed",
                    "progress_percent": 100.0,
                    "artifact_verified": True,
                    "verification": verification,
                })
                job["log"].append("Completed: model passed safetensors and architecture verification.")
            except Exception as exc:
                logger.exception("Real LoRA training failed")
                job.update({"status": "failed", "error": str(exc)})
                job["log"].append(f"Training failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()
        return {
            "status": "started",
            "execution_mode": "real_gpu",
            "project_dir": str(p),
            "base_model": base_model,
            "config_toml": config,
            "comfyui_workflow": handoff,
            "training_command": str(command_file),
            "target_output": str(target),
        }

    @classmethod
    def get_job_status(cls, project_dir: str) -> Dict[str, Any]:
        return cls._active_jobs.get(
            str(Path(project_dir).resolve()),
            {
                "status": "idle",
                "progress_percent": 0.0,
                "current_step": 0,
                "total_steps": 0,
                "current_loss": None,
                "loss_history": [],
                "log": [],
            },
        )
