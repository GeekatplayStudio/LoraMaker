"""Honest local LoRA training orchestration.

A completed job is a matching Kohya/sd-scripts or Diffusers/PEFT process that
produced a verified safetensors adapter. This module never substitutes another
model or fabricates weights, loss values, or a successful completion.

Engines
-------
* ``Kohya sd-scripts`` - preferred when an sd-scripts checkout is found. Used for
  SDXL and FLUX.  The trainer is launched with the Kohya venv interpreter when
  one exists next to the checkout, so its own dependencies are used.
* ``Diffusers/PEFT`` - the built-in ``diffusers_lora_worker.py``.  Real SDXL
  training with no external installation.  Used for SDXL when Kohya is absent.

Architectures without a verified trainer are reported as unavailable instead of
being "trained" by a stand-in.
"""
import json
import logging
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.dataset_service import DatasetService
from app.services.training_presets import DEFAULT_PRESET, PRESETS, list_presets, resolve_preset

logger = logging.getLogger(__name__)

ENGINE_KOHYA = "Kohya sd-scripts"
ENGINE_DIFFUSERS = "Diffusers/PEFT"


class TrainingService:
    _active_jobs: Dict[str, Dict[str, Any]] = {}
    _active_procs: Dict[str, Any] = {}
    _bnb_cache: Dict[str, bool] = {}
    _TRAINERS = {
        "sdxl-1.0": {
            "name": "SDXL 1.0 (Stable Diffusion XL)",
            "category": "image",
            "script": "sdxl_train_network.py",
            "network_module": "networks.lora",
            "builtin_worker": True,
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors",
                "D:/ComfyUI/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors",
                "D:/ComfyUI/ComfyUI/models/checkpoints/juggernautXL_version6Rundiffusion.safetensors",
                "D:/ComfyUI/models/checkpoints/sd_xl_base_1.0_0.9vae.safetensors",
                "checkpoints/sd_xl_base_1.0_0.9vae.safetensors",
                "checkpoints/sd_xl_base_1.0.safetensors",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ),
        },
        "flux-1-dev": {
            "name": "FLUX.1 [dev] 12B",
            "category": "image",
            "script": "flux_train_network.py",
            "network_module": "networks.lora_flux",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/unet/flux1-dev.safetensors",
                "D:/ComfyUI/models/unet/flux1-dev.safetensors",
                "unet/flux1-dev.safetensors",
                "diffusion_models/flux1-dev.safetensors",
            ),
            "components": {
                "clip_l": (
                    "D:/ComfyUI/ComfyUI/models/clip/clip_l.safetensors",
                    "D:/ComfyUI/models/clip/clip_l.safetensors",
                    "clip/clip_l.safetensors",
                    "text_encoders/clip_l.safetensors",
                ),
                "t5xxl": (
                    "D:/ComfyUI/ComfyUI/models/clip/t5xxl_fp16.safetensors",
                    "D:/ComfyUI/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors",
                    "D:/ComfyUI/ComfyUI/models/clip/oldt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "D:/ComfyUI/models/clip/t5xxl_fp16.safetensors",
                    "clip/t5xxl_fp16.safetensors",
                    "clip/t5xxl_fp8_e4m3fn.safetensors",
                    "text_encoders/t5xxl_fp16.safetensors",
                    "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
                ),
                "ae": (
                    "D:/ComfyUI/ComfyUI/models/vae/ae.safetensors",
                    "D:/ComfyUI/models/vae/ae.safetensors",
                    "vae/ae.safetensors",
                ),
            },
        },
        "minimax-video": {
            "name": "MiniMax / Hunyuan Video 720p",
            "category": "video",
            "network_module": "networks.lora",
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
            "network_module": "networks.lora",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp8.safetensors",
                "D:/ComfyUI/ComfyUI/models/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
                "D:/ComfyUI/models/diffusion_models/wan2.1_t2v_14B_fp8.safetensors",
            ),
        },
        "ltx-video-turbo": {
            "name": "LTX-Video 2.5 Turbo",
            "category": "video",
            "network_module": "networks.lora",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.1.safetensors",
                "D:/ComfyUI/models/checkpoints/ltx-video-2b-v0.9.1.safetensors",
            ),
        },
        "qwen-image": {
            "name": "Qwen2.5-VL / Qwen-Image",
            "category": "image",
            "network_module": "networks.lora",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/qwen2.5-vl-7b.safetensors",
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ),
        },
        "z-image": {
            "name": "Z-Image DiT",
            "category": "image",
            "network_module": "networks.lora",
            "checkpoints": (
                "D:/ComfyUI/ComfyUI/models/checkpoints/z-image-dit.safetensors",
                "Tongyi-MAI/Z-Image",
                "ZhipuAI/Z-Image-DiT",
            ),
        },
    }

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _sha256(path: Path) -> str:
        p = Path(path)
        if not p.exists():
            return "not_found"
        if p.is_dir():
            cfg = p / "config.json"
            if cfg.is_file():
                return TrainingService._sha256(cfg)
            return hashlib.sha256(str(p.resolve()).encode("utf-8")).hexdigest()
        digest = hashlib.sha256()
        with p.open("rb") as handle:
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
                    Path.home() / ".cache" / "huggingface" / "hub" / hf_name / "snapshots",
                ]
                hf_home = os.environ.get("HF_HOME")
                if hf_home:
                    possible_hubs.append(Path(hf_home) / "hub" / hf_name / "snapshots")
                for m_root in all_roots:
                    possible_hubs.append(m_root / "huggingface" / "hub" / hf_name / "snapshots")
                for hf_hub in possible_hubs:
                    if hf_hub.exists():
                        for snap in sorted(hf_hub.iterdir(), reverse=True):
                            if snap.is_dir() and (snap / "model_index.json").is_file():
                                return snap
        return None

    @classmethod
    def script_roots(cls) -> List[Path]:
        """Candidate sd-scripts directories: settings, env, and common layouts."""
        roots: List[Path] = []
        try:
            from app.core.config import settings
            for raw in (settings.KOHYA_ROOT, settings.KOHYA_FALLBACK_ROOT):
                if raw:
                    roots.append(Path(raw))
        except Exception:
            pass
        env_root = os.environ.get("KOHYA_SD_SCRIPTS_DIR")
        if env_root:
            roots.insert(0, Path(env_root))
        roots += [
            Path("D:/kohya_ss/sd-scripts"),
            Path("D:/ComfyUI/lora-training/fluxgym/sd-scripts"),
            Path("C:/kohya_ss/sd-scripts"),
            Path.home() / "kohya_ss" / "sd-scripts",
            Path.home() / "sd-scripts",
        ]
        expanded: List[Path] = []
        for r in roots:
            for cand in (r, r / "sd-scripts"):
                if cand not in expanded:
                    expanded.append(cand)
        return expanded

    @classmethod
    def _script(cls, filename: str) -> Optional[Path]:
        return next((root / filename for root in cls.script_roots() if (root / filename).is_file()), None)

    @classmethod
    def _trainer_python(cls, script: Optional[Path]) -> str:
        """Interpreter for the trainer: the Kohya venv when present, else ours."""
        if script:
            for base in (script.parent, script.parent.parent):
                for rel in ("venv/Scripts/python.exe", "venv/bin/python", ".venv/Scripts/python.exe", ".venv/bin/python"):
                    cand = base / rel
                    if cand.is_file():
                        return str(cand)
        return sys.executable

    @classmethod
    def _bitsandbytes_available(cls, python: str) -> bool:
        if python in cls._bnb_cache:
            return cls._bnb_cache[python]
        ok = False
        try:
            res = subprocess.run(
                [python, "-c", "import bitsandbytes, torch; assert torch.cuda.is_available()"],
                capture_output=True, timeout=60,
            )
            ok = res.returncode == 0
        except Exception:
            ok = False
        cls._bnb_cache[python] = ok
        return ok

    @staticmethod
    def cpu_training_allowed() -> bool:
        """Opt-in for CPU runs of the built-in trainer (tests, smoke checks)."""
        return os.environ.get("LORAMAKER_ALLOW_CPU_TRAINING", "").strip().lower() in {"1", "true", "yes"}

    @classmethod
    def _worker_script(cls) -> Path:
        return Path(__file__).resolve().parent / "diffusers_lora_worker.py"

    # ---------------------------------------------------------- capabilities
    @classmethod
    def _resolve_backend(cls, base_model: str, hw: Dict[str, Any]) -> Dict[str, Any]:
        """Common resolution used by capabilities and by the preflight."""
        spec = cls._TRAINERS[base_model]
        script = cls._script(spec["script"]) if "script" in spec else None
        checkpoint = cls._first_file(spec.get("checkpoints", ()))
        missing: List[str] = []
        if not hw["cuda_available"] and not cls.cpu_training_allowed():
            missing.append("CUDA GPU unavailable")

        engine = None
        if script:
            engine = ENGINE_KOHYA
        elif spec.get("builtin_worker"):
            engine = ENGINE_DIFFUSERS
        else:
            needed = spec.get("script", "a Kohya sd-scripts trainer")
            missing.append(f"no verified trainer installed for {spec.get('name', base_model)} (needs {needed})")

        if not checkpoint:
            missing.append(f"missing base weights: {spec.get('name', base_model)}")

        components: Dict[str, Path] = {}
        for comp_name, comp_candidates in spec.get("components", {}).items():
            resolved = cls._first_file(comp_candidates)
            if not resolved or not resolved.is_file():
                missing.append(f"missing component: {comp_name}")
            else:
                components[comp_name] = resolved

        python = cls._trainer_python(script) if engine == ENGINE_KOHYA else sys.executable
        return {
            "spec": spec,
            "script": script,
            "checkpoint": checkpoint,
            "components": components,
            "engine": engine,
            "python": python,
            "missing": missing,
        }

    @classmethod
    def get_training_capabilities(cls) -> Dict[str, Any]:
        from app.services.hardware_service import HardwareService
        hw = HardwareService.get_hardware_profile()
        architectures = []
        unavailable = []

        for model_id in cls._TRAINERS:
            b = cls._resolve_backend(model_id, hw)
            is_avail = len(b["missing"]) == 0
            entry = {
                "id": model_id,
                "name": b["spec"].get("name", model_id),
                "category": b["spec"].get("category", "image"),
                "available": is_avail,
                "engine": b["engine"],
                "trainer_script": str(b["script"].resolve()) if b["script"] else None,
                "trainer_python": b["python"] if is_avail else None,
                "base_checkpoint": str(b["checkpoint"].resolve()) if b["checkpoint"] else None,
                "missing_requirements": b["missing"],
            }
            architectures.append(entry)
            if not is_avail:
                unavailable.append({"id": model_id, "available": False, "reason": "; ".join(b["missing"])})

        return {
            "success": True,
            "cuda_available": hw["cuda_available"],
            "device": hw["device_name"],
            "vram_gb": hw["total_vram_gb"],
            "tier_id": hw.get("tier_id"),
            "tier_name": hw["tier_name"],
            "kohya_roots_checked": [str(r) for r in cls.script_roots()],
            "presets": list_presets(),
            "default_preset": DEFAULT_PRESET,
            "architectures": architectures,
            "unavailable_architectures": unavailable,
        }

    @classmethod
    def get_preset_plan(cls, project_dir: str, base_model: str, preset: str = DEFAULT_PRESET) -> Dict[str, Any]:
        """Resolved parameters a preset would use for this project and GPU."""
        from app.services.hardware_service import HardwareService
        hw = HardwareService.get_hardware_profile()
        frames = 2
        try:
            frames = max(2, DatasetService.get_project_dataset(project_dir).get("captioned_count", 0))
        except Exception:
            pass
        python = sys.executable
        if base_model in cls._TRAINERS:
            try:
                python = cls._resolve_backend(base_model, hw)["python"]
            except Exception:
                pass
        plan = resolve_preset(preset, hw, frames, base_model, bitsandbytes_available=cls._bitsandbytes_available(python))
        plan["device_name"] = hw.get("device_name")
        plan["vram_tier"] = hw.get("tier_name")
        return plan

    @classmethod
    def _backend(cls, base_model: str) -> Dict[str, Any]:
        if base_model not in cls._TRAINERS:
            raise ValueError(f"'{base_model}' has no registered trainer integration. Supported: {', '.join(cls._TRAINERS)}.")
        from app.services.hardware_service import HardwareService
        hw = HardwareService.get_hardware_profile()
        b = cls._resolve_backend(base_model, hw)
        if b["missing"]:
            raise ValueError("Cannot start real training: " + "; ".join(b["missing"]))
        b["hardware"] = hw
        return b

    # -------------------------------------------------------------- manifests
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
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write an audit manifest; recorded CLI is the actual executable config."""
        p = Path(project_dir)
        meta = DatasetService.load_project_meta(project_dir) or {}
        output = p / "Training" / "output"
        output.mkdir(parents=True, exist_ok=True)
        spec = cls._TRAINERS.get(base_model, {})
        has_script = "script" in spec and cls._script(spec["script"]) is not None
        is_supported = (has_script or bool(spec.get("builtin_worker"))) and cls._first_file(spec.get("checkpoints", ())) is not None
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
        manifest.update(extra or {})
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

    # ---------------------------------------------------------------- command
    @classmethod
    def _command(
        cls,
        backend: Dict[str, Any],
        image_dir: Path,
        output_dir: Path,
        output_name: str,
        base_model: str,
        plan: Dict[str, Any],
        framing_mode: str = "bucket",
    ) -> List[str]:
        """Build the trainer CLI from a resolved preset plan.

        Speed/reliability flags: cached latents (+ to disk), SDPA attention,
        constant LR schedule, UNet-only, hard step cap, no bucket upscaling,
        fp16 VAE guard, and 8-bit AdamW only when bitsandbytes is importable.
        """
        engine = backend.get("engine", ENGINE_DIFFUSERS)
        script_file = backend.get("script") if engine == ENGINE_KOHYA else cls._worker_script()
        python = backend.get("python") or sys.executable
        network_mod = backend.get("spec", {}).get("network_module", "networks.lora")
        res = int(plan["resolution"])
        mp = plan["mixed_precision"]
        cmd = [
            python,
            str(script_file),
            f"--pretrained_model_name_or_path={backend['checkpoint']}",
            f"--train_data_dir={image_dir}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            "--caption_extension=.txt",
            f"--resolution={res},{res}",
            f"--network_dim={plan['lora_rank']}",
            f"--network_alpha={plan['lora_alpha']}",
            f"--learning_rate={plan['learning_rate']}",
            f"--max_train_steps={plan['max_train_steps']}",
            f"--train_batch_size={plan['batch_size']}",
            f"--optimizer_type={plan['optimizer_type']}",
            f"--mixed_precision={mp}",
            "--network_train_unet_only",
        ]
        if plan.get("gradient_checkpointing", True):
            cmd.append("--gradient_checkpointing")

        if engine == ENGINE_KOHYA:
            cmd += [
                f"--network_module={network_mod}",
                "--save_model_as=safetensors",
                f"--save_precision={'bf16' if mp == 'bf16' else 'fp16'}",
                "--lr_scheduler=constant",
                "--cache_latents",
                "--cache_latents_to_disk",
                "--sdpa",
                "--max_data_loader_n_workers=0",
                "--seed=42",
            ]
            if framing_mode == "bucket":
                cmd += [
                    "--enable_bucket",
                    "--bucket_no_upscale",
                    f"--min_bucket_reso={max(256, res // 2)}",
                    f"--max_bucket_reso={res}",
                    "--bucket_reso_steps=64",
                ]
            elif framing_mode == "crop":
                cmd.append("--random_crop")
            if base_model == "sdxl-1.0":
                if mp == "fp16":
                    cmd.append("--no_half_vae")
            if base_model == "flux-1-dev":
                vram = float(backend.get("hardware", {}).get("total_vram_gb") or 0)
                cmd += [
                    f"--clip_l={backend['components']['clip_l']}",
                    f"--t5xxl={backend['components']['t5xxl']}",
                    f"--ae={backend['components']['ae']}",
                    "--cache_text_encoder_outputs",
                    "--cache_text_encoder_outputs_to_disk",
                    "--apply_t5_attn_mask",
                    "--timestep_sampling=shift",
                    "--discrete_flow_shift=3.1582",
                    "--model_prediction_type=raw",
                    "--guidance_scale=1.0",
                    "--loss_type=l2",
                ]
                if vram < 23.0:
                    cmd += ["--fp8_base", "--blocks_to_swap=18" if vram < 16.0 else "--blocks_to_swap=8"]
        else:
            cmd += [f"--base_model_id={base_model}", "--seed=42"]
            if cls.cpu_training_allowed():
                cmd.append("--allow_cpu")
        return cmd

    # ----------------------------------------------------------------- verify
    @classmethod
    def _verify(cls, model: Path, base_model: str) -> Dict[str, Any]:
        if not model.is_file() or model.stat().st_size < 64 * 1024:
            raise RuntimeError("Trainer did not produce a substantial safetensors LoRA.")
        from safetensors import safe_open
        import torch
        with safe_open(str(model), framework="pt", device="cpu") as f:
            keys, metadata = list(f.keys()), (f.metadata() or {})
            down_keys = [k for k in keys if k.endswith(".lora_down.weight")]
            up_keys = [k for k in keys if k.endswith(".lora_up.weight")]
            if len(down_keys) < 10:
                raise RuntimeError("Output lacks sufficient LoRA tensor pairs.")
            # A trained adapter has non-zero "up" matrices: they start at zero
            # and only gradient descent moves them.  Random or untouched
            # adapters fail this check.
            sample = up_keys[: min(len(up_keys), 64)]
            nonzero = 0
            for k in sample:
                t = f.get_tensor(k).float()
                if torch.isfinite(t).all() and float(t.abs().sum()) > 0:
                    nonzero += 1
            if nonzero < max(1, len(sample) // 2):
                raise RuntimeError(
                    f"Output adapter is not trained: only {nonzero}/{len(sample)} up-projection tensors are non-zero."
                )
        architecture = metadata.get("modelspec.architecture", "").lower()
        if "sdxl" in base_model:
            expected = "stable-diffusion-xl"
        elif "flux" in base_model:
            expected = "flux"
        else:
            expected = ""
        if expected and expected not in architecture:
            raise RuntimeError(f"Output metadata is not compatible with {base_model}: {architecture or 'missing'}")
        return {
            "tensor_count": len(keys),
            "adapter_pairs": len(down_keys),
            "nonzero_up_tensors": f"{nonzero}/{len(sample)}",
            "architecture": architecture,
        }

    # --------------------------------------------------------------- pipeline
    @classmethod
    def start_training_pipeline(
        cls,
        project_dir: str,
        base_model: str = "sdxl-1.0",
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        epochs: Optional[int] = None,
        repeats: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        execution_mode: str = "real_gpu",
        framing_mode: str = "bucket",
        preset: str = DEFAULT_PRESET,
        resolution: Optional[int] = None,
        max_train_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        if execution_mode not in {"real_gpu", "validate_only", "dry_run"}:
            raise ValueError("execution_mode must be 'real_gpu' or 'validate_only'")
        for name, val in (("lora_rank", lora_rank), ("lora_alpha", lora_alpha), ("epochs", epochs),
                          ("repeats", repeats), ("batch_size", batch_size), ("resolution", resolution),
                          ("max_train_steps", max_train_steps)):
            if val is not None and val < 1:
                raise ValueError(f"{name} must be positive")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if preset not in PRESETS and preset != "custom":
            raise ValueError(f"Unknown preset '{preset}'. Choose one of: custom, {', '.join(PRESETS)}")

        p = Path(project_dir).resolve()
        if not p.exists():
            raise ValueError(f"Project directory '{project_dir}' does not exist. Please select an existing project or create a new one.")
        dataset = DatasetService.get_project_dataset(str(p))
        if dataset["total_count"] == 0:
            raise ValueError(f"Project '{p.name}' has no keyframe images in Keyframes_Out. Real training requires at least two valid, fully captioned keyframes.")
        if dataset["total_count"] < 2:
            raise ValueError(f"Project '{p.name}' only has {dataset['total_count']} keyframe. Real training requires at least two valid, fully captioned keyframes.")
        if dataset["captioned_count"] < dataset["total_count"]:
            missing = dataset["total_count"] - dataset["captioned_count"]
            raise ValueError(f"Project '{p.name}' has {missing} uncaptioned keyframe(s) out of {dataset['total_count']}. Real training requires at least two valid, fully captioned keyframes.")

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
            raise ValueError("Corrupted or unreadable images detected. Real training requires at least two valid, fully captioned keyframes.")

        backend = cls._backend(base_model)
        hw = backend["hardware"]
        bnb = cls._bitsandbytes_available(backend["python"]) if hw["cuda_available"] else False
        frames = dataset["captioned_count"]

        if preset == "custom":
            # Legacy explicit parameters; the step cap is derived from them.
            rank = lora_rank or 16
            alpha = lora_alpha or rank
            ep = epochs or 5
            rp = repeats or 10
            bs = batch_size or 1
            steps = max_train_steps or max(1, (frames * rp * ep) // bs)
            plan = resolve_preset(
                "balanced", hw, frames, base_model, bitsandbytes_available=bnb,
                overrides={"rank": rank, "alpha": alpha, "learning_rate": learning_rate or 1e-4,
                           "resolution": resolution or 1024, "max_train_steps": steps, "batch_size": bs},
            )
            plan.update({"preset": "custom", "preset_name": "Custom", "epochs": ep, "repeats": rp, "estimated_minutes": None})
        else:
            plan = resolve_preset(
                preset, hw, frames, base_model, bitsandbytes_available=bnb,
                overrides={"rank": lora_rank, "alpha": lora_alpha, "learning_rate": learning_rate,
                           "resolution": resolution, "max_train_steps": max_train_steps, "batch_size": batch_size},
            )

        # The built-in worker handles one resolution; bucket exports are square-padded for it.
        export_framing = framing_mode if backend["engine"] == ENGINE_KOHYA else ("crop" if framing_mode == "crop" else "pad")
        exported = DatasetService.export_for_kohya(
            str(p),
            repeats=plan["repeats"],
            class_token="character",
            target_resolution=plan["resolution"],
            framing_mode=export_framing,
        )
        if exported["exported_frames"] < 2:
            raise ValueError("Dataset export did not produce two usable image/caption pairs.")

        meta = DatasetService.load_project_meta(str(p)) or {}
        output_dir = p / "Training" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        name = f"{meta.get('character_name', 'character')}_{base_model}_lora"
        target = output_dir / f"{name}.safetensors"
        command = cls._command(backend, p / "Training" / "img", output_dir, name, base_model, plan, framing_mode=framing_mode)
        cwd_dir = str(backend["script"].parent) if backend.get("script") else str(p)
        command_file = p / "Training" / "real_training_command.json"
        command_file.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": cwd_dir,
                    "engine": backend["engine"],
                    "python": backend["python"],
                    "plan": plan,
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
            "engine": backend["engine"],
            "preset": plan["preset"],
            "plan": plan,
            "trainer_script": str(backend["script"]) if backend.get("script") else str(cls._worker_script()),
            "trainer_python": backend["python"],
            "base_checkpoint": str(backend["checkpoint"]),
            "target_model_file": str(target),
            "command_file": str(command_file),
            "progress_percent": 0.0,
            "current_step": 0,
            "total_steps": plan["max_train_steps"],
            "current_loss": None,
            "loss_history": [],
            "telemetry_verified": False,
            "started_at": time.time(),
            "eta_seconds": None,
            "log": [
                "Preflight passed: trainer, checkpoint, CUDA, captions, and dataset export verified.",
                f"Engine: {backend['engine']} | preset: {plan['preset_name']} | {plan['max_train_steps']} steps @ {plan['resolution']}px, rank {plan['lora_rank']}, {plan['optimizer_type']}, {plan['mixed_precision']}",
            ],
        }
        config = cls.generate_kohya_toml_config(
            str(p), base_model, plan["lora_rank"], plan["lora_alpha"], plan["learning_rate"],
            plan["epochs"], plan["batch_size"], plan["resolution"],
            extra={"preset": plan["preset"], "max_train_steps": plan["max_train_steps"], "engine": backend["engine"]},
        )
        handoff = cls.generate_comfyui_workflow(str(p), base_model, plan["lora_rank"], plan["lora_alpha"])

        if execution_mode in {"validate_only", "dry_run"}:
            job.update({"status": "validated", "progress_percent": 100.0})
            job["log"].append("Validation complete; no model weights were created.")
            return {
                "status": "validated",
                "execution_mode": "validate_only",
                "project_dir": str(p),
                "base_model": base_model,
                "engine": backend["engine"],
                "plan": plan,
                "config_toml": config,
                "comfyui_workflow": handoff,
                "training_command": str(command_file),
                "target_output": str(target),
            }

        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass

        def worker():
            started = time.time()
            all_lines = []
            try:
                env = os.environ.copy()
                env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "TF_ENABLE_ONEDNN_OPTS": "0",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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
                cls._active_procs[str(p)] = proc
                job["pid"] = proc.pid
                pattern = re.compile(r"(\d+)/(\d+).*?(?:avr_loss|loss)[=:\s]+([0-9.]+)", re.I)
                fixed_pattern = re.compile(r"fixed_batch_loss_before=([0-9.]+)\s+fixed_batch_loss_after=([0-9.]+)")
                learning = {}

                def stream_chunks(pipe):
                    buf = ""
                    while True:
                        chunk = pipe.read(64)
                        if not chunk:
                            if buf.strip():
                                yield buf.strip()
                            break
                        buf += chunk
                        while "\r" in buf or "\n" in buf:
                            idx_r = buf.find("\r")
                            idx_n = buf.find("\n")
                            if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
                                split_idx = idx_r
                            else:
                                split_idx = idx_n
                            line = buf[:split_idx].strip()
                            buf = buf[split_idx + 1:]
                            if line:
                                yield line

                first_step_at = None
                for line in stream_chunks(proc.stdout):
                    all_lines.append(line)
                    fixed = fixed_pattern.search(line)
                    if fixed:
                        before, after = float(fixed.group(1)), float(fixed.group(2))
                        learning = {"fixed_batch_loss_before": before, "fixed_batch_loss_after": after,
                                    "learning_verified": after < before}
                        job["learning"] = learning
                    found = pattern.search(line)
                    if found and not fixed:
                        step, total, loss = int(found.group(1)), int(found.group(2)), float(found.group(3))
                        now = time.time()
                        if first_step_at is None:
                            first_step_at = now
                        rate = (step - 1) / (now - first_step_at) if step > 1 and now > first_step_at else None
                        job.update({
                            "current_step": step,
                            "total_steps": total,
                            "current_loss": loss,
                            "telemetry_verified": True,
                            "progress_percent": round(100 * step / max(total, 1), 1),
                            "eta_seconds": round((total - step) / rate) if rate else None,
                        })
                        job["loss_history"].append({"step": step, "loss": loss})
                    if len(line) < 500:
                        job["log"].append(line)
                        if len(job["log"]) > 300:
                            job["log"] = job["log"][-300:]
                proc.wait()
                if proc.returncode != 0:
                    err_lines = [l for l in all_lines if any(k in l.lower() for k in ("error", "exception", "failed", "found:", "cuda out of memory", "no module named"))]
                    detail = err_lines[-1] if err_lines else (all_lines[-1] if all_lines else f"Exited with code {proc.returncode}")
                    raise RuntimeError(f"Trainer exited with code {proc.returncode}: {detail}")
                verification = cls._verify(target, base_model)
                metrics = {
                    "schema": "lora-maker-training-metrics-v1",
                    "verified": True,
                    "base_model": base_model,
                    "preset": plan["preset"],
                    "plan": plan,
                    "base_checkpoint": str(backend["checkpoint"]),
                    "base_checkpoint_sha256": cls._sha256(backend["checkpoint"]),
                    "trainer_script": job["trainer_script"],
                    "trainer_python": backend["python"],
                    "training_engine": backend["engine"],
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
                    **learning,
                }
                target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                job.update({
                    "status": "completed",
                    "progress_percent": 100.0,
                    "artifact_verified": True,
                    "verification": verification,
                    "duration_seconds": metrics["duration_seconds"],
                })
                job["log"].append(
                    f"Completed in {metrics['duration_seconds']}s: {verification['adapter_pairs']} adapter pairs, "
                    f"{verification['nonzero_up_tensors']} trained up-tensors verified."
                )
                if learning:
                    job["log"].append(
                        f"Learning check: fixed-batch loss {learning['fixed_batch_loss_before']:.4f} -> "
                        f"{learning['fixed_batch_loss_after']:.4f} ({'improved' if learning['learning_verified'] else 'no improvement'})."
                    )
            except Exception as exc:
                logger.exception("Real LoRA training failed")
                err_str = str(exc)
                job.update({"status": "failed", "error": err_str})
                job["log"].append(f"Training failed: {err_str}")
                try:
                    from app.services.diagnostics_service import DiagnosticsService
                    DiagnosticsService.record_error(
                        endpoint=f"/api/training/status?project_dir={p.name}",
                        method="WORKER",
                        error=exc,
                        traceback_str="\n".join(all_lines[-25:]) if all_lines else str(exc),
                        payload={"project_dir": str(p), "base_model": base_model, "output_model": str(target), "engine": backend["engine"]},
                        status_code=500,
                    )
                except Exception:
                    pass
            finally:
                cls._active_procs.pop(str(p), None)

        threading.Thread(target=worker, daemon=True).start()
        return {
            "status": "started",
            "execution_mode": "real_gpu",
            "project_dir": str(p),
            "base_model": base_model,
            "engine": backend["engine"],
            "plan": plan,
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
                "telemetry_verified": False,
                "log": [],
            },
        )

    @classmethod
    def stop_training(cls, project_dir: str) -> Dict[str, Any]:
        p = Path(project_dir).resolve()
        key = str(p)
        proc = cls._active_procs.get(key)
        job = cls._active_jobs.get(key)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        cls._active_procs.pop(key, None)
        if job:
            job.update({"status": "stopped", "progress_percent": 0.0})
            job["log"].append("Training manually stopped by user request.")
        return {"status": "stopped", "project_dir": str(p)}
