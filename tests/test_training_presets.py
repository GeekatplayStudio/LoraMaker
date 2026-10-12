from pathlib import Path

import pytest

from app.services import training_service as ts
from app.services.training_presets import PRESETS, DEFAULT_PRESET, list_presets, resolve_preset
from app.services.training_service import TrainingService, ENGINE_KOHYA, ENGINE_DIFFUSERS
from app.services.dataset_service import DatasetService


def _hw(tier="mid", vram=12.0, bf16=True):
    return {"tier_id": tier, "total_vram_gb": vram, "supports_bf16": bf16, "cuda_available": True,
            "device_name": "Test GPU", "tier_name": tier}


def test_presets_are_step_budgeted_and_ordered():
    ids = [p["id"] for p in list_presets()]
    assert ids == ["smoke_test", "fast", "balanced", "quality"]
    assert DEFAULT_PRESET in PRESETS
    steps = [PRESETS[i]["steps"] for i in ids]
    assert steps == sorted(steps)
    assert PRESETS["smoke_test"]["steps"] <= 30


def test_resolve_caps_resolution_by_tier_and_derives_repeats():
    plan = resolve_preset("balanced", _hw("entry", 8.0), dataset_frame_count=20)
    assert plan["resolution"] == 768  # entry tier ceiling
    assert plan["max_train_steps"] == 900
    assert plan["planned_samples"] >= plan["max_train_steps"]
    assert plan["optimizer_type"] == "AdamW"  # bitsandbytes not available
    assert plan["mixed_precision"] == "bf16"

    plan8 = resolve_preset("fast", _hw("mid", 12.0), dataset_frame_count=20, bitsandbytes_available=True)
    assert plan8["optimizer_type"] == "AdamW8bit"
    ultra = resolve_preset("fast", _hw("ultra", 24.0), dataset_frame_count=20, bitsandbytes_available=True)
    assert ultra["optimizer_type"] == "AdamW"
    assert ultra["gradient_checkpointing"] is False


def test_resolve_overrides_and_unknown_preset():
    plan = resolve_preset("fast", _hw(), 10, overrides={"rank": 32, "max_train_steps": 50, "resolution": None})
    assert plan["lora_rank"] == 32 and plan["max_train_steps"] == 50 and plan["resolution"] == 768
    with pytest.raises(ValueError):
        resolve_preset("turbo", _hw(), 10)


def test_command_uses_cached_latents_and_step_cap_for_kohya(tmp_path):
    script = tmp_path / "sd-scripts" / "sdxl_train_network.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub")
    backend = {
        "engine": ENGINE_KOHYA, "script": script, "python": "python-kohya",
        "checkpoint": tmp_path / "sdxl.safetensors",
        "spec": TrainingService._TRAINERS["sdxl-1.0"], "components": {}, "hardware": _hw(),
    }
    plan = resolve_preset("fast", _hw(bf16=False), 20)
    cmd = TrainingService._command(backend, tmp_path / "img", tmp_path / "out", "x_sdxl-1.0_lora", "sdxl-1.0", plan)
    assert cmd[0] == "python-kohya" and cmd[1] == str(script)
    for flag in ("--cache_latents", "--cache_latents_to_disk", "--sdpa", "--network_train_unet_only",
                 "--max_train_steps=400", "--resolution=768,768", "--no_half_vae", "--lr_scheduler=constant",
                 "--enable_bucket", "--bucket_no_upscale", "--save_model_as=safetensors"):
        assert flag in cmd, flag
    assert not any(c.startswith("--max_train_epochs") for c in cmd)


def test_command_targets_builtin_worker_without_kohya(tmp_path):
    backend = {
        "engine": ENGINE_DIFFUSERS, "script": None, "python": "python-ours",
        "checkpoint": tmp_path / "sdxl.safetensors",
        "spec": TrainingService._TRAINERS["sdxl-1.0"], "components": {}, "hardware": _hw(),
    }
    plan = resolve_preset("smoke_test", _hw(), 4)
    cmd = TrainingService._command(backend, tmp_path / "img", tmp_path / "out", "x_sdxl-1.0_lora", "sdxl-1.0", plan)
    assert cmd[1].endswith("diffusers_lora_worker.py")
    assert "--max_train_steps=20" in cmd and "--network_dim=4" in cmd
    assert "--base_model_id=sdxl-1.0" in cmd


def test_sdxl_available_via_builtin_worker_when_kohya_missing(monkeypatch, tmp_path):
    ckpt = tmp_path / "sd_xl_base_1.0.safetensors"
    ckpt.write_bytes(b"0" * 10)
    monkeypatch.setattr(TrainingService, "_script", classmethod(lambda cls, name: None))
    monkeypatch.setattr(TrainingService, "_first_file", classmethod(lambda cls, c: ckpt if "sd_xl" in str(c) else None))
    hw = _hw()
    b = TrainingService._resolve_backend("sdxl-1.0", hw)
    assert b["engine"] == ENGINE_DIFFUSERS and b["missing"] == []
    # Architectures without a verified trainer stay unavailable: no stand-in training.
    b2 = TrainingService._resolve_backend("qwen-image", hw)
    assert b2["engine"] is None and any("no verified trainer" in m for m in b2["missing"])


def test_trainer_python_prefers_kohya_venv(tmp_path):
    script = tmp_path / "kohya_ss" / "sd-scripts" / "sdxl_train_network.py"
    script.parent.mkdir(parents=True)
    script.write_text("#")
    assert TrainingService._trainer_python(script) == ts.sys.executable
    venv_py = tmp_path / "kohya_ss" / "venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("#")
    assert TrainingService._trainer_python(script) == str(venv_py)


def test_verify_rejects_untrained_zero_up_weights(tmp_path):
    import torch
    from safetensors.torch import save_file

    tensors = {}
    for i in range(12):
        tensors[f"lora_unet_block_{i}.lora_down.weight"] = torch.randn(4, 64, dtype=torch.float16)
        tensors[f"lora_unet_block_{i}.lora_up.weight"] = torch.zeros(64, 4, dtype=torch.float16)
        tensors[f"lora_unet_block_{i}.alpha"] = torch.tensor(4.0)
    # pad to pass the size floor
    tensors["lora_unet_pad.lora_down.weight"] = torch.randn(64, 1024, dtype=torch.float16)
    tensors["lora_unet_pad.lora_up.weight"] = torch.zeros(1024, 64, dtype=torch.float16)
    f = tmp_path / "fake_sdxl-1.0_lora.safetensors"
    save_file(tensors, str(f), metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"})
    with pytest.raises(RuntimeError, match="not trained"):
        TrainingService._verify(f, "sdxl-1.0")

    for i in range(12):
        tensors[f"lora_unet_block_{i}.lora_up.weight"] = torch.randn(64, 4, dtype=torch.float16)
    tensors["lora_unet_pad.lora_up.weight"] = torch.randn(1024, 64, dtype=torch.float16)
    save_file(tensors, str(f), metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"})
    info = TrainingService._verify(f, "sdxl-1.0")
    assert info["adapter_pairs"] == 13


def test_export_removes_stale_concept_folders(temp_project_dir):
    from PIL import Image

    DatasetService.initialize_project(temp_project_dir, "Stale", "Zorb")
    kf = Path(temp_project_dir) / "Keyframes_Out"
    for i in range(2):
        Image.new("RGB", (64, 64), (i * 50, 0, 0)).save(kf / f"z_{i}.png")
        (kf / f"z_{i}.txt").write_text("zorb", encoding="utf-8")
    DatasetService.export_for_kohya(temp_project_dir, repeats=10, target_resolution=64, framing_mode="pad")
    DatasetService.export_for_kohya(temp_project_dir, repeats=3, target_resolution=64, framing_mode="pad")
    folders = sorted(d.name for d in (Path(temp_project_dir) / "Training" / "img").iterdir() if d.is_dir())
    assert folders == ["3_Zorb character"]


def test_validate_only_records_plan_and_engine(temp_project_dir, monkeypatch, tmp_path):
    from PIL import Image

    DatasetService.initialize_project(temp_project_dir, "Plan", "Zorb")
    kf = Path(temp_project_dir) / "Keyframes_Out"
    for i in range(3):
        Image.new("RGB", (96, 64), (0, i * 60, 0)).save(kf / f"z_{i}.png")
        (kf / f"z_{i}.txt").write_text(f"zorb pose {i}", encoding="utf-8")
    ckpt = tmp_path / "sd_xl_base_1.0.safetensors"
    ckpt.write_bytes(b"0" * 10)
    monkeypatch.setattr(TrainingService, "_script", classmethod(lambda cls, name: None))
    monkeypatch.setattr(TrainingService, "_first_file", classmethod(lambda cls, c: ckpt if "sd_xl" in str(c) else None))
    monkeypatch.setattr(TrainingService, "_bitsandbytes_available", classmethod(lambda cls, py: False))
    from app.services.hardware_service import HardwareService
    monkeypatch.setattr(HardwareService, "get_hardware_profile", staticmethod(lambda: _hw("mid", 12.0)))

    res = TrainingService.start_training_pipeline(temp_project_dir, "sdxl-1.0", preset="smoke_test", execution_mode="validate_only")
    assert res["status"] == "validated" and res["engine"] == ENGINE_DIFFUSERS
    assert res["plan"]["max_train_steps"] == 20 and res["plan"]["resolution"] == 512
    import json
    cmdfile = json.loads(Path(res["training_command"]).read_text(encoding="utf-8"))
    assert "--max_train_steps=20" in cmdfile["command"]
    assert cmdfile["plan"]["preset"] == "smoke_test"
    assert not list((Path(temp_project_dir) / "Training" / "output").glob("*.safetensors"))
    status = TrainingService.get_job_status(temp_project_dir)
    assert status["status"] == "validated" and status["telemetry_verified"] is False
