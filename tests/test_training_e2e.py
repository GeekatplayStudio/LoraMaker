"""End-to-end proof that the training pipeline really trains a LoRA.

Runs the full TrainingService flow (preflight -> dataset export -> subprocess
trainer -> artifact verification -> provenance metrics) on CPU against the
Hugging Face tiny SDXL test pipeline, then checks that:

* the trainer printed one loss line per optimizer step,
* the saved adapter has non-zero, finite up-projection weights,
* the loss on a fixed batch is lower after training than before (learning),
* the adapter loads through diffusers and changes the UNet output, and
* EvaluationService reports the artifact as verified with matching provenance.

The tiny model (~20 MB) is downloaded once; the test is skipped when it is
unavailable (offline CI without a cache).
"""
import json
import os
import time
from pathlib import Path

import pytest

TINY = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


@pytest.fixture(scope="module")
def tiny_model_dir():
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(TINY))
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"tiny SDXL test model unavailable: {exc}")


@pytest.fixture
def cpu_hw(monkeypatch):
    from app.services.hardware_service import HardwareService

    hw = {"tier_id": "mid", "total_vram_gb": 12.0, "supports_bf16": False, "cuda_available": False,
          "device_name": "CPU (test)", "tier_name": "test"}
    monkeypatch.setattr(HardwareService, "get_hardware_profile", staticmethod(lambda: hw))
    monkeypatch.setenv("LORAMAKER_ALLOW_CPU_TRAINING", "1")
    return hw


def _make_project(project_dir: str, n: int = 4):
    from PIL import Image, ImageDraw
    from app.services.dataset_service import DatasetService

    DatasetService.initialize_project(project_dir, "E2E", "Zorb")
    kf = Path(project_dir) / "Keyframes_Out"
    for i in range(n):
        im = Image.new("RGB", (160, 120), (30 + 40 * i, 80, 200 - 30 * i))
        ImageDraw.Draw(im).ellipse((20 + 10 * i, 20, 100 + 10 * i, 100), fill=(250, 200, 50))
        im.save(kf / f"zorb_{i:02d}.png")
        (kf / f"zorb_{i:02d}.txt").write_text(f"Zorb, a yellow blob character, pose {i}", encoding="utf-8")


def _wait(project_dir: str, timeout: float = 600.0):
    from app.services.training_service import TrainingService

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = TrainingService.get_job_status(project_dir)
        if status["status"] in {"completed", "failed", "stopped"}:
            return status
        time.sleep(0.5)
    raise AssertionError("training did not finish in time")


@pytest.mark.timeout(900)
def test_pipeline_trains_a_real_lora_end_to_end(temp_project_dir, tiny_model_dir, cpu_hw, monkeypatch):
    import torch
    from app.services.training_service import TrainingService, ENGINE_DIFFUSERS
    from app.services.evaluation_service import EvaluationService

    _make_project(temp_project_dir)
    monkeypatch.setattr(TrainingService, "_script", classmethod(lambda cls, name: None))
    monkeypatch.setattr(TrainingService, "_first_file", classmethod(
        lambda cls, c: tiny_model_dir if "sd_xl" in str(c) or "stable-diffusion-xl" in str(c) else None))
    monkeypatch.setattr(TrainingService, "_bitsandbytes_available", classmethod(lambda cls, py: False))

    steps = 30
    res = TrainingService.start_training_pipeline(
        temp_project_dir, "sdxl-1.0", preset="smoke_test", execution_mode="real_gpu",
        resolution=64, max_train_steps=steps, learning_rate=1e-3, batch_size=2,
    )
    assert res["status"] == "started" and res["engine"] == ENGINE_DIFFUSERS
    assert res["plan"]["max_train_steps"] == steps and res["plan"]["resolution"] == 64

    status = _wait(temp_project_dir)
    assert status["status"] == "completed", "\n".join(status.get("log", [])[-30:])
    assert status["artifact_verified"] is True
    assert status["telemetry_verified"] is True

    # 1. One telemetry line per optimizer step, all finite.
    assert status["current_step"] == steps and status["total_steps"] == steps
    assert len(status["loss_history"]) == steps
    assert all(0 < h["loss"] < 100 for h in status["loss_history"])

    # 2. The adapter learned: fixed-batch loss went down.
    learning = status["learning"]
    assert learning["fixed_batch_loss_after"] < learning["fixed_batch_loss_before"], learning
    assert learning["learning_verified"] is True

    # 3. Provenance file matches the artifact and records the plan.
    target = Path(res["target_output"])
    metrics = json.loads(target.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["verified"] is True and metrics["training_engine"] == ENGINE_DIFFUSERS
    assert metrics["output_sha256"] == TrainingService._sha256(target)
    assert metrics["plan"]["preset"] == "smoke_test" and metrics["total_steps"] == steps
    assert metrics["learning_verified"] is True

    # 4. Tensor-level check: Kohya keys, non-zero finite up weights.
    from safetensors import safe_open

    with safe_open(str(target), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        ups = [f.get_tensor(k).float() for k in keys if k.endswith(".lora_up.weight")]
    assert keys and all(k.startswith("lora_unet_") for k in keys)
    assert all(torch.isfinite(t).all() and t.abs().sum() > 0 for t in ups)

    # 5. EvaluationService agrees, by provenance and by structure.
    truth = EvaluationService._artifact_truth(target)
    assert truth["verified"] is True and truth["data"]["training_engine"] == ENGINE_DIFFUSERS
    listed = {m["filename"]: m for m in EvaluationService.list_trained_models(temp_project_dir)}
    assert listed[target.name]["verified"] is True
    inspect = EvaluationService.inspect_lora_model(temp_project_dir, target.name)
    assert inspect["success"] and inspect["health_grade"] == "OPTIMAL" and inspect["rank"] == 4

    # 6. The adapter loads in diffusers and changes what the UNet predicts.
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(str(tiny_model_dir))
    unet = pipe.unet
    g = torch.Generator().manual_seed(0)
    cfg = unet.config
    lat = torch.randn(1, cfg.in_channels, 8, 8, generator=g)
    t = torch.tensor([500])
    enc = torch.randn(1, 77, cfg.cross_attention_dim, generator=g)
    text_embeds = torch.randn(1, cfg.projection_class_embeddings_input_dim - 6 * cfg.addition_time_embed_dim, generator=g)
    time_ids = torch.tensor([[64, 64, 0, 0, 64, 64]], dtype=torch.float32)
    kw = {"text_embeds": text_embeds, "time_ids": time_ids}
    with torch.no_grad():
        base = unet(lat, t, encoder_hidden_states=enc, added_cond_kwargs=kw).sample.clone()
        pipe.load_lora_weights(str(target), adapter_name="e2e")
        with_lora = unet(lat, t, encoder_hidden_states=enc, added_cond_kwargs=kw).sample
        pipe.set_adapters(["e2e"], adapter_weights=[0.0])
        zero_scale = unet(lat, t, encoder_hidden_states=enc, added_cond_kwargs=kw).sample
    assert not torch.allclose(base, with_lora), "LoRA had no effect on the UNet output"
    assert torch.allclose(base, zero_scale, atol=1e-5), "scale 0 should reproduce the base model"


def test_untrained_adapter_is_rejected_by_pipeline_verification(tmp_path):
    """The old simulated worker wrote zero up-weights; that must fail now."""
    import torch
    from safetensors.torch import save_file
    from app.services.training_service import TrainingService
    from app.services.evaluation_service import EvaluationService

    tensors = {}
    for i in range(16):
        tensors[f"lora_unet_b{i}.lora_down.weight"] = torch.randn(4, 320, dtype=torch.float16)
        tensors[f"lora_unet_b{i}.lora_up.weight"] = torch.zeros(320, 4, dtype=torch.float16)
        tensors[f"lora_unet_b{i}.alpha"] = torch.tensor(4.0)
    tensors["lora_unet_pad.lora_down.weight"] = torch.randn(64, 2048, dtype=torch.float16)
    tensors["lora_unet_pad.lora_up.weight"] = torch.zeros(2048, 64, dtype=torch.float16)
    f = tmp_path / "sim_sdxl-1.0_lora.safetensors"
    save_file(tensors, str(f), metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"})
    assert f.stat().st_size > 500 * 1024
    with pytest.raises(RuntimeError, match="not trained"):
        TrainingService._verify(f, "sdxl-1.0")
    truth = EvaluationService._artifact_truth(f)
    assert truth["verified"] is False and "untrained" in truth["reason"]
