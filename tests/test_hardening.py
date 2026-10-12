"""Security, concurrency and robustness checks added in the engineering review."""
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.paths import PathNotAllowed, ensure_allowed_file, safe_filename, project_root_of
from app.main import app, LOCAL_ORIGINS
from app.services.dataset_service import DatasetService
from app.services.settings_service import SettingsService
from app.services.training_service import TrainingService

client = TestClient(app)


# --------------------------------------------------------------------- paths
def test_safe_filename_strips_directories_and_bad_chars():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("..\\..\\win.ini") == "win.ini"
    assert safe_filename('a<b>:"c".mp4') == "abc.mp4"
    assert safe_filename("   ") == "upload"


def test_ensure_allowed_file_rejects_wrong_type_and_outside_project(temp_project_dir, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    with pytest.raises(PathNotAllowed):
        ensure_allowed_file(str(secret), {".png"})
    loose = tmp_path / "loose.png"
    loose.write_bytes(b"png")
    ensure_allowed_file(str(loose), {".png"})  # reading a loose image is fine
    with pytest.raises(PathNotAllowed):
        ensure_allowed_file(str(loose), {".png"}, within_project=True)
    DatasetService.initialize_project(temp_project_dir, "P", "C")
    inside = Path(temp_project_dir) / "Keyframes_Out" / "f.png"
    inside.write_bytes(b"png")
    assert ensure_allowed_file(str(inside), {".png"}, within_project=True) == inside.resolve()
    assert project_root_of(inside.resolve()) == Path(temp_project_dir).resolve()


def test_api_refuses_to_serve_or_delete_arbitrary_files(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    assert client.get("/api/datasets/image", params={"image_path": str(secret)}).status_code == 403
    assert client.get("/api/video/stream", params={"video_path": str(secret)}).status_code == 403
    loose = tmp_path / "loose.png"
    loose.write_bytes(b"png")
    res = client.request("DELETE", "/api/datasets/frame", json={"image_path": str(loose)})
    assert res.status_code == 403 and loose.exists()
    res = client.put("/api/datasets/caption", json={"image_path": str(loose), "caption": "x"})
    assert res.status_code == 403 and not loose.with_suffix(".txt").exists()


def test_cors_is_limited_to_local_origins():
    assert all(o.startswith("http://127.0.0.1") or o.startswith("http://localhost") for o in LOCAL_ORIGINS)
    res = client.options("/api/system/info", headers={"Origin": "https://evil.example",
                                                      "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in {k.lower() for k in res.headers}
    res = client.get("/api/system/info", headers={"Origin": LOCAL_ORIGINS[0]})
    assert res.headers.get("access-control-allow-origin") == LOCAL_ORIGINS[0]


def test_upload_sanitises_filename_and_requires_project(temp_project_dir, sample_video_path):
    DatasetService.initialize_project(temp_project_dir, "Up", "C")
    with open(sample_video_path, "rb") as fh:
        res = client.post("/api/video/upload", data={"project_dir": temp_project_dir},
                          files={"file": ("../../escape.mp4", fh, "video/mp4")})
    assert res.status_code == 200, res.text
    saved = Path(res.json()["saved_path"])
    assert saved.parent == (Path(temp_project_dir) / "Input_Video").resolve() and saved.name == "escape.mp4"
    with open(sample_video_path, "rb") as fh:
        res = client.post("/api/video/upload", data={"project_dir": temp_project_dir},
                          files={"file": ("notes.txt", fh, "text/plain")})
    assert res.status_code == 400
    with open(sample_video_path, "rb") as fh:
        res = client.post("/api/video/upload", data={"project_dir": str(Path(temp_project_dir) / "nope")},
                          files={"file": ("a.mp4", fh, "video/mp4")})
    assert res.status_code == 400


# ------------------------------------------------------------------ training
def _ready_project(project_dir: str):
    from PIL import Image
    DatasetService.initialize_project(project_dir, "Lock", "Zorb")
    kf = Path(project_dir) / "Keyframes_Out"
    for i in range(2):
        Image.new("RGB", (64, 64), (i * 90, 0, 0)).save(kf / f"z_{i}.png")
        (kf / f"z_{i}.txt").write_text("zorb", encoding="utf-8")


def test_second_training_start_is_rejected_while_running(temp_project_dir, monkeypatch):
    _ready_project(temp_project_dir)
    key = str(Path(temp_project_dir).resolve())
    with TrainingService._lock:
        TrainingService._active_jobs[key] = {"status": "running", "log": [], "loss_history": []}
    try:
        with pytest.raises(ValueError, match="already in progress"):
            TrainingService.start_training_pipeline(temp_project_dir, "sdxl-1.0", execution_mode="real_gpu")
    finally:
        with TrainingService._lock:
            TrainingService._active_jobs.pop(key, None)


def test_job_status_is_a_snapshot(temp_project_dir):
    key = str(Path(temp_project_dir).resolve())
    with TrainingService._lock:
        TrainingService._active_jobs[key] = {"status": "running", "log": ["a"], "loss_history": [{"step": 1, "loss": 1.0}]}
    try:
        snap = TrainingService.get_job_status(temp_project_dir)
        snap["log"].append("mutated")
        snap["loss_history"].clear()
        assert TrainingService._active_jobs[key]["log"] == ["a"]
        assert len(TrainingService._active_jobs[key]["loss_history"]) == 1
    finally:
        with TrainingService._lock:
            TrainingService._active_jobs.pop(key, None)
    idle = TrainingService.get_job_status(temp_project_dir)
    idle["log"].append("x")
    assert TrainingService.get_job_status(temp_project_dir)["log"] == []


def test_stop_marks_job_stopped_not_failed(temp_project_dir, monkeypatch):
    """A user stop must win over the worker's non-zero exit code."""
    import subprocess
    _ready_project(temp_project_dir)
    key = str(Path(temp_project_dir).resolve())
    script = Path(temp_project_dir) / "slow.py"
    script.write_text("import time\nfor i in range(1, 200):\n    print(f'{i}/200 loss=0.5', flush=True)\n    time.sleep(0.05)\n")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    job = {"status": "running", "log": [], "loss_history": []}
    with TrainingService._lock:
        TrainingService._active_jobs[key] = job
        TrainingService._active_procs[key] = proc
    try:
        time.sleep(0.3)
        res = TrainingService.stop_training(temp_project_dir)
        assert res["status"] == "stopped" and proc.poll() is not None
        assert TrainingService.get_job_status(temp_project_dir)["status"] == "stopped"
    finally:
        with TrainingService._lock:
            TrainingService._active_jobs.pop(key, None)
            TrainingService._active_procs.pop(key, None)


def test_iter_lines_handles_cr_and_partial_chunks(tmp_path):
    import subprocess
    script = tmp_path / "emit.py"
    script.write_text("import sys\nsys.stdout.write('1/3 loss=0.9\\r2/3 loss=0.8\\n')\nsys.stdout.flush()\nsys.stdout.write('3/3 loss=0.7')\n")
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = list(TrainingService._iter_lines(proc.stdout))
    proc.wait()
    assert lines == ["1/3 loss=0.9", "2/3 loss=0.8", "3/3 loss=0.7"]


# -------------------------------------------------------------------- caching
def test_settings_cache_invalidates_on_save(monkeypatch, tmp_path):
    monkeypatch.setattr(SettingsService, "SETTINGS_FILE", tmp_path / "user_settings.json")
    SettingsService.invalidate_cache()
    calls = {"n": 0}
    real = SettingsService._load_user_settings.__func__

    def counting(cls):
        calls["n"] += 1
        return real(cls)

    monkeypatch.setattr(SettingsService, "_load_user_settings", classmethod(counting))
    SettingsService.get_user_settings()
    SettingsService.get_user_settings()
    assert calls["n"] == 1
    SettingsService.save_user_settings({"OLLAMA_HOST": "http://localhost:11434"})
    assert calls["n"] == 2  # save returns a fresh load, cache invalidated
    SettingsService.get_user_settings()
    assert calls["n"] == 2  # ...and that fresh load is what the cache now holds
    SettingsService.invalidate_cache()


def test_artifact_truth_is_memoised_until_file_changes(tmp_path):
    import torch
    from safetensors.torch import save_file
    from app.services.evaluation_service import EvaluationService

    f = tmp_path / "m_sdxl-1.0_lora.safetensors"
    tensors = {}
    for i in range(12):
        tensors[f"lora_unet_b{i}.lora_down.weight"] = torch.randn(4, 2048, dtype=torch.float16)
        tensors[f"lora_unet_b{i}.lora_up.weight"] = torch.randn(2048, 4, dtype=torch.float16)
    tensors["pad.lora_down.weight"] = torch.randn(64, 4096, dtype=torch.float16)
    tensors["pad.lora_up.weight"] = torch.randn(4096, 64, dtype=torch.float16)
    save_file(tensors, str(f), metadata={"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"})
    EvaluationService._truth_cache.clear()
    calls = {"n": 0}
    real = EvaluationService._compute_artifact_truth.__func__

    def counting(cls, path):
        calls["n"] += 1
        return real(cls, path)

    import unittest.mock as mock
    with mock.patch.object(EvaluationService, "_compute_artifact_truth", classmethod(counting)):
        assert EvaluationService._artifact_truth(f)["verified"] is True
        EvaluationService._artifact_truth(f)
        assert calls["n"] == 1
        f.write_bytes(f.read_bytes() + b"\0")  # size changes -> recompute
        EvaluationService._artifact_truth(f)
        assert calls["n"] == 2


# ------------------------------------------------------------ batch captions
def test_batch_caption_runs_in_background_and_reports_failures(temp_project_dir, monkeypatch):
    from app.services.vision_service import VisionService
    from PIL import Image

    DatasetService.initialize_project(temp_project_dir, "Cap", "Zorb")
    kf = Path(temp_project_dir) / "Keyframes_Out"
    for i in range(3):
        Image.new("RGB", (32, 32)).save(kf / f"z_{i}.png")

    def fake_caption(image_path, trigger_token, style_token, model_name=None, timeout_seconds=45):
        time.sleep(0.05)
        if image_path.endswith("z_1.png"):
            return {"success": False, "error": "ollama down", "caption": None, "image_file": image_path}
        Path(image_path).with_suffix(".txt").write_text(f"{trigger_token}, ok", encoding="utf-8")
        return {"success": True, "caption": f"{trigger_token}, ok", "image_file": image_path}

    monkeypatch.setattr(VisionService, "generate_caption", staticmethod(fake_caption))
    res = client.post("/api/captions/batch", json={"project_dir": temp_project_dir})
    assert res.status_code == 200 and res.json()["status"] == "running" and res.json()["total"] == 3
    # A second start while running is refused.
    assert client.post("/api/captions/batch", json={"project_dir": temp_project_dir}).status_code == 409
    for _ in range(100):
        st = client.get("/api/captions/batch/status", params={"project_dir": temp_project_dir}).json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "completed" and st["processed_count"] == 3 and st["failed_count"] == 1
    ds = DatasetService.get_project_dataset(temp_project_dir)
    assert ds["captioned_count"] == 2  # the failed frame stays uncaptioned, no fabricated text
    # Only the failed frame is retried on the next run.
    res = client.post("/api/captions/batch", json={"project_dir": temp_project_dir})
    assert res.json()["status"] == "running" and res.json()["total"] == 1
    for _ in range(100):
        st = client.get("/api/captions/batch/status", params={"project_dir": temp_project_dir}).json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    # Caption every frame -> nothing to do -> immediate completion, no job.
    (kf / "z_1.txt").write_text("zorb, manual", encoding="utf-8")
    res = client.post("/api/captions/batch", json={"project_dir": temp_project_dir})
    assert res.json()["status"] == "completed" and res.json()["total"] == 0
