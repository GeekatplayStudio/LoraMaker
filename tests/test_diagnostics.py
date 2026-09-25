"""Tests for DiagnosticsService, Console Endpoints, and Rich Error Reporting
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git
"""

from fastapi.testclient import TestClient
from app.main import app
from app.services.diagnostics_service import DiagnosticsService

client = TestClient(app)


def test_diagnostics_service_record_and_retrieve():
    DiagnosticsService.clear_errors()
    
    # Record a test exception
    try:
        raise KeyError("network_module")
    except Exception as e:
        rec = DiagnosticsService.record_error(
            endpoint="/api/training/start",
            method="POST",
            error=e,
            payload={"base_model": "qwen-image"},
            status_code=400,
        )
        assert rec["error_type"] == "KeyError"
        assert "network_module" in rec["message"]
        assert "LoRA trainer network module" in rec["suggestion"]

    report = DiagnosticsService.get_diagnostics_report()
    assert report["error_count"] >= 1
    assert len(report["recent_errors"]) >= 1
    assert report["hardware"]["device_name"] is not None
    assert "storage" in report


def test_diagnostics_api_endpoints():
    # 1. GET /api/system/diagnostics
    res = client.get("/api/system/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert "app_name" in data
    assert "hardware" in data
    assert "recent_errors" in data

    # 2. GET /api/system/logs
    res_logs = client.get("/api/system/logs?limit=20")
    assert res_logs.status_code == 200
    assert "logs" in res_logs.json()

    # 3. POST /api/system/diagnostics/clear
    res_clear = client.post("/api/system/diagnostics/clear")
    assert res_clear.status_code == 200
    assert res_clear.json()["success"] is True


def test_training_error_returns_structured_diagnostics(temp_project_dir):
    from pathlib import Path
    from app.services.dataset_service import DatasetService

    DatasetService.initialize_project(temp_project_dir, "DiagTest", "BendyBot")
    
    # Try starting training with no keyframes - should return 400 with rich structured detail
    payload = {
        "project_dir": temp_project_dir,
        "base_model": "sdxl-1.0",
        "lora_rank": 16,
        "execution_mode": "real_gpu",
    }
    res = client.post("/api/training/start", json=payload)
    assert res.status_code == 400
    data = res.json()

    # Verify structured detail contains traceback and suggestion
    # FastAPI returns either direct dict or nested in detail
    detail = data.get("detail") if isinstance(data.get("detail"), dict) else data
    assert "message" in detail or "detail" in data
    assert "error_type" in detail or "traceback" in detail
