"""Unit and Integration Tests for Hardware Auto-Detection, Presets, and System API
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.hardware_service import HardwareService
from app.core.config import settings

client = TestClient(app)

def test_hardware_service_profile():
    """Verify that HardwareService correctly probes system resources and returns a valid tier."""
    hw = HardwareService.get_hardware_profile()
    assert "backend" in hw
    assert "device_name" in hw
    assert "total_vram_gb" in hw
    assert "tier_id" in hw
    assert "tier_name" in hw
    assert "recommendations" in hw

    recs = hw["recommendations"]
    assert "batch_size" in recs
    assert "gradient_accumulation_steps" in recs
    assert "optimizer" in recs
    assert "mixed_precision" in recs
    assert "recommended_models" in recs
    assert len(recs["recommended_models"]) > 0

def test_api_system_hardware():
    """Verify /api/system/hardware endpoint."""
    res = client.get("/api/system/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "device_name" in data
    assert "tier_name" in data
    assert "badge_color" in data

def test_api_system_presets():
    """Verify /api/system/presets endpoint returns universal presets."""
    res = client.get("/api/system/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    preset_ids = [p["id"] for p in data["presets"]]
    assert "photoreal_portrait" in preset_ids
    assert "anime_manga" in preset_ids
    assert "vintage_cartoon" in preset_ids
    assert "stylized_3d" in preset_ids
    assert "environment_concept" in preset_ids
    assert "product_design" in preset_ids

def test_api_system_info_branding():
    """Verify /api/system/info returns accurate Geekatplay Studio and Vladimir Chopine branding."""
    res = client.get("/api/system/info")
    assert res.status_code == 200
    data = res.json()
    assert data["app_name"] == "Geekatplay LoRA Maker"
    assert data["author"] == "Vladimir Chopine"
    assert data["studio"] == "Geekatplay Studio"
    assert data["repository"] == "https://github.com/GeekatplayStudio/LoraMaker.git"
    assert "hardware_tier" in data

def test_project_init_with_custom_trigger_token(tmp_path):
    """Verify project initialization accepts custom trigger_token and clean metadata."""
    from app.services.dataset_service import DatasetService
    p_dir = tmp_path / "Test_Custom_Project"
    meta = DatasetService.initialize_project(
        project_dir=str(p_dir),
        project_name="Test_Custom_Project",
        character_name="Elena Vance",
        style_description="photorealistic 85mm portrait",
        trigger_token="elena_vance"
    )
    assert meta["project_name"] == "Test_Custom_Project"
    assert meta["character_name"] == "Elena Vance"
    assert meta["trigger_token"] == "elena_vance"
    assert (p_dir / "project.json").exists()
