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

@pytest.fixture(autouse=True)
def preserve_user_settings():
    from app.services.settings_service import SettingsService
    original_file = SettingsService.SETTINGS_FILE
    content = original_file.read_text(encoding="utf-8") if original_file.exists() else None
    yield
    if content is not None:
        original_file.write_text(content, encoding="utf-8")
    elif original_file.exists():
        original_file.unlink()

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

def test_settings_service_cache_redirection(tmp_path):
    """Verify SettingsService sets HF_HOME and TORCH_HOME to protect primary drive."""
    import os
    from app.services.settings_service import SettingsService

    custom_download_dir = str(tmp_path / "custom_models")
    res = SettingsService.save_settings({
        "DOWNLOAD_DIR": custom_download_dir
    })
    assert res["success"] is True
    assert res["DOWNLOAD_DIR"] == custom_download_dir

    # Check env vars were redirected
    assert os.environ.get("HF_HOME") == str(tmp_path / "custom_models" / "huggingface")
    assert os.environ.get("HUGGINGFACE_HUB_CACHE") == str(tmp_path / "custom_models" / "huggingface" / "hub")
    assert os.environ.get("TORCH_HOME") == str(tmp_path / "custom_models" / "torch")

def test_api_settings_endpoints(tmp_path):
    """Verify GET and POST /api/system/settings and scan endpoints."""
    # 1. GET settings
    get_res = client.get("/api/system/settings")
    assert get_res.status_code == 200
    data = get_res.json()
    assert "MODELS_DIR" in data
    assert "DOWNLOAD_DIR" in data
    assert "drives" in data
    assert "paths_status" in data

    # 2. Mock model scan directory
    mock_models = tmp_path / "mock_models"
    (mock_models / "checkpoints").mkdir(parents=True)
    (mock_models / "checkpoints" / "sd_xl_base_1.0.safetensors").write_bytes(b"dummy_weights" * 100)
    (mock_models / "loras").mkdir(parents=True)
    (mock_models / "loras" / "test_lora.safetensors").write_bytes(b"dummy_lora" * 100)

    scan_res = client.post("/api/system/settings/scan", json={"models_path": str(mock_models)})
    assert scan_res.status_code == 200
    scan_data = scan_res.json()
    assert scan_data["success"] is True
    assert scan_data["categories"]["checkpoints"]["count"] == 1
    assert scan_data["categories"]["loras"]["count"] == 1

def test_api_system_browse_endpoints(tmp_path):
    """Verify GET and POST /api/system/browse endpoints for interactive directory exploration."""
    # 1. Root / Drives query
    root_res = client.get("/api/system/browse")
    assert root_res.status_code == 200
    root_data = root_res.json()
    assert root_data["success"] is True
    assert root_data["is_root"] is True
    assert len(root_data["items"]) > 0

    # 2. Browse specific mock directory tree
    tree_root = tmp_path / "browse_root"
    sub1 = tree_root / "checkpoints"
    sub2 = tree_root / "loras"
    sub3 = tree_root / "random_folder"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    sub3.mkdir(parents=True)

    browse_res = client.get(f"/api/system/browse?path={tree_root}")
    assert browse_res.status_code == 200
    b_data = browse_res.json()
    assert b_data["success"] is True
    assert b_data["is_root"] is False
    assert len(b_data["breadcrumbs"]) >= 2
    assert len(b_data["items"]) == 3

    model_subdirs = [i["name"] for i in b_data["items"] if i.get("is_model_dir")]
    assert "checkpoints" in model_subdirs
    assert "loras" in model_subdirs

    # 3. POST variant
    post_res = client.post("/api/system/browse", json={"path": str(tree_root)})
    assert post_res.status_code == 200
    assert post_res.json()["total_items"] == 3

def test_multi_location_model_paths(tmp_path):
    """Verify adding, removing, and scanning multiple model directories across drives."""
    from app.services.settings_service import SettingsService

    loc1 = tmp_path / "models_drive_d"
    loc2 = tmp_path / "models_drive_f"
    (loc1 / "checkpoints").mkdir(parents=True)
    (loc2 / "loras").mkdir(parents=True)
    (loc1 / "checkpoints" / "model1.safetensors").write_bytes(b"model1" * 100)
    (loc2 / "loras" / "lora1.safetensors").write_bytes(b"lora1" * 100)

    # 1. Add loc2 to search paths
    add_res = client.post("/api/system/settings/add_model_path", json={"path": str(loc2)})
    assert add_res.status_code == 200
    data = add_res.json()
    assert str(loc2.resolve()) in data["EXTRA_MODEL_PATHS"]

    # 2. Verify SettingsService.get_all_model_directories() includes both
    # Temporarily set MODELS_DIR to loc1
    SettingsService.save_user_settings({"MODELS_DIR": str(loc1)})
    all_dirs = [str(p) for p in SettingsService.get_all_model_directories()]
    assert str(loc1.resolve()) in all_dirs
    assert str(loc2.resolve()) in all_dirs

    # 3. Multi-location scan should discover models across both folders
    scan_res = client.post("/api/system/settings/scan", json={})
    assert scan_res.status_code == 200
    scan_data = scan_res.json()
    assert scan_data["success"] is True
    assert "locations_scanned" in scan_data
    scanned_paths = [l["path"] for l in scan_data["locations_scanned"]]
    assert str(loc1.resolve()) in scanned_paths
    assert str(loc2.resolve()) in scanned_paths

    # 4. Remove loc2 from search paths
    rem_res = client.post("/api/system/settings/remove_model_path", json={"path": str(loc2)})
    assert rem_res.status_code == 200
    rem_data = rem_res.json()
    assert str(loc2.resolve()) not in rem_data["EXTRA_MODEL_PATHS"]



