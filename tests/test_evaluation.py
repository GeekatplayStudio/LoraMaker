import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.services.evaluation_service import EvaluationService
from app.services.dataset_service import DatasetService

client = TestClient(app)

@pytest.fixture
def active_project(tmp_path):
    p_dir = str(tmp_path / "Test_Studio_Project")
    DatasetService.initialize_project(
        project_dir=p_dir,
        project_name="Test_Studio_Project",
        character_name="BendyBot",
        style_description="vintage 1930s rubber hose cel animation"
    )
    # Create genuine safetensors in Training/output
    import torch
    import safetensors.torch
    out_dir = Path(p_dir) / "Training" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fake_model = out_dir / "BendyBot_qwen-image_lora.safetensors"
    tensors = {
        "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight": torch.randn(64, 16),
        "lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight": torch.randn(16, 64)
    }
    safetensors.torch.save_file(tensors, str(fake_model))
    return p_dir

def test_list_trained_models(active_project):
    models = EvaluationService.list_trained_models(active_project)
    assert len(models) == 1
    assert models[0]["filename"] == "BendyBot_qwen-image_lora.safetensors"
    assert models[0]["verified"] is False
    assert models[0]["is_valid_lora"] is False

def test_generate_suggested_prompts(active_project):
    res = EvaluationService.generate_suggested_prompts(active_project)
    assert res["success"] is True
    assert res["character_name"] == "BendyBot"
    assert res["trigger_token"] == "BendyBot"
    assert len(res["prompts"]) == 6
    assert any("Identity" in p["category"] for p in res["prompts"])
    assert any("Action" in p["category"] for p in res["prompts"])
    assert any("Video" in p["category"] for p in res["prompts"])

def test_get_training_analytics(active_project):
    res = EvaluationService.get_training_analytics(active_project)
    assert res["success"] is False
    assert res["metrics_verified"] is False

def test_render_test_sample_refuses_unverified_artifact(active_project):
    with pytest.raises(ValueError, match="Select a verified"):
        EvaluationService.render_test_sample(project_dir=active_project, prompt="BendyBot", lora_scale=0.85, seed=100)

def test_api_evaluation_endpoints(active_project):
    # 1. Models list
    res = client.get(f"/api/evaluation/models?project_dir={active_project}")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 1

    # 2. Inspect
    res = client.get(f"/api/evaluation/inspect?project_dir={active_project}&model_filename=BendyBot_qwen-image_lora.safetensors")
    assert res.status_code == 400

    # 3. Analytics
    res = client.get(f"/api/evaluation/analytics?project_dir={active_project}")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False

    # 4. Prompts
    res = client.get(f"/api/evaluation/prompts?project_dir={active_project}")
    assert res.status_code == 200
    data = res.json()
    assert len(data["prompts"]) == 6

    # 5. Unverified artifacts cannot be rendered, even with an explicit aspect ratio.
    res = client.post("/api/evaluation/test_render", json={
        "project_dir": active_project,
        "prompt": "BendyBot testing endpoint",
        "lora_scale": 0.9,
        "seed": 42,
        "aspect_ratio": "16:9",
        "framing_mode": "pad"
    })
    assert res.status_code == 500

def test_aspect_ratio_preservation_and_fitting():
    from PIL import Image
    # 1. Test 16:9 input (1920x1080) -> Target (512, 512)
    img_16_9 = Image.new("RGB", (1920, 1080), (255, 0, 0))
    
    # Pad mode: Canvas is exactly (512, 512), image content is scaled to 512x288 and centered (no distortion)
    padded = DatasetService.fit_image_aspect_ratio(img_16_9, 512, 512, mode="pad")
    assert padded.size == (512, 512)

    # Crop mode: Canvas is exactly (512, 512), scaled to fill and center cropped (no distortion)
    cropped = DatasetService.fit_image_aspect_ratio(img_16_9, 512, 512, mode="crop")
    assert cropped.size == (512, 512)

    # 2. Test 9:16 vertical input (1080x1920) -> Target (512, 512)
    img_9_16 = Image.new("RGB", (1080, 1920), (0, 255, 0))
    padded_vert = DatasetService.fit_image_aspect_ratio(img_9_16, 512, 512, mode="pad")
    assert padded_vert.size == (512, 512)

    cropped_vert = DatasetService.fit_image_aspect_ratio(img_9_16, 512, 512, mode="crop")
    assert cropped_vert.size == (512, 512)

    # 3. Test 2:3 portrait input (1000x1500)
    img_2_3 = Image.new("RGB", (1000, 1500), (0, 0, 255))
    padded_2_3 = DatasetService.fit_image_aspect_ratio(img_2_3, 448, 672, mode="pad")
    assert padded_2_3.size == (448, 672)

def test_detect_dataset_aspect_ratio(tmp_path):
    from PIL import Image
    p_dir = tmp_path / "AspectProj"
    DatasetService.initialize_project(str(p_dir), "AspectProj", "BendyBot")
    keyframes_dir = p_dir / "Keyframes_Out"

    # Save three 9:16 vertical images
    for i in range(3):
        im = Image.new("RGB", (1080, 1920), (50, 50, 50))
        im.save(keyframes_dir / f"frame_{i:04d}.png")

    detected = DatasetService.detect_dataset_aspect_ratio(str(p_dir))
    assert detected["detected_ratio"] == "9:16"
    assert detected["is_vertical"] is True
    assert detected["recommended_512"] == (384, 640)
