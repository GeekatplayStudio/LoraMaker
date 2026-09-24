from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health_and_index():
    res = client.get("/")
    assert res.status_code == 200

def test_api_project_lifecycle(temp_project_dir):
    # 1. Init project
    init_res = client.post("/api/projects/init", json={
        "project_dir": temp_project_dir,
        "project_name": "API_Studio",
        "character_name": "BendyBot",
        "style_description": "vintage 1930s rubber hose"
    })
    assert init_res.status_code == 200
    data = init_res.json()
    assert data["success"] is True
    assert data["project"]["project_name"] == "API_Studio"

    # 2. Get project info
    info_res = client.get(f"/api/projects/info?project_dir={temp_project_dir}")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert info_data["metadata"]["character_name"] == "BendyBot"
    assert info_data["dataset"]["total_count"] == 0

def test_api_video_probe_and_frame_cut(sample_video_path, temp_project_dir):
    # Init project first
    client.post("/api/projects/init", json={
        "project_dir": temp_project_dir,
        "project_name": "VideoCutTest",
        "character_name": "BendyBot"
    })

    # Probe video info
    info_res = client.get(f"/api/video/info?video_path={sample_video_path}")
    assert info_res.status_code == 200
    v_info = info_res.json()
    assert v_info["total_frames"] == 48

    # Get frame data URI
    frame_res = client.get(f"/api/video/frame?video_path={sample_video_path}&frame_index=5")
    assert frame_res.status_code == 200
    assert "data:image/jpeg;base64," in frame_res.json()["image_data"]

    # Cut Frame (without live ollama network block for fast deterministic test)
    cut_res = client.post("/api/video/cut_frame", json={
        "video_path": sample_video_path,
        "frame_index": 12,
        "project_dir": temp_project_dir,
        "character_token": "BendyBot",
        "style_token": "vintage rubber hose",
        "auto_caption": False
    })
    assert cut_res.status_code == 200
    cut_data = cut_res.json()
    assert cut_data["success"] is True
    assert cut_data["frame"]["frame_index"] == 12
    assert Path(cut_data["frame"]["file_path"]).exists()

def test_api_captions_and_dataset(temp_project_dir):
    client.post("/api/projects/init", json={
        "project_dir": temp_project_dir,
        "project_name": "CaptionTest",
        "character_name": "BendyBot"
    })

    keyframes_dir = Path(temp_project_dir) / "Keyframes_Out"
    img_path = keyframes_dir / "BendyBot_00001.png"
    img_path.write_bytes(b"dummy")

    # Update caption
    up_res = client.put("/api/datasets/caption", json={
        "image_path": str(img_path),
        "caption": "BendyBot, smiling cel scan"
    })
    assert up_res.status_code == 200

    # List frames
    frames_res = client.get(f"/api/datasets/frames?project_dir={temp_project_dir}")
    assert frames_res.status_code == 200
    frames_data = frames_res.json()
    assert frames_data["total_count"] == 1
    assert frames_data["captioned_count"] == 1

def test_api_supervisor_endpoints(temp_project_dir):
    # Models discovery
    models_res = client.get("/api/supervisor/models")
    assert models_res.status_code == 200
    assert "best_vision_model" in models_res.json()

    # Chat with supervisor
    chat_res = client.post("/api/supervisor/chat", json={
        "message": "Which models are supported?",
        "project_dir": temp_project_dir
    })
    assert chat_res.status_code == 200
    assert "Model Vault" in chat_res.json()["reply"]

    # Script generation
    script_res = client.post("/api/supervisor/script", json={
        "character_name": "BendyBot",
        "premise": "Escaping an assembly line",
        "num_scenes": 2,
        "project_dir": temp_project_dir
    })
    assert script_res.status_code == 200
    assert len(script_res.json()["script"]["scenes"]) == 2

def test_api_training_endpoints(temp_project_dir):
    client.post("/api/projects/init", json={
        "project_dir": temp_project_dir,
        "project_name": "TrainAPITest",
        "character_name": "BendyBot"
    })

    # Profile FLUX
    prof_res = client.get("/api/training/profile?base_model_id=flux-1-dev")
    assert prof_res.status_code == 200
    assert prof_res.json()["recommended_parameters"]["rank"] == 16

    # Profile Qwen Image
    qwen_res = client.get("/api/training/profile?base_model_id=qwen-image")
    assert qwen_res.status_code == 200
    assert qwen_res.json()["model_info"]["name"] == "Qwen Image (DiT / Qwen2.5-VL)"

    # Profile Z-Image
    z_res = client.get("/api/training/profile?base_model_id=z-image")
    assert z_res.status_code == 200
    assert z_res.json()["model_info"]["name"] == "Z-Image (Zhipu / Zero-terminal DiT)"

    # Audit
    audit_res = client.get(f"/api/training/audit?project_dir={temp_project_dir}")
    assert audit_res.status_code == 200
    assert audit_res.json()["is_ready_for_training"] is False

    # Status
    status_res = client.get(f"/api/training/status?project_dir={temp_project_dir}")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "idle"
