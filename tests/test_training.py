from pathlib import Path
import pytest
from app.services.training_service import TrainingService
from app.services.dataset_service import DatasetService

def test_generate_kohya_config(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "TrainProj", "BendyBot")
    toml_path = TrainingService.generate_kohya_toml_config(
        project_dir=temp_project_dir,
        base_model="flux-1-dev",
        lora_rank=16,
        lora_alpha=16,
        learning_rate=1e-4,
        max_train_epochs=10
    )

    p = Path(toml_path)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "flux-1-dev" in content
    assert "network_dim = 16" in content
    assert 'output_name = "BendyBot_flux-1-dev_lora"' in content

def test_generate_comfyui_workflow(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "ComfyTrain", "BendyBot")
    wf_path = TrainingService.generate_comfyui_workflow(
        project_dir=temp_project_dir,
        base_model="wan-2.1-t2v",
        lora_rank=32,
        lora_alpha=32
    )

    p = Path(wf_path)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "wan-2.1-t2v" in content
    assert '"executable": false' in content
    assert "No installed ComfyUI trainer nodes were verified" in content

def test_start_training_rejects_invalid_images_and_never_falls_back(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "PipelineTest", "BendyBot")
    keyframes_dir = Path(temp_project_dir) / "Keyframes_Out"

    # Add 2 sample captioned frames
    for i in range(1, 3):
        (keyframes_dir / f"BendyBot_{i:04d}.png").write_bytes(b"image")
        (keyframes_dir / f"BendyBot_{i:04d}.txt").write_text(f"BendyBot, frame {i}", encoding="utf-8")

    with pytest.raises(ValueError, match="valid, fully captioned keyframes"):
        TrainingService.start_training_pipeline(project_dir=temp_project_dir, base_model="sdxl-1.0", execution_mode="validate_only")
    assert not list((Path(temp_project_dir) / "Training" / "output").glob("*.safetensors"))

def test_unsupported_architectures_are_recorded_but_not_claimed_executable(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "QwenZProj", "BendyBot")
    
    # Test Qwen Image Kohya config
    qwen_toml = TrainingService.generate_kohya_toml_config(
        project_dir=temp_project_dir,
        base_model="qwen-image",
        lora_rank=16,
        lora_alpha=16
    )
    assert Path(qwen_toml).exists()
    content = Path(qwen_toml).read_text(encoding="utf-8")
    assert 'trainer_supported = false' in content

    # Test Z-Image ComfyUI workflow
    zimage_wf = TrainingService.generate_comfyui_workflow(
        project_dir=temp_project_dir,
        base_model="z-image",
        lora_rank=32,
        lora_alpha=16
    )
    assert Path(zimage_wf).exists()
    wf_content = Path(zimage_wf).read_text(encoding="utf-8")
    assert '"executable": false' in wf_content
    assert "z-image" in wf_content
