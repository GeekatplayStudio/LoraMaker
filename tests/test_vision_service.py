import os
from pathlib import Path
import pytest
from app.services.vision_service import VisionService

def test_list_installed_models():
    models = VisionService.list_installed_models()
    assert isinstance(models, list)
    # On this system, Ollama is running and has models
    if models:
        assert "name" in models[0]
        assert "is_vision" in models[0]

def test_get_best_vision_model():
    best = VisionService.get_best_vision_model()
    assert isinstance(best, str)
    assert len(best) > 0
    # Should identify one of the installed vision models (e.g. qwen2.5vl:7b, qwen3-vl:4b, llama3.2-vision)
    assert any(term in best for term in ["vl", "vision", "llava", "qwen"])

def test_caption_file_handling(temp_project_dir):
    test_img = Path(temp_project_dir) / "test_frame.png"
    test_img.write_bytes(b"dummy_png_bytes")

    caption_text = "BendyBot, vintage 1930s rubber hose cel animation, pie eyes, smiling"
    txt_path = VisionService.save_caption_file(str(test_img), caption_text)

    assert txt_path.exists()
    assert txt_path.name == "test_frame.txt"
    assert VisionService.read_caption_file(str(test_img)) == caption_text
