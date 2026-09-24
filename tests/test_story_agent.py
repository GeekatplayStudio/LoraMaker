from pathlib import Path
import pytest
from app.agents.story_agent import StoryAgent
from app.services.dataset_service import DatasetService

def test_story_agent_generation(temp_project_dir):
    DatasetService.initialize_project(temp_project_dir, "StoryProj", "BendyBot")
    script = StoryAgent.generate_animation_script(
        character_name="BendyBot",
        premise="BendyBot escapes an assembly line in 1930",
        num_scenes=3,
        style_preset="vintage 1930s rubber hose cel animation"
    )

    assert script is not None
    assert "character" in script
    assert script["character"] == "BendyBot"
    assert "scenes" in script
    assert len(script["scenes"]) == 3

    for scene in script["scenes"]:
        assert "scene_number" in scene
        assert "duration_sec" in scene
        assert "camera_motion" in scene
        assert "diffusion_prompt" in scene
        assert "BendyBot" in scene["diffusion_prompt"]

    # Test saving to project
    saved_path = StoryAgent.save_script_to_project(temp_project_dir, script)
    assert Path(saved_path).exists()
    assert Path(saved_path).name == "animation_script.json"
