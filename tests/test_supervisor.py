import pytest
from app.agents.supervisor import SupervisorAgent
from app.agents.visual_agent import VisualAgent
from app.agents.trainer_agent import TrainerAgent

def test_supervisor_get_available_models():
    models_info = SupervisorAgent.get_available_models()
    assert "best_vision_model" in models_info
    assert "supported_lora_architectures" in models_info
    # The supervisor must expose only locally executable backends, not the
    # aspirational architecture catalogue.
    assert {model["id"] for model in models_info["supported_lora_architectures"]} <= {"sdxl-1.0", "flux-1-dev", "minimax-video", "qwen-image", "z-image"}
    assert models_info["recommended_video_model"] in {None, "minimax-video"}

def test_supervisor_start_new_project(temp_project_dir):
    res = SupervisorAgent.start_new_project(
        project_dir=temp_project_dir,
        project_name="BendyStudio",
        character_name="BendyBot"
    )
    assert res["meta"]["project_name"] == "BendyStudio"
    assert "Keyframes_Out" in res["meta"]["folder_structure"]
    assert "BendyBot" in res["explanation"]

def test_supervisor_chat_model_intent():
    res = SupervisorAgent.process_chat_message("What models are supported?")
    assert res["tool_called"] == "get_available_models"
    assert "Model Vault" in res["reply"]
    assert "FLUX.1" in res["reply"]
    assert "Wan2.1" in res["reply"]

def test_visual_agent_guidance():
    guidance = VisualAgent.get_style_tokens_and_lora_advice("BendyBot", "1930s Rubber Hose Cel")
    assert guidance["character_token"] == "BendyBot"
    assert "vintage 1930s" in guidance["style_tokens"]
    assert guidance["recommended_base_model"] == "sdxl-1.0"
    assert guidance["lora_rank"] == 32

def test_trainer_agent_audit(temp_project_dir):
    SupervisorAgent.start_new_project(temp_project_dir, "AuditTest", "BendyBot")
    audit = TrainerAgent.audit_dataset_readiness(temp_project_dir)
    assert audit["is_ready_for_training"] is False
    assert len(audit["issues"]) > 0
