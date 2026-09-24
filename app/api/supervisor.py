from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.agents.supervisor import SupervisorAgent
from app.agents.visual_agent import VisualAgent
from app.agents.story_agent import StoryAgent

router = APIRouter(prefix="/api/supervisor", tags=["Supervisor"])

class ChatRequest(BaseModel):
    message: str
    project_dir: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

class ScriptRequest(BaseModel):
    character_name: str = "BendyBot"
    premise: str = "Escaping a vintage assembly line"
    num_scenes: int = 3
    style_preset: str = "vintage 1930s rubber hose cel animation"
    project_dir: Optional[str] = None

@router.get("/models")
def get_models():
    """Supervisor tool: returns all available local and diffusion models."""
    return SupervisorAgent.get_available_models()

@router.post("/chat")
def chat_with_supervisor(req: ChatRequest):
    """Processes user message with Supervisor Agent orchestrator."""
    return SupervisorAgent.process_chat_message(
        user_message=req.message,
        project_dir=req.project_dir,
        history=req.history
    )

@router.post("/script")
def generate_story_script(req: ScriptRequest):
    """Delegates animation script generation to Story Agent and saves to project."""
    script = StoryAgent.generate_animation_script(
        character_name=req.character_name,
        premise=req.premise,
        num_scenes=req.num_scenes,
        style_preset=req.style_preset
    )
    saved_path = None
    if req.project_dir:
        saved_path = StoryAgent.save_script_to_project(req.project_dir, script)

    return {
        "script": script,
        "saved_path": saved_path
    }

@router.get("/style_guidance")
def get_style_guidance(
    character_name: str = Query("BendyBot"),
    preset: str = Query("1930s Rubber Hose Cel")
):
    """Visual Agent advice on style tokens, negative prompts, and LoRA hyperparameters."""
    return VisualAgent.get_style_tokens_and_lora_advice(character_name, preset)
