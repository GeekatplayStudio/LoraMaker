from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/api/evaluation", tags=["LoRA Evaluation & Diagnostics"])

class RenderTestRequest(BaseModel):
    project_dir: str
    prompt: str
    negative_prompt: Optional[str] = ""
    lora_scale: Optional[float] = 0.85
    seed: Optional[int] = 42
    steps: Optional[int] = 30
    model_filename: Optional[str] = None
    base_checkpoint: Optional[str] = None
    aspect_ratio: Optional[str] = "1:1"
    framing_mode: Optional[str] = "pad"

class DeployComfyRequest(BaseModel):
    project_dir: str
    model_filename: Optional[str] = None

@router.get("/base_checkpoints")
def list_base_checkpoints():
    """Lists available SDXL and diffusion base checkpoints installed in ComfyUI."""
    try:
        checkpoints = EvaluationService.get_available_base_checkpoints()
        return {
            "success": True,
            "count": len(checkpoints),
            "checkpoints": checkpoints
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
def list_models(project_dir: str = Query(..., description="Project directory path")):
    """Lists all trained .safetensors files in the project's Training/output folder."""
    try:
        models = EvaluationService.list_trained_models(project_dir)
        return {
            "success": True,
            "project_dir": project_dir,
            "count": len(models),
            "models": models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/inspect")
def inspect_model(
    project_dir: str = Query(..., description="Project directory path"),
    model_filename: Optional[str] = Query(None, description="Safetensors filename to inspect")
):
    """Deeply inspects tensors, rank, alpha, norms, and health grade of a trained LoRA."""
    try:
        res = EvaluationService.inspect_lora_model(project_dir, model_filename)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Inspection failed"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
def get_training_analytics(
    project_dir: str = Query(..., description="Project directory path"),
    model_filename: Optional[str] = Query(None, description="Safetensors filename")
):
    """Returns loss convergence telemetry, dataset stats, and hardware profiles."""
    try:
        res = EvaluationService.get_training_analytics(project_dir, model_filename)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts")
def get_prompts(project_dir: str = Query(..., description="Project directory path")):
    """Generates 6 categorized suggested test prompts tailored to the trained character."""
    try:
        res = EvaluationService.generate_suggested_prompts(project_dir)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test_render")
def test_render(req: RenderTestRequest):
    """Executes a live test render simulating the trained LoRA cel frame generation."""
    try:
        res = EvaluationService.render_test_sample(
            project_dir=req.project_dir,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            lora_scale=req.lora_scale if req.lora_scale is not None else 0.85,
            seed=req.seed if req.seed is not None else 42,
            steps=req.steps or 30,
            model_filename=req.model_filename,
            base_checkpoint=req.base_checkpoint,
            aspect_ratio=req.aspect_ratio or "1:1",
            framing_mode=req.framing_mode or "pad"
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deploy_comfyui")
def deploy_to_comfyui(req: DeployComfyRequest):
    """Copies trained LoRA to D:/ComfyUI/ComfyUI/models/loras/ and creates test workflow."""
    try:
        res = EvaluationService.deploy_to_comfyui(req.project_dir, req.model_filename)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Deployment failed"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
