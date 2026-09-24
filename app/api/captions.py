from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.vision_service import VisionService
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/captions", tags=["Captions"])

class GenerateCaptionRequest(BaseModel):
    image_path: str
    trigger_token: str = "BendyBot"
    style_token: str = "vintage 1930s rubber hose cel animation"
    model_name: Optional[str] = None

class BatchCaptionRequest(BaseModel):
    project_dir: str
    model_name: Optional[str] = None
    force_overwrite: bool = False

@router.get("/models")
def get_vision_models():
    """Returns detected Ollama vision models and recommended model."""
    all_models = VisionService.list_installed_models()
    best = VisionService.get_best_vision_model()
    vision_only = [m for m in all_models if m["is_vision"]]

    return {
        "best_vision_model": best,
        "vision_models": vision_only,
        "all_installed_models": all_models
    }

@router.post("/generate")
def generate_caption(req: GenerateCaptionRequest):
    """Generates LoRA caption for an image and updates the matching .txt file."""
    try:
        res = VisionService.generate_caption(
            image_path=req.image_path,
            trigger_token=req.trigger_token,
            style_token=req.style_token,
            model_name=req.model_name
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
def batch_caption(req: BatchCaptionRequest):
    """Batches caption generation for uncaptioned frames in project."""
    p_dir = Path(req.project_dir)
    if not p_dir.exists():
        raise HTTPException(status_code=404, detail="Project directory not found")

    meta = DatasetService.load_project_meta(req.project_dir) or {}
    char_token = meta.get("trigger_token", "BendyBot")
    style_tok = meta.get("style_token", "vintage 1930s rubber hose cel animation")

    dataset = DatasetService.get_project_dataset(req.project_dir)
    frames_to_process = dataset["frames"] if req.force_overwrite else [f for f in dataset["frames"] if not f["has_caption"]]

    results = []
    for f in frames_to_process:
        res = VisionService.generate_caption(
            image_path=f["file_path"],
            trigger_token=char_token,
            style_token=style_tok,
            model_name=req.model_name
        )
        results.append(res)

    return {
        "processed_count": len(results),
        "results": results
    }
