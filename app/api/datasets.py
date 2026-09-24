from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

class UpdateCaptionRequest(BaseModel):
    image_path: str
    caption: str

class DeleteFrameRequest(BaseModel):
    image_path: str

class ExportKohyaRequest(BaseModel):
    project_dir: str
    repeats: int = 10
    class_token: str = "character"

class ExportComfyUIRequest(BaseModel):
    project_dir: str

@router.get("/frames")
def list_project_frames(project_dir: str = Query(...)):
    """Returns all extracted keyframe scans and corresponding captions in project."""
    p_path = Path(project_dir)
    if not p_path.exists():
        raise HTTPException(status_code=404, detail="Project directory not found")

    return DatasetService.get_project_dataset(project_dir)

@router.get("/image")
def get_frame_image(image_path: str = Query(...)):
    """Serves the thumbnail/preview image for a keyframe."""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(p, media_type=media)

@router.put("/caption")
def update_caption(req: UpdateCaptionRequest):
    """Updates the text caption file for a specific frame."""
    try:
        return DatasetService.update_frame_caption(req.image_path, req.caption)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/frame")
def delete_frame(req: DeleteFrameRequest):
    """Deletes frame image and its corresponding .txt caption file."""
    try:
        return DatasetService.delete_frame_pair(req.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export/kohya")
def export_kohya(req: ExportKohyaRequest):
    """Exports dataset to Kohya_ss sd-scripts format."""
    try:
        return DatasetService.export_for_kohya(
            project_dir=req.project_dir,
            repeats=req.repeats,
            class_token=req.class_token
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/export/comfyui")
def export_comfyui(req: ExportComfyUIRequest):
    """Exports dataset to ComfyUI LoRA training node format with JSONL metadata."""
    try:
        return DatasetService.export_for_comfyui(project_dir=req.project_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
