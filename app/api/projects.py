import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.core.config import settings
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/projects", tags=["Projects"])

class ProjectInitRequest(BaseModel):
    project_dir: Optional[str] = None
    project_name: str = "Studio_Project"
    character_name: str = "Character"
    trigger_token: Optional[str] = None
    style_description: str = "photorealistic portrait, natural lighting"

@router.post("/init")
def initialize_project(req: ProjectInitRequest):
    """Initializes standard project directory structure and metadata."""
    if not req.project_dir:
        # Default into workspace projects folder
        p_dir = settings.DEFAULT_PROJECTS_DIR / req.project_name
    else:
        p_dir = Path(req.project_dir)

    try:
        meta = DatasetService.initialize_project(
            project_dir=str(p_dir.resolve()),
            project_name=req.project_name,
            character_name=req.character_name,
            style_description=req.style_description,
            trigger_token=req.trigger_token
        )
        return {"success": True, "project": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
def list_projects():
    """Lists projects created in default projects directory."""
    projects_dir = settings.DEFAULT_PROJECTS_DIR
    if not projects_dir.exists():
        return {"projects": []}

    projects = []
    for item in projects_dir.iterdir():
        if item.is_dir():
            meta = DatasetService.load_project_meta(str(item))
            if meta:
                summary = DatasetService.get_project_dataset(str(item))
                projects.append({
                    **meta,
                    "project_dir": str(item.resolve()),
                    "total_frames": summary["total_count"],
                    "captioned_frames": summary["captioned_count"]
                })
    return {"projects": projects}

@router.get("/info")
def get_project_info(project_dir: str = Query(..., description="Absolute path to project directory")):
    """Returns metadata and dataset statistics for given project folder."""
    p_path = Path(project_dir)
    if not p_path.exists():
        raise HTTPException(status_code=404, detail="Project directory not found")

    meta = DatasetService.load_project_meta(str(p_path))
    dataset_summary = DatasetService.get_project_dataset(str(p_path))

    return {
        "metadata": meta,
        "dataset": dataset_summary
    }
