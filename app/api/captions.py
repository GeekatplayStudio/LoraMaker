import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
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

class _BatchJobs:
    """Background batch-caption jobs keyed by project directory."""
    _lock = threading.Lock()
    _jobs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls, project_dir: str) -> Dict[str, Any]:
        with cls._lock:
            job = cls._jobs.get(project_dir)
            return dict(job) if job else {"status": "idle", "processed_count": 0, "total": 0, "failed_count": 0, "results": []}

    @classmethod
    def start(cls, project_dir: str, frames: List[Dict[str, Any]], char_token: str, style_tok: str, model_name: Optional[str]) -> Dict[str, Any]:
        with cls._lock:
            existing = cls._jobs.get(project_dir)
            if existing and existing["status"] == "running":
                raise HTTPException(status_code=409, detail="A batch captioning job is already running for this project.")
            job = {"status": "running", "processed_count": 0, "failed_count": 0, "total": len(frames),
                   "results": [], "model_name": model_name, "started_at": time.time()}
            cls._jobs[project_dir] = job

        def run():
            for f in frames:
                with cls._lock:
                    if job["status"] == "cancelled":
                        return
                res = VisionService.generate_caption(
                    image_path=f["file_path"], trigger_token=char_token, style_token=style_tok, model_name=model_name
                )
                with cls._lock:
                    job["processed_count"] += 1
                    if not res.get("success"):
                        job["failed_count"] += 1
                    job["results"].append(res)
            with cls._lock:
                job["status"] = "completed"
                job["finished_at"] = time.time()

        threading.Thread(target=run, daemon=True, name=f"batch-caption-{Path(project_dir).name}").start()
        return dict(job)

    @classmethod
    def cancel(cls, project_dir: str) -> Dict[str, Any]:
        with cls._lock:
            job = cls._jobs.get(project_dir)
            if job and job["status"] == "running":
                job["status"] = "cancelled"
            return dict(job) if job else {"status": "idle"}


@router.post("/batch")
def batch_caption(req: BatchCaptionRequest):
    """Starts a background captioning job for uncaptioned frames; poll /batch/status.

    Each Ollama call can take tens of seconds, so a synchronous loop over a
    whole dataset would exceed any sensible HTTP timeout.
    """
    p_dir = Path(req.project_dir).expanduser().resolve()
    if not (p_dir / "project.json").is_file():
        raise HTTPException(status_code=404, detail="Project directory not found")
    key = str(p_dir)

    meta = DatasetService.load_project_meta(key) or {}
    char_token = meta.get("trigger_token", "BendyBot")
    style_tok = meta.get("style_token", "vintage 1930s rubber hose cel animation")

    dataset = DatasetService.get_project_dataset(key)
    frames = dataset["frames"] if req.force_overwrite else [f for f in dataset["frames"] if not f["has_caption"]]
    if not frames:
        return {"status": "completed", "processed_count": 0, "total": 0, "failed_count": 0, "results": [],
                "message": "All frames are already captioned."}
    job = _BatchJobs.start(key, frames, char_token, style_tok, req.model_name)
    return {**job, "message": f"Captioning {len(frames)} frame(s) in the background."}

@router.get("/batch/status")
def batch_caption_status(project_dir: str = Query(...)):
    """Progress of the background captioning job for a project."""
    return _BatchJobs.get(str(Path(project_dir).expanduser().resolve()))

@router.post("/batch/cancel")
def batch_caption_cancel(req: BatchCaptionRequest):
    """Stop a running batch after the current frame finishes."""
    return _BatchJobs.cancel(str(Path(req.project_dir).expanduser().resolve()))
