from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.agents.trainer_agent import TrainerAgent
from app.services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["Training"])

class StartTrainingRequest(BaseModel):
    project_dir: str
    base_model: str = "flux-1-dev"
    lora_rank: int = 16
    lora_alpha: int = 16
    epochs: int = 5
    repeats: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-4
    execution_mode: str = "real_gpu"
    framing_mode: str = "bucket"

@router.get("/capabilities")
def get_training_capabilities():
    """Return only locally verified trainer/model combinations."""
    return TrainingService.get_training_capabilities()

@router.get("/audit")
def audit_dataset(project_dir: str = Query(...)):
    """Audits dataset readiness for LoRA training."""
    try:
        return TrainerAgent.audit_dataset_readiness(project_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/auto_tune")
def get_auto_tune(
    project_dir: str = Query(...),
    base_model: str = Query("sdxl-1.0")
):
    """Calculates hardware-optimized training hyperparameters adapted to GPU VRAM and dataset size."""
    from app.services.hardware_service import HardwareService
    from app.services.dataset_service import DatasetService
    ds = DatasetService.get_project_dataset(project_dir)
    return HardwareService.get_auto_tuned_params(
        base_model=base_model,
        dataset_frame_count=max(2, ds.get("total_count", 20))
    )

@router.get("/profile")
def get_training_profile(base_model_id: str = Query("flux-1-dev")):
    """Returns recommended training parameters for model architecture."""
    return TrainerAgent.get_training_profile(base_model_id)

@router.post("/start")
def start_training(req: StartTrainingRequest):
    """Starts LoRA training pipeline for project."""
    try:
        return TrainingService.start_training_pipeline(
            project_dir=req.project_dir,
            base_model=req.base_model,
            lora_rank=req.lora_rank,
            lora_alpha=req.lora_alpha,
            epochs=req.epochs,
            repeats=req.repeats,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            execution_mode=req.execution_mode,
            framing_mode=req.framing_mode
        )

    except Exception as e:
        import traceback
        from app.services.diagnostics_service import DiagnosticsService
        tb = traceback.format_exc()
        payload = req.model_dump()
        rec = DiagnosticsService.record_error(
            endpoint="/api/training/start",
            method="POST",
            error=e,
            traceback_str=tb,
            payload=payload,
            status_code=400,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(e),
                "error_type": type(e).__name__,
                "detail": f"{type(e).__name__}: {str(e)}",
                "traceback": tb,
                "endpoint": "/api/training/start",
                "request_payload": payload,
                "suggestion": rec.get("suggestion", "Check server logs or diagnostics report."),
            }
        )

@router.get("/status")
def get_training_status(project_dir: str = Query(...)):
    """Retrieves live training job status and loss metrics."""
    return TrainingService.get_job_status(project_dir)

@router.get("/manifest")
def get_dataset_manifest(project_dir: str = Query(..., description="Project directory path")):
    """Returns dataset preprocessing manifest showing upscaling, aspect ratio, and captions."""
    from app.services.dataset_service import DatasetService
    try:
        return DatasetService.get_dataset_manifest(project_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

