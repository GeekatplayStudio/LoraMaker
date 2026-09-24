from app.api.projects import router as projects_router
from app.api.video import router as video_router
from app.api.captions import router as captions_router
from app.api.datasets import router as datasets_router
from app.api.supervisor import router as supervisor_router
from app.api.training import router as training_router
from app.api.evaluation import router as evaluation_router
from app.api.system import router as system_router

__all__ = [
    "projects_router",
    "video_router",
    "captions_router",
    "datasets_router",
    "supervisor_router",
    "training_router",
    "evaluation_router",
    "system_router"
]

