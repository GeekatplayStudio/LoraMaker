import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.api import (
    projects_router,
    video_router,
    captions_router,
    datasets_router,
    supervisor_router,
    training_router,
    evaluation_router,
    system_router
)

app = FastAPI(
    title=settings.APP_NAME,
    description="Universal AI LoRA Assets Maker, Frame Scrubber, and Multi-Agent Creative Studio by Geekatplay Studio (Vladimir Chopine)",
    version=settings.VERSION
)

# Enable CORS for local desktop usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(projects_router)
app.include_router(video_router)
app.include_router(captions_router)
app.include_router(datasets_router)
app.include_router(supervisor_router)
app.include_router(training_router)
app.include_router(evaluation_router)
app.include_router(system_router)

# Static files path
STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": f"{settings.APP_NAME} Backend Running",
        "docs_url": "/docs",
        "version": settings.VERSION
    }
