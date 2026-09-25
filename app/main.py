import os
import traceback
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from app.core.config import settings
from app.services.diagnostics_service import DiagnosticsService
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

# Initialize logging capture for diagnostics console
DiagnosticsService.setup_logging_hook()

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        orig_handler = loop.get_exception_handler()

        def ignore_windows_stream_disconnects(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
                return
            if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (10054, 10053):
                return
            msg = str(context.get("message", ""))
            if "10054" in msg or "_call_connection_lost" in msg:
                return
            if orig_handler:
                orig_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(ignore_windows_stream_disconnects)
    except Exception:
        pass
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="Universal AI LoRA Assets Maker, Frame Scrubber, and Multi-Agent Creative Studio by Geekatplay Studio (Vladimir Chopine)",
    version=settings.VERSION,
    lifespan=lifespan
)

# Global Exception Handlers for clear, actionable diagnostics
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    rec = DiagnosticsService.record_error(
        endpoint=str(request.url.path),
        method=request.method,
        error=exc,
        traceback_str=tb,
        status_code=500
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {str(exc)}",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
            "endpoint": str(request.url.path),
            "suggestion": rec.get("suggestion", "Check server logs or diagnostics report."),
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(detail),
            "message": str(detail),
            "error_type": "HTTPException",
            "endpoint": str(request.url.path),
            "status_code": exc.status_code,
        }
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
