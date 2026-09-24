import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.video_service import VideoService
from app.services.vision_service import VisionService
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/video", tags=["Video"])

class CutFrameRequest(BaseModel):
    video_path: str
    frame_index: int
    project_dir: str
    character_token: Optional[str] = None
    style_token: Optional[str] = None
    auto_caption: bool = True
    vision_model: Optional[str] = None

@router.get("/info")
def get_video_info(video_path: str = Query(..., description="Path to video file")):
    """Retrieves metadata (fps, duration, resolution, total frames) for video."""
    try:
        return VideoService.get_video_info(video_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stream")
def stream_video(video_path: str = Query(...)):
    """Serves video file directly for native HTML5 video player."""
    p = Path(video_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(p, media_type="video/mp4", filename=p.name)

@router.get("/frame")
def get_frame(
    video_path: str = Query(...),
    frame_index: int = Query(...)
):
    """Returns base64 data URI of frame at specified frame index for exact scrubber display."""
    data_uri = VideoService.get_frame(video_path, frame_index)
    if not data_uri:
        raise HTTPException(status_code=404, detail="Could not retrieve frame at specified index")
    return {"frame_index": frame_index, "image_data": data_uri}

@router.post("/cut_frame")
def cut_frame(req: CutFrameRequest):
    """
    Extracts high-resolution frame from video, saves it to Keyframes_Out,
    and automatically triggers Ollama vision to generate the corresponding .txt caption.
    """
    p_dir = Path(req.project_dir)
    keyframes_out = p_dir / "Keyframes_Out"
    if not keyframes_out.exists():
        keyframes_out.mkdir(parents=True, exist_ok=True)

    meta = DatasetService.load_project_meta(req.project_dir) or {}
    char_token = req.character_token or meta.get("trigger_token", "BendyBot")
    style_tok = req.style_token or meta.get("style_token", "vintage 1930s rubber hose cel animation")

    try:
        # 1. Extract frame
        saved_frame = VideoService.extract_and_save_frame(
            video_path=req.video_path,
            frame_index=req.frame_index,
            output_dir=str(keyframes_out.resolve()),
            character_token=char_token
        )

        caption_result = None
        # 2. Auto-caption if requested
        if req.auto_caption:
            caption_result = VisionService.generate_caption(
                image_path=saved_frame["file_path"],
                trigger_token=char_token,
                style_token=style_tok,
                model_name=req.vision_model
            )

        return {
            "success": True,
            "frame": saved_frame,
            "caption": caption_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_video(
    project_dir: str = Form(...),
    file: UploadFile = File(...)
):
    """Uploads source video file into project's Input_Video directory."""
    p_dir = Path(project_dir)
    input_video_dir = p_dir / "Input_Video"
    input_video_dir.mkdir(parents=True, exist_ok=True)

    target_path = input_video_dir / file.filename
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    info = VideoService.get_video_info(str(target_path.resolve()))
    return {
        "success": True,
        "file_name": file.filename,
        "saved_path": str(target_path.resolve()),
        "video_info": info
    }
