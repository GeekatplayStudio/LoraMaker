import os
from pathlib import Path
import pytest
from app.services.video_service import VideoService

def test_video_metadata_extraction(sample_video_path):
    info = VideoService.get_video_info(sample_video_path)
    assert info["file_name"] == "sample_retro_cartoon.mp4"
    assert info["width"] == 512
    assert info["height"] == 512
    assert info["total_frames"] == 48
    assert info["fps"] == 24.0
    assert info["duration_seconds"] == 2.0
    assert "00:00:02" in info["duration_formatted"]

def test_get_frame_base64(sample_video_path):
    data_uri = VideoService.get_frame(sample_video_path, frame_index=10)
    assert data_uri is not None
    assert data_uri.startswith("data:image/jpeg;base64,")
    assert len(data_uri) > 100

def test_extract_and_save_frame(sample_video_path, temp_project_dir):
    out_dir = Path(temp_project_dir) / "Keyframes_Out"
    res = VideoService.extract_and_save_frame(
        video_path=sample_video_path,
        frame_index=15,
        output_dir=str(out_dir),
        character_token="BendyBot"
    )
    assert res["success"] is True
    assert res["character_token"] == "BendyBot"
    assert res["frame_index"] == 15
    assert res["width"] == 512
    assert res["height"] == 512

    saved_file = Path(res["file_path"])
    assert saved_file.exists()
    assert saved_file.name == "BendyBot_00015.png"

def test_suggest_keyframe_candidates(sample_video_path):
    candidates = VideoService.suggest_keyframe_candidates(sample_video_path, max_candidates=4)
    assert len(candidates) > 0
    for c in candidates:
        assert "frame_index" in c
        assert "timestamp" in c
        assert "suggested_pose" in c
