import os
import sys
import shutil
import tempfile
import cv2
import numpy as np
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(scope="session")
def sample_video_path(tmp_path_factory):
    """
    Generates a synthetic MP4 video file with animated retro shapes
    for testing video seeking, metadata extraction, and frame cutting.
    """
    tmp_dir = tmp_path_factory.mktemp("video_data")
    video_file = tmp_dir / "sample_retro_cartoon.mp4"

    width, height = 512, 512
    fps = 24.0
    num_frames = 48  # 2 seconds of video

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a vintage grayscale / sepia tinted frame
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        
        # Draw bouncing retro cartoon pie-circle character
        center_x = int(256 + 100 * np.sin(i * 0.2))
        center_y = int(256 + 60 * np.cos(i * 0.2))
        cv2.circle(frame, (center_x, center_y), 50, (230, 230, 230), -1)
        cv2.circle(frame, (center_x - 15, center_y - 10), 10, (20, 20, 20), -1)
        cv2.circle(frame, (center_x + 15, center_y - 10), 10, (20, 20, 20), -1)

        # Draw frame number watermark
        cv2.putText(frame, f"Frame {i:03d}", (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        out.write(frame)

    out.release()
    return str(video_file)

@pytest.fixture
def temp_project_dir():
    """Provides a fresh temporary project directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="retro_test_project_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(autouse=True)
def mock_external_heavy_services(monkeypatch):
    """Ensure automated tests run cleanly, deterministically, and fast without external model loading."""
    from app.services.evaluation_service import EvaluationService
    from app.agents.story_agent import StoryAgent
    monkeypatch.setattr(EvaluationService, "get_real_diffusion_pipe", classmethod(lambda cls: None))
    monkeypatch.setattr(StoryAgent, "_get_best_llm_model", classmethod(lambda cls: None))


