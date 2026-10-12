import os
import cv2
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class VideoService:
    """
    Video extraction, seeking, and frame curation service for LoRA dataset generation.
    Supports frame-by-frame scrubbing, high-precision frame capture, and pose suggestion.
    """

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Extracts comprehensive metadata from a video file using OpenCV.
        """
        p = Path(video_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0

            return {
                "file_path": str(p.resolve()),
                "file_name": p.name,
                "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "aspect_ratio": f"{width}:{height}",
                "duration_seconds": round(duration, 2),
                "duration_formatted": VideoService._format_time(duration)
            }
        finally:
            cap.release()

    @staticmethod
    def get_frame(video_path: str, frame_index: int, format: str = "jpeg", quality: int = 85) -> Optional[str]:
        """
        Retrieves a single frame at frame_index and returns it as a base64 encoded data URI.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_index < 0:
                frame_index = 0
            elif frame_index >= total_frames and total_frames > 0:
                frame_index = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                return None

            ext = ".jpg" if format.lower() in ["jpg", "jpeg"] else ".png"
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext == ".jpg" else [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            success, encoded_img = cv2.imencode(ext, frame, encode_param)
            if not success:
                return None

            b64_str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
            mime = "image/jpeg" if ext == ".jpg" else "image/png"
            return f"data:{mime};base64,{b64_str}"
        finally:
            cap.release()

    @staticmethod
    def extract_and_save_frame(
        video_path: str,
        frame_index: int,
        output_dir: str,
        character_token: str = "character",
        custom_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts a single frame and saves it to output_dir with standardized naming for LoRA training:
        e.g., '{character_token}_{frame_index:05d}.png'
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Failed to read frame at index {frame_index}")

            # Normalize character token for clean file naming
            safe_token = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in character_token).strip("_")
            if not safe_token:
                safe_token = "cel_scan"

            if custom_name:
                filename = custom_name if custom_name.endswith(".png") else f"{custom_name}.png"
            else:
                filename = f"{safe_token}_{frame_index:05d}.png"

            file_destination = out_path / filename

            # Write lossless PNG
            success = cv2.imwrite(str(file_destination), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
            if not success:
                raise IOError(f"Could not write image to {file_destination}")

            h, w = frame.shape[:2]
            timestamp = frame_index / fps if fps > 0 else 0.0

            return {
                "success": True,
                "file_name": filename,
                "file_path": str(file_destination.resolve()),
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp, 3),
                "timestamp_formatted": VideoService._format_time(timestamp),
                "width": w,
                "height": h,
                "character_token": safe_token
            }
        finally:
            cap.release()

    @staticmethod
    def suggest_keyframe_candidates(video_path: str, max_candidates: int = 8) -> List[Dict[str, Any]]:
        """
        Measures frame-to-frame change across the timeline and returns the
        frames where the picture changes most (cuts, big pose changes).  Each
        candidate carries its measured change score; no pose labels are invented.
        """
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return []

            # Sample at most ~400 frames evenly to keep analysis fast.
            stride = max(1, total_frames // 400)
            prev = None
            scores: List[tuple] = []
            for f_idx in range(0, total_frames, stride):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
                if prev is not None:
                    scores.append((f_idx, float(np.mean(np.abs(small - prev)))))
                prev = small
            if not scores:
                return []

            # Pick the strongest changes, keeping candidates spread apart.
            min_gap = max(1, total_frames // (max_candidates * 2))
            chosen: List[tuple] = []
            for f_idx, score in sorted(scores, key=lambda x: x[1], reverse=True):
                if all(abs(f_idx - c[0]) >= min_gap for c in chosen):
                    chosen.append((f_idx, score))
                if len(chosen) >= max_candidates:
                    break
            max_score = max(sc for _, sc in scores) or 1.0

            candidates = []
            for f_idx, score in sorted(chosen):
                sec = f_idx / fps
                candidates.append({
                    "frame_index": f_idx,
                    "timestamp": round(sec, 2),
                    "timestamp_formatted": VideoService._format_time(sec),
                    "change_score": round(score / max_score, 3),
                    "reason": f"Visual change {round(100 * score / max_score)}% of the strongest change in this video at {VideoService._format_time(sec)}"
                })
            return candidates
        finally:
            cap.release()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formats seconds into HH:MM:SS.mmm string."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"
