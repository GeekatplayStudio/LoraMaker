from typing import Dict, Any
from app.services.video_service import VideoService

class VideoAgent:
    """
    Video Specialist Agent:
    Handles video ingestion, motion metrics, resolution bucketing,
    and preparation for Wan2.1 and LTX-Video diffusion architectures.
    """

    @staticmethod
    def analyze_source_video(video_path: str) -> Dict[str, Any]:
        """
        Analyzes source video and evaluates suitability for LoRA training and motion extraction.
        """
        info = VideoService.get_video_info(video_path)
        fps = info["fps"]
        width = info["width"]
        height = info["height"]

        # Compute recommended bucketing
        aspect = width / height if height > 0 else 1.0
        if 0.95 <= aspect <= 1.05:
            bucket = "1:1 (Square, 1024x1024 or 512x512)"
        elif aspect > 1.3:
            bucket = "16:9 or 4:3 (Landscape, ideal for Wan2.1 / LTX-Video)"
        else:
            bucket = "9:16 or 3:4 (Portrait)"

        recommendations = []
        if fps < 20:
            recommendations.append("Video has low frame rate (<20 fps); keyframes may have motion blur.")
        if width < 512 or height < 512:
            recommendations.append("Resolution is lower than 512px; consider AI upscaling prior to training.")
        else:
            recommendations.append(f"Resolution {width}x{height} is optimal for high-fidelity cel scans.")

        return {
            **info,
            "recommended_bucket": bucket,
            "quality_assessment": "High" if width >= 720 and fps >= 23 else "Standard",
            "video_agent_notes": recommendations
        }
