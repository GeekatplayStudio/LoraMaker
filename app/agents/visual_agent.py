from typing import Dict, Any, List, Optional
from app.services.vision_service import VisionService
from app.services.video_service import VideoService

class VisualAgent:
    """
    Visual Specialist Agent:
    Curates vintage aesthetics, generates LoRA style tokens, analyzes poses,
    and captions cel scans using local Ollama vision models.
    """

    @staticmethod
    def get_style_tokens_and_lora_advice(
        character_name: str,
        style_preset: str = "1930s Rubber Hose Cel"
    ) -> Dict[str, Any]:
        """
        Recommends trigger words, style tags, and LoRA hyperparameters for character visual consistency.
        """
        clean_token = "".join(c for c in character_name if c.isalnum() or c in ("-", "_")).strip() or "BendyBot"

        presets = {
            "1930s Rubber Hose Cel": {
                "trigger_token": clean_token,
                "style_tokens": "vintage 1930s rubber hose animation, monochrome cel animation, retro black and white cartoon, ink lines, fleischer style",
                "recommended_base_model": "sdxl-1.0",
                "lora_rank": 32,
                "lora_alpha": 16,
                "negative_prompt": "modern 3D render, photorealistic, color photograph, blurry, digital illustration, noisy artifacts"
            },
            "Vintage Technicolor 1940s": {
                "trigger_token": clean_token,
                "style_tokens": "1940s vintage technicolor cel animation, vibrant classic cartoon palette, hand-painted gouache background, cel shade",
                "recommended_base_model": "flux-1-dev",
                "lora_rank": 16,
                "lora_alpha": 16,
                "negative_prompt": "photorealistic, modern cgi, deformed, bad anatomy, flat digital vector"
            },
            "Wan Video Motion LoRA": {
                "trigger_token": clean_token,
                "style_tokens": "smooth 24fps vintage cartoon movement, dynamic squash and stretch, camera panning",
                "recommended_base_model": "wan-2.1-t2v",
                "lora_rank": 32,
                "lora_alpha": 32,
                "negative_prompt": "jittery motion, video artifacts, temporal flicker, morphed limbs"
            }
        }

        selected = presets.get(style_preset, presets["1930s Rubber Hose Cel"])
        return {
            "character_token": clean_token,
            "style_preset": style_preset,
            **selected,
            "guidelines": [
                f"Always prepend '{clean_token}' to your character generation prompts.",
                "Ensure training data has turnaround angles: Front, 3/4 View, Profile, and Back.",
                "Include at least 2 extreme expression cel scans (surprised, laughing, scheming)."
            ]
        }

    @staticmethod
    def analyze_video_poses(video_path: str) -> List[Dict[str, Any]]:
        """Scans video and suggests keyframe candidates for character LoRA turnaround."""
        return VideoService.suggest_keyframe_candidates(video_path, max_candidates=8)

    @staticmethod
    def caption_cel_frame(
        image_path: str,
        character_token: str,
        style_token: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delegates caption generation to the best local Ollama vision model."""
        return VisionService.generate_caption(
            image_path=image_path,
            trigger_token=character_token,
            style_token=style_token,
            model_name=model_name
        )
