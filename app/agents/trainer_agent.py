from typing import Dict, Any, List
from app.services.dataset_service import DatasetService
from app.core.config import settings

class TrainerAgent:
    """
    Trainer Specialist Agent:
    Validates dataset health, caption quality, and recommends fine-tuned LoRA hyperparameters
    for Kohya_ss and ComfyUI pipelines across FLUX, SDXL, and Wan video models.
    """

    @staticmethod
    def audit_dataset_readiness(project_dir: str) -> Dict[str, Any]:
        """
        Audits the project dataset and checks readiness for training.
        """
        data = DatasetService.get_project_dataset(project_dir)
        total = data["total_count"]
        captioned = data["captioned_count"]
        uncaptioned = data["uncaptioned_count"]

        issues: List[str] = []
        recommendations: List[str] = []

        if total == 0:
            issues.append("No cel keyframes have been extracted yet. Use the Timeline Scrubber to cut frames.")
        elif total < 8:
            issues.append(f"Only {total} frames extracted. Recommended minimum is 10-15 keyframes for clean character LoRA identity.")
        
        if uncaptioned > 0:
            issues.append(f"{uncaptioned} frames are missing .txt caption descriptions. Run Ollama captioning on these frames.")

        is_ready = len(issues) == 0 and total >= 8 and captioned == total

        aspect_info = DatasetService.detect_dataset_aspect_ratio(project_dir)
        recommendations.append(
            f"Native format detected: {aspect_info['detected_ratio']} ({aspect_info['native_resolution'][0]}×{aspect_info['native_resolution'][1]}). Multi-aspect bucketing ensures zero squeezing or distortion."
        )

        if is_ready:
            recommendations.append("Dataset is fully captioned and ready for training!")
            recommendations.append("Estimated training time on RTX 3090: ~8 to 15 minutes for 1000 steps.")

        return {
            "is_ready_for_training": is_ready,
            "total_frames": total,
            "captioned_frames": captioned,
            "uncaptioned_frames": uncaptioned,
            "aspect_ratio": aspect_info["detected_ratio"],
            "native_resolution": aspect_info["native_resolution"],
            "aspect_float": aspect_info["aspect_float"],
            "issues": issues,
            "recommendations": recommendations
        }

    @staticmethod
    def get_training_profile(base_model_id: str) -> Dict[str, Any]:
        """Returns recommended hyperparameters for chosen architecture."""
        matching = next((m for m in settings.SUPPORTED_LORA_ARCHITECTURES if m["id"] == base_model_id), None)
        if not matching:
            matching = settings.SUPPORTED_LORA_ARCHITECTURES[0]

        profiles = {
            "flux-1-dev": {
                "rank": 16,
                "alpha": 16,
                "lr": 1e-4,
                "epochs": 10,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss or ai-toolkit"
            },
            "qwen-image": {
                "rank": 16,
                "alpha": 16,
                "lr": 1e-4,
                "epochs": 10,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss or diffusers"
            },
            "z-image": {
                "rank": 32,
                "alpha": 16,
                "lr": 1e-4,
                "epochs": 12,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss or ComfyUI"
            },
            "sdxl-1.0": {
                "rank": 32,
                "alpha": 16,
                "lr": 1e-4,
                "epochs": 12,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss sd-scripts"
            },
            "wan-2.1-t2v": {
                "rank": 32,
                "alpha": 32,
                "lr": 8e-5,
                "epochs": 8,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss or ComfyUI"
            },
            "ltx-video": {
                "rank": 16,
                "alpha": 16,
                "lr": 1e-4,
                "epochs": 10,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "diffusers"
            },
            "flux-1-schnell": {
                "rank": 16,
                "alpha": 16,
                "lr": 1.5e-4,
                "epochs": 8,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss or ai-toolkit"
            },
            "minimax-video": {
                "rank": 32,
                "alpha": 32,
                "lr": 8e-5,
                "epochs": 8,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "comfyui or diffusers"
            },
            "wan-2.1-turbo": {
                "rank": 32,
                "alpha": 32,
                "lr": 1e-4,
                "epochs": 6,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "kohya_ss or ComfyUI"
            },
            "ltx-video-turbo": {
                "rank": 16,
                "alpha": 16,
                "lr": 1.5e-4,
                "epochs": 8,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "diffusers or ComfyUI"
            },
            "cogvideox-5b": {
                "rank": 32,
                "alpha": 32,
                "lr": 1e-4,
                "epochs": 10,
                "optimizer": "AdamW8bit",
                "recommended_trainer": "diffusers or ComfyUI"
            }
        }

        defaults = profiles.get(base_model_id, profiles["flux-1-dev"])
        return {
            "model_info": matching,
            "recommended_parameters": defaults
        }
