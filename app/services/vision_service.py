import os
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import ollama
from app.core.config import settings

logger = logging.getLogger(__name__)

class VisionService:
    """
    Service for discovering local Ollama vision models and generating 
    accurate LoRA training captions and text files.
    """

    @classmethod
    def list_installed_models(cls) -> List[Dict[str, Any]]:
        """
        Queries local Ollama instance in real time to get all installed models.
        """
        try:
            client = ollama.Client(host=settings.OLLAMA_HOST)
            res = client.list()
            model_list = []
            # res can have models attribute
            raw_models = getattr(res, "models", []) or res.get("models", [])
            for m in raw_models:
                name = getattr(m, "model", None) or m.get("model") or getattr(m, "name", None) or m.get("name")
                size = getattr(m, "size", 0) or m.get("size", 0)
                modified_at = getattr(m, "modified_at", "") or m.get("modified_at", "")
                details = getattr(m, "details", {}) or m.get("details", {})
                
                # Check if it has vision capabilities
                is_vision = cls._is_vision_model(name)
                
                model_list.append({
                    "name": name,
                    "size_gb": round(size / (1024 ** 3), 2) if size else 0,
                    "is_vision": is_vision,
                    "modified_at": str(modified_at)
                })
            return model_list
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {settings.OLLAMA_HOST}: {e}")
            return []

    @classmethod
    def _is_vision_model(cls, model_name: str) -> bool:
        """Determines if a model has multimodal/vision capabilities."""
        if not model_name:
            return False
        name_lower = model_name.lower()
        vision_keywords = ["vl", "vision", "llava", "moondream", "minicpm-v", "bakllava", "cogvlm"]
        return any(k in name_lower for k in vision_keywords)

    @classmethod
    def get_best_vision_model(cls) -> str:
        """
        Finds the newest and highest quality vision model installed in Ollama.
        Prioritizes: qwen2.5vl:7b > qwen3-vl:4b > llama3.2-vision > llava > others.
        """
        installed = cls.list_installed_models()
        installed_names = [m["name"] for m in installed]

        # 1. Match against configured preferred order
        for pref in settings.PREFERRED_VISION_MODELS:
            # Check exact match or prefix match (e.g. qwen2.5vl matches qwen2.5vl:7b)
            for inst in installed_names:
                if inst == pref or inst.startswith(pref.split(":")[0]):
                    return inst

        # 2. Check any other installed vision model
        for m in installed:
            if m["is_vision"]:
                return m["name"]

        # 3. Fallback default
        return "qwen2.5vl:7b"

    @classmethod
    def generate_caption(
        cls,
        image_path: str,
        trigger_token: str = "BendyBot",
        style_token: str = "vintage 1930s rubber hose cel animation",
        model_name: Optional[str] = None,
        timeout_seconds: int = 45
    ) -> Dict[str, Any]:
        """
        Generates a tailored LoRA training caption for an extracted cel frame scan
        and automatically writes the .txt file adjacent to the image.
        """
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image frame not found: {image_path}")

        # Choose model
        active_model = model_name or cls.get_best_vision_model()

        # Prompt crafted for LoRA dataset training (Danbooru/WD + natural language hybrid)
        prompt = (
            f"You are a professional AI dataset curator specializing in LoRA training for image and video diffusion models. "
            f"Analyze this image cel scan in detail and generate a rich, accurate caption for LoRA training.\n\n"
            f"Rules for the caption:\n"
            f"1. Start with the main character trigger token: '{trigger_token}'\n"
            f"2. Mention the artistic style token: '{style_token}'\n"
            f"3. Describe the character's pose (e.g. A-pose, standing, 3/4 turn, running, gesturing), facial expression, eyes, mouth.\n"
            f"4. Describe clothing, accessories, ink outline thickness, vintage film grain, cel shading, and background elements.\n"
            f"5. Output ONLY the plain caption text without introductory phrases like 'Here is the caption' or quotes."
        )

        try:
            client = ollama.Client(host=settings.OLLAMA_HOST, timeout=timeout_seconds)
            
            with open(img_p, "rb") as f:
                img_bytes = f.read()

            response = client.generate(
                model=active_model,
                prompt=prompt,
                images=[img_bytes],
                stream=False
            )

            raw_caption = response.get("response", "").strip()
            # Clean quotes if any
            clean_caption = raw_caption.strip('"\' \n')
            
            # Ensure trigger token is present at the beginning
            if trigger_token and not clean_caption.lower().startswith(trigger_token.lower()):
                clean_caption = f"{trigger_token}, {clean_caption}"

            # Save caption to .txt file in same directory
            txt_path = cls.save_caption_file(str(img_p), clean_caption)

            return {
                "success": True,
                "model_used": active_model,
                "caption": clean_caption,
                "caption_file": str(txt_path),
                "image_file": str(img_p)
            }
        except Exception as e:
            logger.error(f"Error calling Ollama vision model {active_model}: {e}")
            # Fallback high quality heuristic caption so workflow never breaks
            fallback_caption = (
                f"{trigger_token}, {style_token}, vintage animated character cel frame, "
                f"crisp monochrome ink linework, rubber hose animation style, "
                f"retro cartoon character pose, high contrast keyframe scan"
            )
            txt_path = cls.save_caption_file(str(img_p), fallback_caption)
            return {
                "success": False,
                "warning": f"Ollama generation fallback ({str(e)})",
                "model_used": active_model,
                "caption": fallback_caption,
                "caption_file": str(txt_path),
                "image_file": str(img_p)
            }

    @staticmethod
    def save_caption_file(image_path: str, caption_text: str) -> Path:
        """
        Saves caption_text to a text file with the same stem as image_path in the same directory.
        e.g., `image_0001.png` -> `image_0001.txt`
        """
        img_p = Path(image_path)
        txt_p = img_p.with_suffix(".txt")
        txt_p.write_text(caption_text.strip(), encoding="utf-8")
        return txt_p

    @staticmethod
    def read_caption_file(image_path: str) -> Optional[str]:
        """Reads caption file corresponding to an image."""
        img_p = Path(image_path)
        txt_p = img_p.with_suffix(".txt")
        if txt_p.exists():
            return txt_p.read_text(encoding="utf-8").strip()
        return None
