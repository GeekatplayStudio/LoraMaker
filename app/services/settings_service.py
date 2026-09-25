"""User Configuration & Storage Management Service
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git

Allows the user to specify custom base model directories, dedicated download folders,
ComfyUI installations, and Kohya paths. Automatically sets HF_HOME and TORCH_HOME
to user-specified folders to prevent C: drive exhaustion and duplicate downloads.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class SettingsService:
    SETTINGS_FILE = settings.BASE_DIR / "user_settings.json"

    @classmethod
    def get_user_settings(cls) -> Dict[str, Any]:
        """Loads persisted user settings or returns intelligent defaults."""
        data = {}
        if cls.SETTINGS_FILE.exists():
            try:
                data = json.loads(cls.SETTINGS_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read user_settings.json: {e}")

        comfy_root = Path(data.get("COMFYUI_ROOT", str(settings.COMFYUI_ROOT)))
        models_dir = Path(data.get("MODELS_DIR", str(comfy_root / "models" if comfy_root.exists() else Path("D:/ComfyUI/ComfyUI/models"))))
        download_dir = Path(data.get("DOWNLOAD_DIR", str(models_dir)))
        kohya_root = Path(data.get("KOHYA_ROOT", str(settings.KOHYA_ROOT)))
        kohya_fallback = Path(data.get("KOHYA_FALLBACK_ROOT", str(settings.KOHYA_FALLBACK_ROOT)))
        ollama_host = data.get("OLLAMA_HOST", settings.OLLAMA_HOST)

        # Drive free space monitoring
        drives_info = cls._get_drives_space([models_dir, download_dir, Path("C:/"), Path("D:/")])

        return {
            "success": True,
            "COMFYUI_ROOT": str(comfy_root.resolve()) if comfy_root.exists() else str(comfy_root),
            "MODELS_DIR": str(models_dir.resolve()) if models_dir.exists() else str(models_dir),
            "DOWNLOAD_DIR": str(download_dir.resolve()) if download_dir.exists() else str(download_dir),
            "KOHYA_ROOT": str(kohya_root.resolve()) if kohya_root.exists() else str(kohya_root),
            "KOHYA_FALLBACK_ROOT": str(kohya_fallback.resolve()) if kohya_fallback.exists() else str(kohya_fallback),
            "OLLAMA_HOST": ollama_host,
            "paths_status": {
                "comfyui_exists": comfy_root.is_dir(),
                "models_dir_exists": models_dir.is_dir(),
                "download_dir_exists": download_dir.is_dir(),
                "kohya_root_exists": kohya_root.is_dir(),
                "kohya_fallback_exists": kohya_fallback.is_dir(),
            },
            "drives": drives_info,
            "is_customized": cls.SETTINGS_FILE.exists()
        }

    @classmethod
    def save_user_settings(cls, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Saves user settings to user_settings.json and updates runtime environment variables."""
        current = cls.get_user_settings()
        updated = {
            "COMFYUI_ROOT": str(Path(new_settings.get("COMFYUI_ROOT", current["COMFYUI_ROOT"])).resolve()),
            "MODELS_DIR": str(Path(new_settings.get("MODELS_DIR", current["MODELS_DIR"])).resolve()),
            "DOWNLOAD_DIR": str(Path(new_settings.get("DOWNLOAD_DIR", current["DOWNLOAD_DIR"])).resolve()),
            "KOHYA_ROOT": str(Path(new_settings.get("KOHYA_ROOT", current["KOHYA_ROOT"])).resolve()),
            "KOHYA_FALLBACK_ROOT": str(Path(new_settings.get("KOHYA_FALLBACK_ROOT", current["KOHYA_FALLBACK_ROOT"])).resolve()),
            "OLLAMA_HOST": str(new_settings.get("OLLAMA_HOST", current["OLLAMA_HOST"])).strip(),
        }

        # Create download dir if missing
        dl_path = Path(updated["DOWNLOAD_DIR"])
        try:
            dl_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create download dir {dl_path}: {e}")

        # Protect C: drive by directing HuggingFace and Torch caches to user-specified download dir
        hf_cache = dl_path / "huggingface" / "hub"
        hf_cache.mkdir(parents=True, exist_ok=True)
        torch_cache = dl_path / "torch"
        torch_cache.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(dl_path / "huggingface")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
        os.environ["TORCH_HOME"] = str(torch_cache)

        # Update in-memory AppSettings
        settings.COMFYUI_ROOT = Path(updated["COMFYUI_ROOT"])
        settings.KOHYA_ROOT = Path(updated["KOHYA_ROOT"])
        settings.KOHYA_FALLBACK_ROOT = Path(updated["KOHYA_FALLBACK_ROOT"])
        settings.OLLAMA_HOST = updated["OLLAMA_HOST"]

        # Persist to JSON
        cls.SETTINGS_FILE.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        logger.info(f"User settings persisted to {cls.SETTINGS_FILE}. Download cache set to {dl_path}.")

        return cls.get_user_settings()

    # Convenience alias
    save_settings = save_user_settings

    @classmethod
    def scan_models_directory(cls, custom_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Scans the user-configured models folder and returns an inventory
        of checkpoints, unets, clip models, VAEs, diffusion models, and LoRAs.
        """
        if custom_dir:
            base_p = Path(custom_dir)
        else:
            current = cls.get_user_settings()
            base_p = Path(current["MODELS_DIR"])

        if not base_p.exists():
            return {
                "success": False,
                "error": f"Models directory '{base_p}' does not exist.",
                "total_models": 0,
                "categories": {}
            }

        categories = {
            "checkpoints": [],
            "unet": [],
            "diffusion_models": [],
            "clip": [],
            "vae": [],
            "loras": []
        }

        total_bytes = 0
        total_count = 0

        # Scan subdirectories if standard layout, or flat scan
        for cat in categories.keys():
            cat_dir = base_p / cat
            if cat_dir.is_dir():
                for f in sorted(cat_dir.glob("*")):
                    if f.is_file() and f.suffix.lower() in [".safetensors", ".ckpt", ".pt", ".bin"]:
                        sz = f.stat().st_size
                        total_bytes += sz
                        total_count += 1
                        categories[cat].append({
                            "name": f.name,
                            "path": str(f.resolve()),
                            "size_mb": round(sz / (1024 * 1024), 1),
                            "size_gb": round(sz / (1024 ** 3), 2)
                        })

        # Also search for standalone safetensors in root if flat folder
        for f in sorted(base_p.glob("*.safetensors")):
            if f.is_file():
                sz = f.stat().st_size
                total_bytes += sz
                total_count += 1
                categories["checkpoints"].append({
                    "name": f.name,
                    "path": str(f.resolve()),
                    "size_mb": round(sz / (1024 * 1024), 1),
                    "size_gb": round(sz / (1024 ** 3), 2)
                })

        return {
            "success": True,
            "models_dir": str(base_p.resolve()),
            "total_models": total_count,
            "total_size_gb": round(total_bytes / (1024 ** 3), 2),
            "categories": {
                k: {
                    "count": len(v),
                    "models": v
                } for k, v in categories.items()
            }
        }

    @staticmethod
    def _get_drives_space(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
        """Returns drive capacity and free space in GB for all distinct drives referenced."""
        drives = {}
        seen_roots = set()
        for p in paths:
            try:
                resolved = p.resolve()
                root = resolved.anchor
                if not root or root in seen_roots:
                    continue
                seen_roots.add(root)
                total, used, free = shutil.disk_usage(root)
                drives[root] = {
                    "root": root,
                    "total_gb": round(total / (1024 ** 3), 1),
                    "used_gb": round(used / (1024 ** 3), 1),
                    "free_gb": round(free / (1024 ** 3), 1),
                    "percent_free": round((free / max(1, total)) * 100, 1),
                    "is_low_space": (free / max(1, total)) < 0.10
                }
            except Exception:
                continue
        return drives
