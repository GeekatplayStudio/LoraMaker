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

        # Multi-location model directories support
        raw_extra = data.get("EXTRA_MODEL_PATHS", [])
        extra_paths = []
        if isinstance(raw_extra, list):
            for ep in raw_extra:
                if ep and isinstance(ep, str) and ep.strip():
                    p_obj = Path(ep.strip())
                    p_str = str(p_obj.resolve()) if p_obj.exists() else str(p_obj)
                    p_primary = str(models_dir.resolve()) if models_dir.exists() else str(models_dir)
                    if p_str not in extra_paths and p_str.lower() != p_primary.lower():
                        extra_paths.append(p_str)

        all_model_paths = [str(models_dir.resolve()) if models_dir.exists() else str(models_dir)]
        for ep in extra_paths:
            if ep not in all_model_paths:
                all_model_paths.append(ep)

        # Drive free space monitoring across ALL configured model directories
        probe_paths = [models_dir, download_dir, Path("C:/"), Path("D:/")]
        for ep in extra_paths:
            probe_paths.append(Path(ep))
        drives_info = cls._get_drives_space(probe_paths)

        return {
            "success": True,
            "COMFYUI_ROOT": str(comfy_root.resolve()) if comfy_root.exists() else str(comfy_root),
            "MODELS_DIR": str(models_dir.resolve()) if models_dir.exists() else str(models_dir),
            "EXTRA_MODEL_PATHS": extra_paths,
            "ALL_MODEL_PATHS": all_model_paths,
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
    def get_all_model_directories(cls) -> List[Path]:
        """Returns all configured model directory Paths that exist on disk."""
        current = cls.get_user_settings()
        roots = []
        seen = set()
        candidates = current.get("ALL_MODEL_PATHS", [current["MODELS_DIR"]]) + [current["DOWNLOAD_DIR"]]
        for c in candidates:
            if not c:
                continue
            p = Path(c)
            if p.exists() and p.is_dir():
                resolved = p.resolve()
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    roots.append(resolved)
        return roots

    @classmethod
    def add_model_path(cls, path_to_add: str) -> Dict[str, Any]:
        """Adds a new model folder location and persists."""
        p_clean = Path(path_to_add.strip())
        resolved_str = str(p_clean.resolve()) if p_clean.exists() else str(p_clean)
        current = cls.get_user_settings()
        extras = list(current.get("EXTRA_MODEL_PATHS", []))
        if resolved_str.lower() != current["MODELS_DIR"].lower() and resolved_str not in extras:
            extras.append(resolved_str)
        return cls.save_user_settings({"EXTRA_MODEL_PATHS": extras})

    @classmethod
    def remove_model_path(cls, path_to_remove: str) -> Dict[str, Any]:
        """Removes a model folder location from EXTRA_MODEL_PATHS and persists."""
        p_clean = Path(path_to_remove.strip())
        target = str(p_clean.resolve()) if p_clean.exists() else str(p_clean)
        current = cls.get_user_settings()
        extras = [ep for ep in current.get("EXTRA_MODEL_PATHS", []) if ep.lower() != target.lower() and ep.lower() != str(p_clean).lower()]
        return cls.save_user_settings({"EXTRA_MODEL_PATHS": extras})

    @classmethod
    def save_user_settings(cls, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Saves user settings to user_settings.json and updates runtime environment variables."""
        current = cls.get_user_settings()
        
        extra_paths = new_settings.get("EXTRA_MODEL_PATHS", current.get("EXTRA_MODEL_PATHS", []))
        if not isinstance(extra_paths, list):
            extra_paths = []
        clean_extra = []
        for ep in extra_paths:
            if ep and isinstance(ep, str) and ep.strip():
                p_c = Path(ep.strip())
                val = str(p_c.resolve()) if p_c.exists() else str(p_c)
                if val not in clean_extra:
                    clean_extra.append(val)

        updated = {
            "COMFYUI_ROOT": str(Path(new_settings.get("COMFYUI_ROOT", current["COMFYUI_ROOT"])).resolve()),
            "MODELS_DIR": str(Path(new_settings.get("MODELS_DIR", current["MODELS_DIR"])).resolve()),
            "EXTRA_MODEL_PATHS": clean_extra,
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
        Scans model folders across all configured locations or a specific folder,
        returning an inventory of checkpoints, unets, clip models, VAEs, diffusion models, and LoRAs.
        """
        if custom_dir:
            scan_dirs = [Path(custom_dir)]
        else:
            scan_dirs = cls.get_all_model_directories()
            if not scan_dirs:
                current = cls.get_user_settings()
                scan_dirs = [Path(current["MODELS_DIR"])]

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
        seen_paths = set()
        locations_scanned = []

        for base_p in scan_dirs:
            if not base_p.exists():
                locations_scanned.append({
                    "path": str(base_p),
                    "exists": False,
                    "models_count": 0,
                    "size_gb": 0.0
                })
                continue

            loc_bytes = 0
            loc_count = 0

            # Scan standard subdirectories
            for cat in categories.keys():
                cat_dir = base_p / cat
                if cat_dir.is_dir():
                    for f in sorted(cat_dir.glob("*")):
                        resolved_f = str(f.resolve())
                        if f.is_file() and f.suffix.lower() in [".safetensors", ".ckpt", ".pt", ".bin"] and resolved_f not in seen_paths:
                            seen_paths.add(resolved_f)
                            sz = f.stat().st_size
                            total_bytes += sz
                            loc_bytes += sz
                            total_count += 1
                            loc_count += 1
                            categories[cat].append({
                                "name": f.name,
                                "path": resolved_f,
                                "size_mb": round(sz / (1024 * 1024), 1),
                                "size_gb": round(sz / (1024 ** 3), 2),
                                "location": str(base_p.resolve())
                            })

            # Also search for standalone safetensors in root if flat folder
            for f in sorted(base_p.glob("*.safetensors")):
                resolved_f = str(f.resolve())
                if f.is_file() and resolved_f not in seen_paths:
                    seen_paths.add(resolved_f)
                    sz = f.stat().st_size
                    total_bytes += sz
                    loc_bytes += sz
                    total_count += 1
                    loc_count += 1
                    categories["checkpoints"].append({
                        "name": f.name,
                        "path": resolved_f,
                        "size_mb": round(sz / (1024 * 1024), 1),
                        "size_gb": round(sz / (1024 ** 3), 2),
                        "location": str(base_p.resolve())
                    })

            locations_scanned.append({
                "path": str(base_p.resolve()),
                "exists": True,
                "models_count": loc_count,
                "size_gb": round(loc_bytes / (1024 ** 3), 2)
            })

        return {
            "success": True,
            "models_dir": str(scan_dirs[0].resolve()) if scan_dirs else "",
            "total_models": total_count,
            "total_size_gb": round(total_bytes / (1024 ** 3), 2),
            "locations_scanned": locations_scanned,
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

    @classmethod
    def browse_filesystem(cls, target_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Interactive filesystem browser for Windows and POSIX systems.
        If target_path is None or empty or 'ROOT' or 'drives', returns available storage drives.
        Otherwise, lists subdirectories, parent directory, and breadcrumbs with model hints.
        """
        import string
        clean_target = (target_path or "").strip()

        # Check if root drives requested or empty
        if not clean_target or clean_target.upper() in ("ROOT", "DRIVES", "/", "\\"):
            drives = []
            for letter in string.ascii_uppercase:
                d = f"{letter}:\\"
                if os.path.exists(d):
                    try:
                        u = shutil.disk_usage(d)
                        drives.append({
                            "name": f"{letter}: Drive",
                            "path": d,
                            "is_drive": True,
                            "free_gb": round(u.free / (1024 ** 3), 1),
                            "total_gb": round(u.total / (1024 ** 3), 1),
                            "percent_free": round((u.free / max(1, u.total)) * 100, 1)
                        })
                    except Exception:
                        drives.append({
                            "name": f"{letter}: Drive",
                            "path": d,
                            "is_drive": True,
                            "free_gb": 0.0,
                            "total_gb": 0.0,
                            "percent_free": 0.0
                        })
            return {
                "success": True,
                "is_root": True,
                "current_path": "",
                "parent_path": None,
                "breadcrumbs": [],
                "items": drives,
                "total_items": len(drives)
            }

        p = Path(clean_target).resolve()
        if not p.exists() or not p.is_dir():
            # If path doesn't exist, try parent or fallback to root drives
            if p.parent and p.parent.exists() and p.parent.is_dir():
                p = p.parent
            else:
                return cls.browse_filesystem(None)

        # Build breadcrumbs
        parts = p.parts
        breadcrumbs = []
        if parts:
            curr = Path(parts[0])
            breadcrumbs.append({"name": parts[0], "path": str(curr)})
            for part in parts[1:]:
                curr = curr / part
                breadcrumbs.append({"name": part, "path": str(curr)})

        # Parent path: if at drive root (p.parent == p), parent is root drives ("")
        parent_path = str(p.parent) if p.parent != p else ""

        # Scan subdirectories
        items = []
        known_model_dirs = {"checkpoints", "unet", "loras", "vae", "clip", "diffusion_models", "models"}

        try:
            with os.scandir(str(p)) as it:
                for entry in it:
                    try:
                        # Skip system/hidden folders
                        if entry.name.startswith(("$", ".")) or entry.name in ("System Volume Information", "$RECYCLE.BIN"):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            is_model_dir = entry.name.lower() in known_model_dirs
                            items.append({
                                "name": entry.name,
                                "path": entry.path,
                                "is_dir": True,
                                "is_drive": False,
                                "is_model_dir": is_model_dir
                            })
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError) as e:
            return {
                "success": False,
                "error": f"Access denied or error reading directory: {e}",
                "is_root": False,
                "current_path": str(p),
                "parent_path": parent_path,
                "breadcrumbs": breadcrumbs,
                "items": []
            }

        items.sort(key=lambda x: x["name"].lower())

        return {
            "success": True,
            "is_root": False,
            "current_path": str(p),
            "parent_path": parent_path,
            "breadcrumbs": breadcrumbs,
            "items": items,
            "total_items": len(items)
        }
