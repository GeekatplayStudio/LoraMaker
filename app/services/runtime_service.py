"""Portable, read-only discovery of optional local training runtimes.

This module deliberately does not create directories, download models, or
claim that a runtime is usable merely because a path was configured.  Callers
can use the returned profile to decide which integrations may be offered.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from app.core.config import AppSettings, settings


class RuntimeService:
    """Inspect configured ComfyUI and Kohya roots without platform assumptions."""

    @staticmethod
    def _path_state(path: Path, *, children: Iterable[str] = ()) -> dict[str, Any]:
        """Return only read-only facts about a configured path."""
        resolved = path.expanduser().resolve(strict=False)
        state: dict[str, Any] = {
            "configured_path": str(resolved),
            "exists": resolved.exists(),
            "is_directory": resolved.is_dir(),
        }
        for child in children:
            child_path = resolved / child
            state[f"{child}_exists"] = child_path.exists()
        return state

    @staticmethod
    def _kohya_script_root(root: Path) -> Path:
        """Accept either a kohya checkout or a direct sd-scripts path."""
        root = root.expanduser().resolve(strict=False)
        return root if root.name.lower() == "sd-scripts" else root / "sd-scripts"

    @classmethod
    def discover(cls, config: Optional[AppSettings] = None) -> dict[str, Any]:
        """Discover configured installations and known training entry points.

        A discovered script means only that the local file exists; it is not a
        training capability assertion.  Hardware and model validation remain
        the responsibility of the actual training backend.
        """
        config = config or settings
        comfy_root = Path(config.COMFYUI_ROOT)
        primary_kohya_root = cls._kohya_script_root(Path(config.KOHYA_ROOT))
        fallback_kohya_root = cls._kohya_script_root(Path(config.KOHYA_FALLBACK_ROOT))

        kohya_roots = []
        seen: set[Path] = set()
        for root in (primary_kohya_root, fallback_kohya_root):
            resolved = root.expanduser().resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            state = cls._path_state(
                resolved,
                children=("sdxl_train_network.py", "flux_train_network.py"),
            )
            state["sd_scripts_root"] = state.pop("configured_path")
            kohya_roots.append(state)

        comfy = cls._path_state(comfy_root, children=("models",))
        resolved_comfy = comfy_root.expanduser().resolve(strict=False)
        models = resolved_comfy / "models"
        comfy.update({
            "checkpoints_path": str(models / "checkpoints"),
            "checkpoints_exists": (models / "checkpoints").is_dir(),
            "loras_path": str(models / "loras"),
            "loras_exists": (models / "loras").is_dir(),
        })
        return {"comfyui": comfy, "kohya": kohya_roots}

    @staticmethod
    def accelerator_profile() -> dict[str, Any]:
        """Safely report the active PyTorch backend, including CPU-only hosts."""
        profile: dict[str, Any] = {
            "platform": platform.system(),
            "python": platform.python_version(),
            "backend": "cpu",
            "torch_available": False,
            "cuda_available": False,
            "mps_available": False,
        }
        try:
            import torch

            profile["torch_available"] = True
            profile["torch_version"] = torch.__version__
            profile["cuda_available"] = bool(torch.cuda.is_available())
            mps = getattr(torch.backends, "mps", None)
            profile["mps_available"] = bool(mps and mps.is_available())
            if profile["cuda_available"]:
                profile["backend"] = "cuda"
            elif profile["mps_available"]:
                profile["backend"] = "mps"
        except Exception as exc:  # optional dependency or unavailable driver
            profile["torch_error"] = type(exc).__name__
        return profile

    @classmethod
    def runtime_profile(cls, config: Optional[AppSettings] = None) -> dict[str, Any]:
        """Return a serializable, read-only runtime profile for diagnostics."""
        return {
            "runtime": {
                "platform": platform.platform(),
                "python_executable": sys.executable,
            },
            "accelerator": cls.accelerator_profile(),
            "installations": cls.discover(config),
        }
