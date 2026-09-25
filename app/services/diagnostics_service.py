"""Diagnostics & Error Telemetry Service
Geekatplay Studio - Vladimir Chopine
Repository: https://github.com/GeekatplayStudio/LoraMaker.git

Captures in-memory server logs, detailed Python tracebacks, hardware telemetry,
and provides rich diagnostic reports for 1-click copying and UI display.
"""

from __future__ import annotations

import collections
import datetime
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings


class InMemoryLogHandler(logging.Handler):
    """Ring-buffer logging handler that captures real-time server messages."""

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.buffer: collections.deque = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Filter harmless client-side socket closures during HTML5 video range scrubbing
            if "10054" in msg or "_call_connection_lost" in msg:
                return
            self.buffer.append({
                "timestamp": datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            })
        except Exception:
            self.handleError(record)


class DiagnosticsService:
    """Central error & logging collector for the entire studio."""

    _log_handler: Optional[InMemoryLogHandler] = None
    _errors: collections.deque = collections.deque(maxlen=100)
    _initialized: bool = False

    @classmethod
    def setup_logging_hook(cls):
        """Install memory handler on the root and uvicorn loggers."""
        if cls._initialized:
            return
        cls._log_handler = InMemoryLogHandler(capacity=1000)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        cls._log_handler.setFormatter(formatter)

        root = logging.getLogger()
        root.addHandler(cls._log_handler)

        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
            l = logging.getLogger(name)
            l.addHandler(cls._log_handler)

        cls._initialized = True

    @classmethod
    def record_error(
        cls,
        endpoint: str,
        method: str = "POST",
        error: Any = None,
        traceback_str: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
        suggestion: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a structured exception for display in the console and diagnostic report."""
        if not traceback_str and isinstance(error, BaseException):
            traceback_str = traceback.format_exc()
        if not traceback_str:
            traceback_str = "No Python traceback available."

        err_type = type(error).__name__ if isinstance(error, BaseException) else "Exception"
        err_msg = str(error) if error is not None else "Unknown error"

        # Generate intelligent suggestion based on error pattern
        if not suggestion:
            suggestion = cls._generate_suggestion(err_type, err_msg, endpoint)

        record = {
            "id": f"err_{int(time.time() * 1000)}_{len(cls._errors) + 1}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iso_timestamp": datetime.datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "error_type": err_type,
            "message": err_msg,
            "traceback": traceback_str,
            "request_payload": payload,
            "suggestion": suggestion,
        }
        cls._errors.appendleft(record)
        return record

    @classmethod
    def _generate_suggestion(cls, err_type: str, err_msg: str, endpoint: str) -> str:
        msg_lower = err_msg.lower()
        if "project directory" in msg_lower or ("project" in msg_lower and "does not exist" in msg_lower):
            return "Project directory not found. Select an active project from the top header dropdown or click '+ New Project' to create one."
        if "uncaptioned" in msg_lower:
            return "Uncaptioned keyframes detected. Open the Keyframes Gallery tab and click '✨ Auto-Caption All Pending (Ollama)' or provide text captions for all frames."
        if "no keyframe" in msg_lower or "0 keyframe" in msg_lower:
            return "No images in Keyframes_Out. Use the Video Scrubber tab to extract frames or drop training images into the project's Keyframes_Out folder."
        if "network_module" in msg_lower:
            return "LoRA trainer network module definition was missing. Fixed in latest training service update."
        if "checkpoint" in msg_lower or "not found" in msg_lower:
            return "Base model checkpoint could not be found. Go to Settings tab to verify or add model folders."
        if "cuda" in msg_lower or "out of memory" in msg_lower:
            return "GPU VRAM exhausted. Lower batch size, resolution, or rank in the training configuration."
        if "keyframe" in msg_lower:
            return "At least 2 valid, captioned keyframes in Keyframes_Out are required to train a real LoRA."
        if "connection" in msg_lower or "failed to fetch" in msg_lower:
            return "Backend connection lost. Ensure the Python Uvicorn server is running on http://127.0.0.1:7860."
        return "Review the Python traceback above or click 'Copy Diagnostic Report' to share with the assistant."

    @classmethod
    def get_recent_errors(cls, limit: int = 50) -> List[Dict[str, Any]]:
        return list(cls._errors)[:limit]

    @classmethod
    def get_recent_logs(cls, limit: int = 150) -> List[Dict[str, Any]]:
        if not cls._log_handler:
            return []
        return list(cls._log_handler.buffer)[-limit:]

    @classmethod
    def clear_errors(cls):
        cls._errors.clear()

    @classmethod
    def get_diagnostics_report(cls) -> Dict[str, Any]:
        """Compile a full diagnostic snapshot of system health, hardware, and recent errors."""
        from app.services.hardware_service import HardwareService
        from app.services.settings_service import SettingsService

        hw = HardwareService.get_hardware_profile()
        cfg = SettingsService.get_user_settings()
        roots = [str(p) for p in SettingsService.get_all_model_directories()]

        # Package versions
        packages = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "os": platform.system(),
        }
        try:
            import torch
            packages["torch"] = torch.__version__
            packages["torch_cuda"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                packages["torch_device"] = torch.cuda.get_device_name(0)
        except Exception:
            packages["torch"] = "Not available"

        try:
            import diffusers
            packages["diffusers"] = diffusers.__version__
        except Exception:
            packages["diffusers"] = "Not available"

        try:
            import transformers
            packages["transformers"] = transformers.__version__
        except Exception:
            packages["transformers"] = "Not available"

        try:
            import safetensors
            packages["safetensors"] = safetensors.__version__
        except Exception:
            packages["safetensors"] = "Not available"

        return {
            "app_name": settings.APP_NAME,
            "version": settings.VERSION,
            "author": settings.AUTHOR,
            "studio": settings.STUDIO,
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware": {
                "device_name": hw.get("device_name", "Unknown"),
                "cuda_available": hw.get("cuda_available", False),
                "total_vram_gb": hw.get("total_vram_gb", 0.0),
                "tier_name": hw.get("tier_name", "Standard"),
            },
            "environment": packages,
            "storage": {
                "models_dir": cfg.get("MODELS_DIR"),
                "download_dir": cfg.get("DOWNLOAD_DIR"),
                "extra_model_paths": cfg.get("EXTRA_MODEL_PATHS", []),
                "all_searched_roots": roots,
                "drives": cfg.get("drives", []),
            },
            "recent_errors": list(cls._errors)[:20],
            "recent_logs": list(cls._log_handler.buffer)[-50:] if cls._log_handler else [],
            "error_count": len(cls._errors),
        }
