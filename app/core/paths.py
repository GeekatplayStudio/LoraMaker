"""Path validation helpers for endpoints that read, write or delete user files.

The Studio is a local desktop app, but its HTTP API is reachable by any page
open in the user's browser.  Endpoints that take a filesystem path therefore
only operate on the file types they are meant for, and mutating endpoints only
touch files that live inside a Studio project (a directory that holds
``project.json``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
}


class PathNotAllowed(ValueError):
    """Raised when a request names a file the endpoint must not touch."""


def safe_filename(name: str, default: str = "upload") -> str:
    """Strip directories and control characters from a client-supplied filename."""
    base = Path(name or "").name
    cleaned = "".join(c for c in base if c.isprintable() and c not in '<>:"/\\|?*')
    cleaned = cleaned.strip(" .")
    return cleaned or default


def project_root_of(path: Path, max_depth: int = 4) -> Optional[Path]:
    """Return the nearest ancestor that contains ``project.json``."""
    current = path if path.is_dir() else path.parent
    for _ in range(max_depth):
        if (current / "project.json").is_file():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def ensure_allowed_file(
    raw_path: str,
    suffixes: Iterable[str],
    *,
    must_exist: bool = True,
    within_project: bool = False,
) -> Path:
    """Validate a client-supplied path.

    * The suffix must be one of ``suffixes``.
    * With ``must_exist`` the path must be an existing regular file.
    * With ``within_project`` the file must sit inside a Studio project.
    """
    if not raw_path or not str(raw_path).strip():
        raise PathNotAllowed("A file path is required.")
    path = Path(str(raw_path).strip()).expanduser().resolve()
    allowed = {s.lower() for s in suffixes}
    if path.suffix.lower() not in allowed:
        raise PathNotAllowed(f"'{path.name}' is not an allowed file type ({', '.join(sorted(allowed))}).")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if within_project and project_root_of(path) is None:
        raise PathNotAllowed(f"'{path.name}' is not inside a Studio project (no project.json found).")
    return path
