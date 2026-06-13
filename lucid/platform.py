"""Platform-specific and cross-platform helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def set_process_priority() -> None:
    """Lower process priority to keep the system responsive (best effort)."""
    try:
        os.nice(15)
    except (AttributeError, OSError) as e:
        logger.debug("Could not lower process priority: %s", e)


def safe_mkdir(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    """Remove characters illegal on Windows / most filesystems."""
    # Keep alphanumerics, underscores, hyphens, dots, and spaces.
    return "".join(c for c in name if c.isalnum() or c in " ._-").strip()


def is_safe_to_spawn_workers() -> bool:
    """Check whether we are inside a multiprocessing context already."""
    return "SPAWN" not in os.environ.get("_LUCID_WORKER", "")


def mark_worker_env() -> None:
    """Set an environment variable so child processes know they are workers."""
    os.environ["_LUCID_WORKER"] = "SPAWN"
