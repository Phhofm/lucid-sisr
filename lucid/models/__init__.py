"""LUCID models package."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from platformdirs import user_cache_dir

from .icnet import ICNet

__all__ = ["ICNet", "ensure_model_downloaded", "get_default_model_path"]

MODEL_URL = (
    "https://github.com/umzi2/Dataset_Preprocessing/releases/download/"
    "IC9600_duplicate/ic9600.pth"
)

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    return Path(user_cache_dir("lucid", "Phips"))


def get_default_model_path() -> Path:
    """Get the default path for the ICNet model weights."""
    return get_cache_dir() / "complexity.pth"


def ensure_model_downloaded(model_path: Optional[Path] = None) -> Path:
    """Download ICNet model weights if not present.

    Args:
        model_path: Optional custom path. Uses default cache if not provided.

    Returns:
        Path to model weights file.
    """
    path = model_path or get_default_model_path()

    if path.exists():
        logger.info("Model already cached at %s", path)
        return path

    logger.info("Downloading ICNet model weights to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        state = torch.hub.load_state_dict_from_url(
            url=MODEL_URL,
            map_location="cpu",
            weights_only=True,
        )
        torch.save(state, path)
        logger.info("Model downloaded successfully")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model weights from {MODEL_URL}. "
            "Please download manually and use --icnet option."
        ) from exc

    return path
