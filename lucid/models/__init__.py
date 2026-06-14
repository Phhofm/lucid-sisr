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

# Bundled model path (shipped with the package)
_BUNDLED_MODEL = Path(__file__).parent.parent / "resources" / "icnet.pth"


def get_cache_dir() -> Path:
    return Path(user_cache_dir("lucid", "Phips"))


def get_default_model_path() -> Path:
    """Get the default path for the ICNet model weights."""
    return get_cache_dir() / "complexity.pth"


def ensure_model_downloaded(model_path: Optional[Path] = None) -> Path:
    """Ensure ICNet model weights are available, with bundled fallback.

    Checks in order:
    1. Custom path (if provided via --icnet)
    2. Bundled model in package resources
    3. Cache directory (previously downloaded)
    4. Download from GitHub releases

    Args:
        model_path: Optional custom path. Uses bundled/cache if not provided.

    Returns:
        Path to model weights file.
    """
    # 1. Custom path
    if model_path is not None and model_path.exists():
        logger.info("Using custom model path: %s", model_path)
        return model_path

    # 2. Bundled model
    if _BUNDLED_MODEL.exists():
        logger.info("Using bundled ICNet model: %s", _BUNDLED_MODEL)
        return _BUNDLED_MODEL

    # 3. Cache directory
    cache_path = get_default_model_path()
    if cache_path.exists():
        logger.info("Using cached model: %s", cache_path)
        return cache_path

    # 4. Download from GitHub
    logger.info("Downloading ICNet model weights to %s", cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        state = torch.hub.load_state_dict_from_url(
            url=MODEL_URL,
            map_location="cpu",
            weights_only=True,
        )
        torch.save(state, cache_path)
        logger.info("Model downloaded successfully")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model weights from {MODEL_URL}. "
            "Please download manually and use --icnet option."
        ) from exc

    return cache_path
