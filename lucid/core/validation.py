"""Corrupt/fake image pre-flight validation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

logger = logging.getLogger(__name__)

# Restrict decompression bomb size to ~200 MP (safe default)
Image.MAX_IMAGE_PIXELS = 200_000_000
# Reject truncated images (common for corrupted JPEGs)
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Maximum dimension for a single image before we downscale
MAX_IMAGE_DIM = 8192


def probe_image(path: Path) -> tuple[int, int, str] | None:
    """Quickly probe image dimensions and mode without full decode.

    Returns:
        (width, height, mode) or None if the file is not a valid image.
    """
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1], im.mode
    except Exception as exc:
        logger.debug("Image probe failed for %s: %s", path, exc)
        return None


def validate_image(path: Path, strict: bool = False) -> tuple[Image.Image, tuple[int, int]] | None:
    """Validate and load an image, rejecting corrupted/truncated files.

    Args:
        path: Path to the image file.
        strict: If True, raises on error; if False, returns None.

    Returns:
        PIL Image in RGB mode and its (width, height) tuple, or None.
    """
    try:
        im = Image.open(path)
        im.load()  # Force eager decode to catch truncation errors
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im, im.size
    except Exception as exc:
        if strict:
            raise
        logger.warning("Skipping invalid image %s: %s", path, exc)
        return None


def is_safe_to_process(path: Path, max_dim: int = MAX_IMAGE_DIM, strict: bool = False) -> bool:
    """Check if an image can be safely processed (dimensions, file integrity)."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w > max_dim or h > max_dim:
                if strict:
                    raise ValueError(f"Image {path.name} exceeds {max_dim}px limit")
                return False
        validate_image(path, strict=False)
        return True
    except Exception as exc:
        if strict:
            raise
        logger.debug("Image %s failed safety check: %s", path, exc)
        return False


def load_image_as_rgb(path: Path, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray | None:
    """Load an image as RGB numpy array, handling large images gracefully.

    Returns:
        RGB array or None on failure.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w > max_dim or h > max_dim:
                # Downscale to max_dim while preserving aspect
                # Use BICUBIC to prevent halo/ringing artifacts associated with Lanczos
                scale = max_dim / max(w, h)
                im = im.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            if im.mode != "RGB":
                im = im.convert("RGB")
            return np.array(im)
    except Exception as exc:
        logger.warning("Could not load image %s: %s", path, exc)
        return None
