"""Buffered, throttled, resumable disk I/O helpers."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BUF_SIZE = 64


class BufferedTileWriter:
    """Batch tile images to disk with optional I/O throttling and a flush counter.

    This amortizes the cost of thousands of small writes and lets the OS
    scheduler breathe on slow (USB) drives.
    """

    def __init__(
        self,
        out_dir: Path,
        throttle_sec: float = 0.0,
        buffer_size: int = DEFAULT_BUF_SIZE,
    ):
        self.out_dir = out_dir
        self.throttle_sec = throttle_sec
        self.buffer_size = buffer_size
        self._buffer: list[tuple[str, np.ndarray]] = []
        self._written = 0
        self._flushes = 0

    def write(self, name: str, rgb_tile: np.ndarray) -> None:
        self._buffer.append((name, rgb_tile))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        for name, rgb_tile in self._buffer:
            path = self.out_dir / name
            bgr = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            self._written += 1
            if self.throttle_sec > 0:
                time.sleep(self.throttle_sec)
        self._buffer.clear()
        self._flushes += 1

    @property
    def written(self) -> int:
        return self._written

    def __enter__(self) -> BufferedTileWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.flush()


def save_tile_atomic(path: Path, rgb_tile: np.ndarray) -> None:
    """Write a single RGB tile to disk, converting to BGR for OpenCV."""
    bgr = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def load_rgb_array(path: Path) -> np.ndarray | None:
    """Load an image as an RGB numpy array, or None on failure."""
    try:
        # Prefer PIL because it handles more formats and corrupted files safely.
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception:
        # Fallback to OpenCV
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_grayscale(path: Path) -> np.ndarray | None:
    """Load an image as an 8-bit grayscale numpy array, or None on failure."""
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            return np.array(im)
    except Exception:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return arr
