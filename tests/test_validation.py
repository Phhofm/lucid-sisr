"""Unit tests for image validation."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from lucid.core.validation import is_safe_to_process, load_image_as_rgb, validate_image


def test_is_safe_to_process_valid():
    """A valid small PNG should pass safety check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        img = Image.new("RGB", (64, 64), color="red")
        img.save(img_path)
        assert is_safe_to_process(img_path)


def test_validate_image_truncated():
    """A truncated/corrupted file should fail validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "truncated.png"
        # Write incomplete PNG data
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert validate_image(img_path) is None


def test_load_image_as_rgb():
    """Valid image should load as RGB array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "rgb.png"
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)
        loaded = load_image_as_rgb(img_path)
        assert loaded is not None
        assert loaded.shape == (64, 64, 3)


if __name__ == "__main__":
    test_is_safe_to_process_valid()
    test_validate_image_truncated()
    test_load_image_as_rgb()
    print("All validation tests passed.")
