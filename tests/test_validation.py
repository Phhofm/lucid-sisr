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


def test_load_image_respects_exif_orientation():
    """EXIF-rotated photos must load with the rotation applied.

    Phone/Flickr photos carry orientation tags; tiling the raw pixel
    data produces sideways GT. Orientation 6 = rotate 90° CW to display,
    so the 200x120 stored frame must load as 120x200.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "rotated.jpg"
        img = Image.new("RGB", (200, 120), (10, 20, 30))
        exif = img.getexif()
        exif[0x0112] = 6
        img.save(img_path, exif=exif)
        loaded = load_image_as_rgb(img_path)
        # array shape is (h, w): the raw stored frame would give (120, 200, 3)
        assert loaded is not None
        assert loaded.shape == (200, 120, 3), loaded.shape


def test_load_image_warns_on_16bit(caplog):
    """16-bit sources must warn: PIL silently collapses them to 8-bit."""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "deep.png"
        Image.new("I;16", (64, 64)).save(img_path)
        with caplog.at_level(logging.WARNING, logger="lucid.core.validation"):
            loaded = load_image_as_rgb(img_path)
        assert loaded is not None
        assert any("16" in r.message or "I" in r.message for r in caplog.records)


if __name__ == "__main__":
    test_is_safe_to_process_valid()
    test_validate_image_truncated()
    test_load_image_as_rgb()
    print("All validation tests passed.")
