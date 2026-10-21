"""Extraction tests: uniform resample kernel + per-scale pass-rate stats."""
from __future__ import annotations

import numpy as np
from PIL import Image

from lucid.core.filtering import extract_tiles_to_staging

FIXTURE_SEED = 0


def _rgb_fixture(w: int, h: int) -> np.ndarray:
    """Deterministic structured RGB image (passes all signal gates)."""
    rng = np.random.default_rng(FIXTURE_SEED)
    small = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
    return np.array(
        Image.fromarray(small).resize((w, h), Image.BICUBIC), dtype=np.uint8
    )


def test_multiscale_resample_uses_pillow_bicubic(tmp_path):
    """Downscaled tiles must come from one kernel across the pipeline.

    The >8192px downscale path (validation.py) uses Pillow BICUBIC
    (a=-0.5, the MATLAB/benchmark convention). Multiscale tiling must
    match it, not double up on a different overshoot kernel.
    """
    src = _rgb_fixture(1024, 768)
    Image.fromarray(src).save(tmp_path / "src.png")
    image_stage, num_passed, err, stats = extract_tiles_to_staging(
        tmp_path / "src.png", 256, tmp_path / "stage", multiscale=True
    )
    assert err is None

    # The scale-0.5 tiles must equal a manual Pillow BICUBIC pipeline
    expected = np.array(
        Image.fromarray(src).resize((512, 384), Image.BICUBIC), dtype=np.uint8
    )
    staged = np.array(Image.open(tmp_path / "stage" / "src" / "src_s050_t0_0.png"))
    assert np.array_equal(staged, expected[:256, :256])


def test_extract_reports_per_scale_stats(tmp_path):
    """(stage_dir, num_passed, err, scale_stats) with seen/signal-passed
    counts per scale tag, so per-scale pass rates can be reported."""
    src_path = tmp_path / "src.png"
    Image.fromarray(_rgb_fixture(1024, 768)).save(src_path)
    image_stage, num_passed, err, stats = extract_tiles_to_staging(
        src_path, 256, tmp_path / "stage", multiscale=True
    )
    assert err is None
    assert isinstance(stats, dict)
    # scale tags match the tile-name tokens _sXXX_
    assert "s100" in stats, stats
    seen, signal_passed = stats["s100"]
    assert 0 < signal_passed <= seen
    assert num_passed == sum(p for _, p in stats.values())
