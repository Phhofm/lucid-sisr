"""Unit tests for signal metrics."""
import numpy as np

from lucid.core.metrics import THRESHOLDS, measure_all


def test_uniform_tile_fails():
    """A completely uniform tile should fail all metrics."""
    gray = np.zeros((512, 512), dtype=np.uint8)
    metrics, passes = measure_all(gray)
    assert not passes, "Uniform tile should fail"
    assert len(metrics) == 6


def test_random_tile_produces_six_metrics():
    """A random noise tile produces exactly 6 metrics (may not pass due to noise)."""
    np.random.seed(42)
    gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    metrics, passes = measure_all(gray)
    assert len(metrics) == 6
    # Random noise typically fails noise_ratio or aliasing checks, so we
    # do not assert passes here — the key invariant is the metric count.


def test_gradient_tile_passes():
    """A textured tile (multi-frequency sine pattern) should pass all signal filters."""
    y, x = np.mgrid[0:256, 0:256]
    base = (
        np.sin(x / 5) * 30
        + np.sin(y / 7) * 30
        + np.sin((x + y) / 10) * 20
        + np.sin(x * y / 200) * 15
    )
    tile = (base + 128).clip(0, 255).astype(np.uint8)
    gray = np.tile(tile, (2, 2))  # tile to 512x512 for consistency
    metrics, passes = measure_all(gray)
    assert len(metrics) == 6
    assert passes, "Textured tile should pass all signal filters"


def test_thresholds_exist():
    """All threshold keys should be defined."""
    assert "entropy_min" in THRESHOLDS
    assert "lap_var_min" in THRESHOLDS
    assert "lap_var_max" in THRESHOLDS
    assert "blockiness_max" in THRESHOLDS
    assert "aliasing_max" in THRESHOLDS
    assert "grad_energy_min" in THRESHOLDS
    assert "noise_ratio_max" in THRESHOLDS


if __name__ == "__main__":
    test_uniform_tile_fails()
    test_random_tile_produces_six_metrics()
    test_thresholds_exist()
    print("All metric tests passed.")
