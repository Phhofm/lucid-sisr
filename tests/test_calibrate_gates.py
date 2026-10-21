"""Tests for data-driven gate calibration."""
from __future__ import annotations

import cv2
import numpy as np

from lucid.calibrate_gates import _sample_crops, calibrate_gates, suggest_gate_thresholds


def test_suggest_thresholds_key_names_and_directions():
    values = {
        "entropy": [5.0 + i * 0.05 for i in range(100)],  # 5.0..9.95
        "grad_energy": [0.1 + i * 0.1 for i in range(100)],  # 0.1..10.0
        "noise_ratio": [1.0 + i * 0.01 for i in range(100)],  # 1.0..2.0
    }
    out = suggest_gate_thresholds(values, percentile=97.0)
    # min-gates take the low tail (p3 for percentile=97)
    assert "entropy_min" in out and "grad_energy_min" in out
    assert out["entropy_min"] == 5.1  # round(5.15 / 0.1) * 0.1
    # max-gates take the high tail (p97)
    assert out["noise_ratio_max"] == 1.95  # round(1.97 / 0.05 = 39.4) = 39 -> 1.95


def test_suggest_thresholds_skips_missing_metrics():
    out = suggest_gate_thresholds({"aliasing": [0.1, 0.2, 0.3]}, percentile=99.0)
    assert out == {"aliasing_max": 0.3}


def test_sample_crops_skips_undersized_images():
    img = np.zeros((64, 64), np.uint8)
    assert _sample_crops(img, 512, np.random.default_rng(0)) == []
    big = np.zeros((600, 600, 3), np.uint8)
    crops = _sample_crops(big, 512, np.random.default_rng(0))
    assert len(crops) == 8 and all(c.shape == (512, 512, 3) for c in crops)


def test_calibrate_gates_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(3)
    for i in range(4):
        img = cv2.resize(
            rng.integers(0, 256, (256, 256), dtype=np.uint8),
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(str(tmp_path / f"ref{i}.png"), img)

    suggested = calibrate_gates(tmp_path, tile_size=256, percentile=99.0)
    out = capsys.readouterr().out
    assert "Calibrated on 32 crops" in out  # 4 images x 8 crops (full-size, no jitter)
    assert "ref-kept" in out
    assert "complexity" in out  # ICNet complexity distribution from bundled weights
    assert set(suggested) == {
        "entropy_min", "lap_var_min", "grad_energy_min",
        "blockiness_max", "noise_ratio_max", "aliasing_max",
    }
