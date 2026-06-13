"""Tests for the shipped ICNet score calibration.

These cover the runtime applier, the CLI size validation, and the
``lucid-calibrate`` console script surface.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lucid.resources.icnet_calibration import (
    SUPPORTED_SIZES,
    apply_calibration,
    calibration_report,
    calibration_status_label,
    is_supported_size,
    reload_calibration,
    suggested_threshold,
)

# ---------------------------------------------------------------------------
# Applier behavior
# ---------------------------------------------------------------------------


def test_apply_calibration_identity_at_native_size():
    """512 px is the native size — calibration must be a no-op there."""
    for raw in (0.0, 0.1, 0.3, 0.45, 0.6, 0.9, 1.0):
        assert apply_calibration(raw, 512) == pytest.approx(raw, abs=1e-6)


def test_apply_calibration_is_monotonic():
    """Non-decreasing input must map to a non-decreasing output."""
    raw_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for size in SUPPORTED_SIZES:
        if size == 512:
            continue
        out = [apply_calibration(v, size) for v in raw_values]
        for prev, curr in zip(out, out[1:]):
            assert curr + 1e-9 >= prev, (
                f"Calibration not monotonic at {size}px: {prev} -> {curr}"
            )


def test_apply_calibration_clamps_to_unit_interval():
    """All outputs must lie in [0, 1] regardless of input."""
    for size in SUPPORTED_SIZES:
        for raw in (-1.0, 0.0, 0.5, 1.0, 1.7, 100.0):
            out = apply_calibration(raw, size)
            assert 0.0 <= out <= 1.0, f"Out of range for {size}px, raw={raw}: {out}"


def test_suggested_threshold_returns_useful_value():
    """The suggested threshold for any supported size should be in (0, 1)."""
    for size in SUPPORTED_SIZES:
        t = suggested_threshold(size)
        assert 0.0 < t < 1.0, f"Bad suggested threshold for {size}px: {t}"


def test_suggested_threshold_falls_back_for_unknown_size():
    """Unknown sizes get the bundled default (0.45)."""
    assert suggested_threshold(640) == pytest.approx(0.45, abs=1e-6)
    assert suggested_threshold(2048) == pytest.approx(0.45, abs=1e-6)


def test_is_supported_size_matches_allowlist():
    for size in SUPPORTED_SIZES:
        assert is_supported_size(size)
    for size in (16, 64, 96, 640, 2048, 4096):
        assert not is_supported_size(size)


def test_status_label_distinguishes_native_and_calibrated():
    """The 512-px case must be labelled differently from the others."""
    native = calibration_status_label(512)
    for size in SUPPORTED_SIZES:
        if size == 512:
            continue
        cal = calibration_status_label(size)
        assert native != cal
        assert "512" in native or "Native" in native


def test_calibration_report_includes_all_sizes():
    """The human-readable report should mention every supported size."""
    report = calibration_report()
    for size in SUPPORTED_SIZES:
        assert str(size) in report, f"Size {size} missing from calibration report"


def test_reload_calibration_refreshes_cache(tmp_path, monkeypatch):
    """``reload_calibration()`` must re-read the JSON file from disk."""
    from lucid.resources import icnet_calibration as mod

    custom_payload = {
        "calibration_version": "99",
        "calibrated_on": "unit-test",
        "calibrated_date": "1970-01-01",
        "reference_size": 512,
        "models": {
            "128": {
                "raw_to_calibrated": {"type": "linear", "a": 2.0, "b": -0.5},
                "suggested_threshold": 0.5,
                "r_squared": 0.99,
                "max_abs_error": 0.05,
                "n_observations": 42,
                "notes": "test",
            },
            "256": {
                "raw_to_calibrated": {"type": "identity"},
                "suggested_threshold": 0.45,
                "r_squared": 1.0,
                "max_abs_error": 0.0,
                "n_observations": 42,
                "notes": "identity",
            },
            "512": {
                "raw_to_calibrated": {"type": "identity"},
                "suggested_threshold": 0.45,
                "r_squared": 1.0,
                "max_abs_error": 0.0,
                "n_observations": 42,
                "notes": "native",
            },
            "1024": {
                "raw_to_calibrated": {"type": "linear", "a": 0.5, "b": 0.2},
                "suggested_threshold": 0.5,
                "r_squared": 0.95,
                "max_abs_error": 0.04,
                "n_observations": 42,
                "notes": "test",
            },
        },
    }
    fake_path = tmp_path / "icnet_calibration.json"
    fake_path.write_text(json.dumps(custom_payload))

    monkeypatch.setattr(mod, "_CALIBRATION_PATH", fake_path)
    reload_calibration()
    # 128 model: a=2.0, b=-0.5 -> apply_calibration(0.5, 128) = 2*0.5 - 0.5 = 0.5
    assert apply_calibration(0.5, 128) == pytest.approx(0.5, abs=1e-6)
    # 256 is identity: apply_calibration(0.7, 256) = 0.7
    assert apply_calibration(0.7, 256) == pytest.approx(0.7, abs=1e-6)
    # Restore the shipped calibration for any later tests
    monkeypatch.undo()
    reload_calibration()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_rejects_unsupported_tile_size():
    """Both undersized and oversized requests should exit non-zero."""
    for bad in (64, 96, 640, 2048):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lucid",
                "--input",
                "/tmp",
                "--output",
                "/tmp",
                "--tile-size",
                str(bad),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0, (
            f"--tile-size {bad} should be rejected, got exit 0"
        )
        assert "not in the supported set" in (result.stdout + result.stderr)


def test_cli_accepts_all_supported_sizes(tmp_path):
    """All four supported sizes should at least pass the validator."""
    for size in SUPPORTED_SIZES:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lucid",
                "--input", str(tmp_path),
                "--output", str(tmp_path),
                "--tile-size", str(size),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = result.stdout + result.stderr
        assert "not in the supported set" not in combined, (
            f"--tile-size {size} should be accepted, output:\n{combined}"
        )


def test_no_calibrate_flag_is_recognized(tmp_path):
    """--no-calibrate must not be rejected as an unknown flag."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lucid",
            "--input", str(tmp_path),
            "--output", str(tmp_path),
            "--tile-size", "512",
            "--dry-run",
            "--no-calibrate",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert "unrecognized arguments" not in combined
    assert "no such option" not in combined
    assert "--no-calibrate" not in combined or "Score calibration disabled" in combined


def test_lucid_calibrate_console_script_help():
    """The `lucid-calibrate` script must be installed and respond to --help.

    We invoke it as a module so this works in editable installs and in tests
    where the console-script shim may not be on PATH.
    """
    result = subprocess.run(
        [sys.executable, "-m", "lucid.calibrate", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # The wrapper either dispatches to scripts/calibrate_icnet.py (which
    # supports --help) or prints a fallback banner. Both are acceptable
    # signals that the entry point is wired correctly.
    assert result.returncode == 0, (
        f"lucid-calibrate --help failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_shipped_calibration_json_exists_and_well_formed():
    """The shipped JSON must exist and parse with all required keys."""
    from lucid.resources import icnet_calibration as mod

    path = Path(mod._CALIBRATION_PATH)  # type: ignore[attr-defined]
    assert path.exists(), f"Calibration JSON missing: {path}"
    payload = json.loads(path.read_text())
    assert "models" in payload
    assert "reference_size" in payload
    for size in SUPPORTED_SIZES:
        key = str(size)
        assert key in payload["models"], f"Size {key} missing from shipped JSON"
        entry = payload["models"][key]
        transform = entry.get("raw_to_calibrated", {})
        assert transform.get("type") in ("linear", "identity")
        if transform.get("type") == "linear":
            assert "a" in transform and "b" in transform
