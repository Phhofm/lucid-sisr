"""Runtime applier for the ICNet score calibration.

The shipped JSON (``icnet_calibration.json``) maps raw scores at non-512
sizes to the 512-trained score distribution using a linear fit per size.
The fit was generated offline by ``scripts/calibrate_icnet.py``.

The 512 case is the identity — no transformation needed.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Supported tile sizes. Anything outside this set is hard-rejected at the
#: CLI layer. 512 is the only "native" size; the rest rely on calibration.
SUPPORTED_SIZES: tuple[int, ...] = (128, 256, 512, 1024)

#: Default threshold at the native 512 size, when no calibration entry exists.
DEFAULT_THRESHOLD: float = 0.45

_CALIBRATION_PATH = Path(__file__).resolve().parent / "icnet_calibration.json"


@lru_cache(maxsize=1)
def _load_calibration() -> dict[str, Any]:
    """Load the shipped calibration JSON once per process."""
    if not _CALIBRATION_PATH.exists():
        logger.warning(
            "Calibration JSON not found at %s; falling back to identity mapping.",
            _CALIBRATION_PATH,
        )
        return {"models": {}}
    with open(_CALIBRATION_PATH) as f:
        return json.load(f)


def reload_calibration() -> None:
    """Clear the lru_cache so a regenerated JSON is picked up on next call."""
    _load_calibration.cache_clear()


def get_model_for_size(tile_size: int) -> dict[str, Any] | None:
    """Return the calibration model entry for ``tile_size``, or None if absent."""
    cal = _load_calibration()
    return cal.get("models", {}).get(str(tile_size))


def is_supported_size(tile_size: int) -> bool:
    """Return True if ``tile_size`` is in the supported set (128, 256, 512, 1024)."""
    return tile_size in SUPPORTED_SIZES


def apply_calibration(raw_score: float, tile_size: int) -> float:
    """Map a raw ICNet score at ``tile_size`` to the 512-calibrated distribution.

    Args:
        raw_score: Score produced by ICNet at the user's chosen tile size.
        tile_size: The size the score was computed at.

    Returns:
        The calibrated score, clamped to ``[0.0, 1.0]``. The 512 case is
        identity. If no calibration entry exists for ``tile_size`` the
        function logs a warning once and returns the raw score unchanged.
    """
    if tile_size == 512:
        return float(max(0.0, min(1.0, raw_score)))
    model = get_model_for_size(tile_size)
    if model is None:
        logger.warning(
            "No calibration model for --tile-size %d; returning raw score. "
            "Run `lucid-calibrate` to generate one.",
            tile_size,
        )
        return float(max(0.0, min(1.0, raw_score)))
    transform = model.get("raw_to_calibrated", {})
    if transform.get("type") == "identity":
        return float(max(0.0, min(1.0, raw_score)))
    if transform.get("type") == "linear":
        a = float(transform.get("a", 1.0))
        b = float(transform.get("b", 0.0))
        return float(max(0.0, min(1.0, a * raw_score + b)))
    logger.warning("Unknown calibration transform type: %r", transform)
    return float(max(0.0, min(1.0, raw_score)))


def suggested_threshold(tile_size: int) -> float:
    """Return the recommended ``--threshold`` for ``tile_size``.

    For 512, returns 0.45. For other supported sizes, returns the value
    that maps to a calibrated 0.45 after the linear transform.
    """
    model = get_model_for_size(tile_size)
    if model is None:
        return DEFAULT_THRESHOLD
    return float(model.get("suggested_threshold", DEFAULT_THRESHOLD))


def calibration_r_squared(tile_size: int) -> float:
    """Return the R² of the linear fit for ``tile_size`` (1.0 for 512, 0.0 if unknown)."""
    model = get_model_for_size(tile_size)
    if model is None:
        return 0.0
    return float(model.get("r_squared", 0.0))


def calibration_status_label(tile_size: int) -> str:
    """Return a short human-readable status string for the GUI badge."""
    if tile_size == 512:
        return "Native"
    model = get_model_for_size(tile_size)
    if model is None:
        return "Uncalibrated"
    r2 = model.get("r_squared", 0.0)
    if r2 >= 0.98:
        return f"Calibrated (R^2={r2:.2f})"
    if r2 >= 0.90:
        return f"Approximate (R^2={r2:.2f})"
    return f"Loose fit (R^2={r2:.2f})"


def calibration_report() -> str:
    """Return a multi-line summary of the shipped calibration."""
    cal = _load_calibration()
    lines = [
        f"Calibration version: {cal.get('calibration_version', 'unknown')}",
        f"Calibrated on:       {cal.get('calibrated_on', 'unknown')}",
        f"Calibrated date:     {cal.get('calibrated_date', 'unknown')}",
        f"Reference size:      {cal.get('reference_size', 512)}",
        "",
    ]
    for size_str, m in cal.get("models", {}).items():
        lines.append(
            f"  size={size_str:>4s}  R^2={m.get('r_squared', 0):.3f}  "
            f"max_err={m.get('max_abs_error', 0):.4f}  "
            f"thresh={m.get('suggested_threshold', 0.45):.2f}  "
            f"n={m.get('n_observations', 0)}"
        )
    return "\n".join(lines)
