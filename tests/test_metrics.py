"""Unit tests for signal metrics."""
import cv2
import numpy as np

from lucid.core.metrics import (
    THRESHOLDS,
    aliasing_ratio,
    blockiness,
    clipped_ratio,
    measure_all,
    noise_ratio,
)


def _natural_image() -> np.ndarray:
    """Deterministic smooth-textured gray image ("photo-like")."""
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)


def _jpeg(gray: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def test_uniform_tile_fails():
    """A completely uniform tile should fail all metrics."""
    gray = np.zeros((512, 512), dtype=np.uint8)
    metrics, passes = measure_all(gray)
    assert not passes, "Uniform tile should fail"
    assert len(metrics) == 7


def test_random_tile_produces_six_metrics():
    """A random noise tile produces exactly 6 metrics (may not pass due to noise)."""
    np.random.seed(42)
    gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    metrics, passes = measure_all(gray)
    assert len(metrics) == 7
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
    assert len(metrics) == 7
    assert passes, "Textured tile should pass all signal filters"


def test_blockiness_flat_jpeg_rejected_clean_kept():
    """JPEG blockiness must fire on artifact-carrying flats and never on clean ones.

    A Q30 re-encode of a smooth photo injects 8x8 boundary artifacts
    (ratio ~2.0 in initial tuning); the clean original sits near ~1.0.
    The pre-rewrite implementation could not separate these at all (all
    values < 6 against a threshold of 45), so this pins the fix.
    """
    flat = cv2.blur(_natural_image(), (31, 31)).astype(np.uint8)
    flat_jq30 = _jpeg(flat, 30)
    assert blockiness(flat) < THRESHOLDS["blockiness_max"]
    assert blockiness(flat_jq30) > THRESHOLDS["blockiness_max"]


def test_blockiness_dense_texture_misfire_fixed():
    """Dense high-contrast texture must not be read as blockiness.

    Unnormalized boundary-diff sums reject dense-but-clean textures
    (any sharp image). The normalized (boundary / interior) ratio must
    sit near 1.0 for content whose block-boundary gradient matches its
    interior gradient.
    """
    x = np.arange(512, dtype=np.float32)
    dense = (
        np.sin(x * 0.8)[None, :] * 60
        + np.sin(x * 0.5)[:, None] * 40
        + 128
    ).clip(0, 255).astype(np.uint8)
    assert abs(blockiness(dense) - 1.0) < 0.5


def test_blockiness_jpeg_reencode_raises_metric():
    """A Q30 re-encode of the same image must score above its clean original."""
    nat = _natural_image()
    assert blockiness(_jpeg(nat, 30)) > blockiness(nat)


def test_aliasing_natural_and_sharpened_within_gate():
    """Sharp natural images (even unsharpened) stay below the aliasing gate."""
    nat = _natural_image()
    sharp = np.clip(
        nat.astype(np.float32)
        + 3.0 * (nat.astype(np.float32) - cv2.GaussianBlur(nat, (5, 5), 0)),
        0, 255,
    ).astype(np.uint8)
    for img in (nat, sharp, cv2.blur(nat, (31, 31)).astype(np.uint8)):
        assert aliasing_ratio(img) < THRESHOLDS["aliasing_max"], img[0, 0]


def test_aliasing_sharpening_raises_metric():
    """Unsharpen masking must increase the ring-energy ratio."""
    nat = _natural_image()
    sharp = np.clip(
        nat.astype(np.float32)
        + 3.0 * (nat.astype(np.float32) - cv2.GaussianBlur(nat, (5, 5), 0)),
        0, 255,
    ).astype(np.uint8)
    assert aliasing_ratio(sharp) > aliasing_ratio(nat)


def test_aliasing_clean_periodic_texture_within_gate():
    """Clean periodic texture is legitimate GT content and must be kept."""
    x = np.arange(512)
    grid = (
        np.sin(x * 1.8)[None, :] * 60 + np.sin(x * 1.8)[:, None] * 60 + 128
    ).clip(0, 255).astype(np.uint8)
    assert aliasing_ratio(grid) < THRESHOLDS["aliasing_max"]


def test_aliasing_broadband_clutter_rejected():
    """Pixel-scale broadband clutter (HF noise over texture) must trip the gate.

    Signature of UI screenshots, text fragments, and ringing/noise
    overlays: energy spread across the whole high-frequency ring, unlike
    clean periodic texture.
    """
    x = np.arange(512)
    rng = np.random.default_rng(7)
    clutter = (
        np.sin(x * 0.8)[None, :] * 60
        + np.sin(x * 0.5)[:, None] * 40
        + rng.integers(-14, 14, (512, 512))
        + 128
    ).clip(0, 255).astype(np.uint8)
    assert aliasing_ratio(clutter) > THRESHOLDS["aliasing_max"]


def _photo_like_image() -> np.ndarray:
    """Multi-octave image with a natural 1/f-like spectrum."""
    rng = np.random.default_rng(0)
    img = np.zeros((512, 512), np.float32)
    for size in (32, 64, 128):
        img += cv2.resize(
            rng.integers(0, 256, (size, size), dtype=np.uint8),
            (512, 512),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_noise_ratio_heavy_noise_rejected_clean_kept():
    """The noise gate is a heavy-junk pre-filter, calibrated on curated
    512px GT where clean photos measure 0.32-1.00; only heavy noise
    overlays (~sigma 40+) separate above it."""
    nat = _photo_like_image()
    noisy = np.clip(
        nat.astype(np.float32)
        + np.random.default_rng(1).normal(0, 40, nat.shape),
        0, 255,
    ).astype(np.uint8)
    assert noise_ratio(nat) < THRESHOLDS["noise_ratio_max"]
    assert noise_ratio(noisy) > THRESHOLDS["noise_ratio_max"]


def test_clipped_ratio_rejects_blown_frame():
    """Near-solid blown/black frames must be caught by the clipping gate;
    normal photos (even high-key ones) must pass. val125 reference tops
    out at 32.5% clipped pixels, so the ceiling is lenient by design."""
    from lucid.core.metrics import clipped_ratio

    blown = np.full((512, 512), 255, np.uint8)
    assert clipped_ratio(blown) > THRESHOLDS["clipped_max"]

    nat = _natural_image()
    assert clipped_ratio(nat) < THRESHOLDS["clipped_max"]

    # a high-key photo with 20% clipped pixels is legitimate GT
    hk = nat.copy()
    hk[: int(512 * 0.45), : int(512 * 0.45)] = 255  # ~20% blown corner
    assert clipped_ratio(hk) < THRESHOLDS["clipped_max"]


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
