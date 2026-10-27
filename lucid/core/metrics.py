"""Signal quality metrics for tile filtering.

These metrics assess technical image properties important for SISR training:
- Entropy: Information content (avoid flat/noisy tiles)
- Laplacian variance: Sharpness/focus (avoid blurry images)
- Gradient energy: Edge density (ensure texture for learning)
- Blockiness: Compression artifacts (filter out JPG degradation)
- Noise ratio: High-frequency noise vs structure (avoid noisy tiles)
- Aliasing ratio: Oversharpening artifacts (detect haloing)
- Clipped ratio: Blown highlights / crushed shadows (degenerate frames)
"""

import cv2
import numpy as np

# Thresholds optimized through BHI experiments for SISR training
# Key insight: These detect technical degradation, not human aesthetic quality
THRESHOLDS = {
    "entropy_min": 5.2,  # Slightly relaxed for 512px context
    "lap_var_min": 80.0,
    "lap_var_max": 10000.0,
    # blockiness is a normalized ratio (boundary vs interior gradient),
    # ~1.0 for clean content; Q30 JPEG flats measure ~2.0
    "blockiness_max": 1.7,
    # Hann-windowed ring-energy ratio, calibrated on curated 512px GT:
    # clean natural photos measure 0.31-0.74 (median 0.51); heavy noise
    # / UI patterns / strong ringing land above ~0.80
    "aliasing_max": 0.80,
    "grad_energy_min": 0.70,
    # hf/gradient proxy, calibrated on curated 512px GT: clean photos
    # measure 0.32-1.00 (median 0.66); pre-filter for heavy noise
    # (sigma~25) only — moderate noise is the ICNet/dedupe stages' job
    "noise_ratio_max": 1.35,
    # fraction of pixels at the 0/255 rails; curated reference tops out at
    # 32.5% (snow/sky), so the ceiling only catches near-solid blowouts
    "clipped_max": 0.40,
}


def _is_uniform(gray: np.ndarray) -> bool:
    """Quick check for completely uniform (single-value) tiles."""
    return np.unique(gray).size <= 1


def clipped_ratio(gray: np.ndarray) -> float:
    """Fraction of pixels sitting on the 0/255 rails.

    Near-solid blown or black frames carry no recoverable signal for SR
    training. Curated GT legitimately clips up to ~32% (snow, sky), so only
    degenerate frames exceed the ceiling.
    """
    g = np.asarray(gray)
    return float(np.mean((g <= 2) | (g >= 253)))


def measure_all(gray: np.ndarray) -> tuple[list[float], bool]:
    """Run all metrics and signal rejection in one pass.

    Returns:
        (metrics_list, passes) where passes is True if the tile meets all thresholds.
    """
    if _is_uniform(gray):
        return [0.0] * 7, False

    metrics = [
        float(entropy(gray)),
        float(laplacian_variance(gray)),
        float(gradient_energy(gray)),
        float(blockiness(gray)),
        float(noise_ratio(gray)),
        float(aliasing_ratio(gray)),
        float(clipped_ratio(gray)),
    ]

    if metrics[0] < THRESHOLDS["entropy_min"]:
        return metrics, False
    if not (THRESHOLDS["lap_var_min"] < metrics[1] < THRESHOLDS["lap_var_max"]):
        return metrics, False
    if metrics[2] < THRESHOLDS["grad_energy_min"]:
        return metrics, False
    if metrics[3] > THRESHOLDS["blockiness_max"]:
        return metrics, False
    if metrics[4] > THRESHOLDS["noise_ratio_max"]:
        return metrics, False
    if metrics[5] > THRESHOLDS["aliasing_max"]:
        return metrics, False
    if metrics[6] > THRESHOLDS["clipped_max"]:
        return metrics, False

    return metrics, True


def entropy(gray: np.ndarray) -> float:
    """Calculate image entropy (information content).

    High entropy indicates diverse pixel values (good information).
    Low entropy suggests flat, uniform regions (poor learning signal).

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Entropy value in bits (0-8 for 8-bit images).
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    p = hist.ravel() / (hist.sum() + 1e-9)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def laplacian_variance(gray: np.ndarray) -> float:
    """Calculate Laplacian variance (sharpness/focus metric).

    Measures edge strength - too low means blurry, too high means oversharpened.
    The sweet spot (80-10000) captures naturally sharp images.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Variance of Laplacian (higher = sharper).
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def blockiness(gray: np.ndarray) -> float:
    """Detect JPEG 8x8 blockiness artifacts.

    Compares the mean gradient across JPEG block boundaries (every 8 px)
    against the mean interior gradient. Ratios near 1.0 mean boundary
    discontinuities match natural content; JPEG blocks push the boundary
    mean above the interior mean, especially in flat regions.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Boundary/interior gradient ratio (1.0 = no block artifacts).
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    g = gray.astype(np.int16)

    # Adjacent-difference fields; JPEG boundary pairs sit at 7|8 offsets.
    dv = np.abs(g[:, 1:] - g[:, :-1]).astype(np.float64)
    dh = np.abs(g[1:, :] - g[:-1, :]).astype(np.float64)
    v_mask = np.zeros(dv.shape[1], dtype=bool)
    v_mask[7::8] = True
    h_mask = np.zeros(dh.shape[0], dtype=bool)
    h_mask[7::8] = True

    boundary = (dv[:, v_mask].mean() + dh[h_mask, :].mean()) / 2.0
    interior = (dv[:, ~v_mask].mean() + dh[~h_mask, :].mean()) / 2.0
    return float((boundary + 1.0) / (interior + 1.0))


def aliasing_ratio(gray: np.ndarray) -> float:
    """Detect high-frequency pollution via windowed frequency analysis.

    Stripes, UI text/layout patterns, ringing, and heavy sharpening put
    energy in the outer frequency ring. A Hann window suppresses the
    tile-boundary discontinuity leakage that made the unwindowed version
    reject ordinary sharp natural content.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Ratio of outer ring magnitude to total (higher = more pollution).
    """
    h, w = gray.shape
    windowed = gray.astype(np.float64) * np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    r = int(min(h // 2, w // 2) * 0.75)

    # Mask for outer frequency ring
    mask = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) >= r
    return float(mag[mask].mean() / (mag.mean() + 1e-9))


def gradient_energy(gray: np.ndarray) -> float:
    """Calculate gradient energy (edge density).

    Mean Sobel magnitude - ensures tiles have sufficient texture for SR learning.
    Very low values indicate smooth/uniform regions with little information.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Mean gradient magnitude (higher = more edges).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return float(np.mean(cv2.sqrt(gx**2 + gy**2)))


def noise_ratio(gray: np.ndarray) -> float:
    """Calculate noise-to-gradient ratio.

    High values indicate high-frequency noise dominates structural gradients,
    suggesting noisy images that confuse SR training.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Ratio of high-frequency energy to gradient energy (lower is better).
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hf = gray.astype(np.float32) - blur.astype(np.float32)
    hf_energy = np.mean(hf**2)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad_energy = np.mean(gx**2 + gy**2)

    return float(hf_energy / (grad_energy + 1e-9))
