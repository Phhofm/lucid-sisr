"""Signal quality metrics for tile filtering.

These metrics assess technical image properties important for SISR training:
- Entropy: Information content (avoid flat/noisy tiles)
- Laplacian variance: Sharpness/focus (avoid blurry images)
- Gradient energy: Edge density (ensure texture for learning)
- Blockiness: Compression artifacts (filter out JPG degradation)
- Noise ratio: High-frequency noise vs structure (avoid noisy tiles)
- Aliasing ratio: Oversharpening artifacts (detect haloing)
"""

import cv2
import numpy as np

# Thresholds optimized through BHI experiments for SISR training
# Key insight: These detect technical degradation, not human aesthetic quality
THRESHOLDS = {
    "entropy_min": 5.2,  # Slightly relaxed for 512px context
    "lap_var_min": 80.0,
    "lap_var_max": 10000.0,
    "blockiness_max": 45.0,
    "aliasing_max": 0.65,
    "grad_energy_min": 0.70,
    "noise_ratio_max": 0.65,
}


def _is_uniform(gray: np.ndarray) -> bool:
    """Quick check for completely uniform (single-value) tiles."""
    return np.unique(gray).size <= 1


def measure_all(gray: np.ndarray) -> tuple[list[float], bool]:
    """Run all metrics and signal rejection in one pass.

    Returns:
        (metrics_list, passes) where passes is True if the tile meets all thresholds.
    """
    if _is_uniform(gray):
        return [0.0] * 6, False

    metrics = [
        float(entropy(gray)),
        float(laplacian_variance(gray)),
        float(gradient_energy(gray)),
        float(blockiness(gray)),
        float(noise_ratio(gray)),
        float(aliasing_ratio(gray)),
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
    """Detect compression blockiness artifacts.

    Measures vertical and horizontal differences at block boundaries (every 8 pixels).
    High values indicate JPEG compression artifacts detrimental to SR training.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Normalized blockiness score (0 = no blocks, higher = more artifacts).
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    # Compare columns at block boundaries (8px spacing for JPEG blocks)
    v_right, v_left = gray[:, 8::8], gray[:, 7::8]
    v_len = min(v_right.shape[1], v_left.shape[1])
    v_score = np.sum(
        np.abs(v_right[:, :v_len].astype(np.int16) - v_left[:, :v_len].astype(np.int16))
    )

    # Compare rows at block boundaries
    h_bottom, h_top = gray[8::8, :], gray[7::8, :]
    h_len = min(h_bottom.shape[0], h_top.shape[0])
    h_score = np.sum(
        np.abs(h_bottom[:h_len, :].astype(np.int16) - h_top[:h_len, :].astype(np.int16))
    )

    return float(v_score + h_score) / (h * w)


def aliasing_ratio(gray: np.ndarray) -> float:
    """Detect aliasing/oversharpening via frequency analysis.

    High values in the outer frequency ring indicate edge enhancement artifacts
    (halos, ringing) that are too sharp for natural image learning.

    Args:
        gray: Grayscale image as numpy array.

    Returns:
        Ratio of outer ring magnitude to total (higher = more aliasing).
    """
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_grid, x_grid = np.ogrid[:h, :w]
    r = int(min(h // 2, w // 2) * 0.75)

    # Mask for outer frequency ring (high frequencies = oversharpening)
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
