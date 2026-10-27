"""Data-driven signal-gate calibration.

Point the command at a reference folder of tiles you already trust
(e.g. a curated validation set). It samples crops at the training tile
size, measures the six signal metrics, and derives each gate's
threshold from the distribution instead of from a magic number:

- gates with a minimum (entropy, lap_var, grad_energy) -> 1st percentile
- gates with a maximum (blockiness, noise_ratio, aliasing) -> 99th percentile

Rationale: a curated GT reference should look nothing like junk, so
"reject what clean data never measures" is the whole job of the cheap
pre-filter; semantic selection stays with ICNet + dedupe.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .core.metrics import measure_all

logger = logging.getLogger(__name__)

# gate key -> (direction, round-to). mins reject only the worst 1% of the
# reference; maxes reject only the worst 1% (99th percentile).
_GATES = {
    "entropy": ("min", 0.1),
    "lap_var": ("min", 1.0),
    "grad_energy": ("min", 0.1),
    "blockiness": ("max", 0.05),
    "noise_ratio": ("max", 0.05),
    "aliasing": ("max", 0.05),
    "clipped": ("max", 0.01),
}

_CROPS_PER_IMAGE = 8


def _sample_crops(img: np.ndarray, tile_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sample up to ``_CROPS_PER_IMAGE`` RGB crops at tile size."""
    h, w = img.shape[:2]
    if h < tile_size or w < tile_size:
        return []
    rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    crops = []
    for _ in range(_CROPS_PER_IMAGE):
        y = int(rng.integers(0, h - tile_size + 1)) if h > tile_size else 0
        x = int(rng.integers(0, w - tile_size + 1)) if w > tile_size else 0
        crops.append(rgb[y : y + tile_size, x : x + tile_size])
    return crops


def _complexity_report(
    crops: list[np.ndarray], tile_size: int, model_path: Path
) -> dict[str, float]:
    """Calibrated ICNet complexity distribution over the sampled crops."""
    import torch

    from .models.icnet import ICNet
    from .resources import resolve_device
    from .resources.icnet_calibration import apply_calibration

    device = resolve_device("auto")
    model = ICNet(is_pretrain=False, size1=tile_size, size2=tile_size // 2)
    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            chunk = crops[i : i + 32]
            t = (
                torch.stack([torch.from_numpy(c) for c in chunk])
                .to(device)
                .permute(0, 3, 1, 2)
                .float()
            )
            t = (t / 255.0 - mean) / std
            raw = model(t)[0].flatten().cpu().float().numpy()
            scores.extend(apply_calibration(float(s), tile_size) for s in raw)

    pct = {p: round(float(np.percentile(scores, p)), 3) for p in (1, 5, 50, 99)}
    print(
        f"\ncomplexity (calibrated): p1={pct[1]} p5={pct[5]} "
        f"p50={pct[50]} p99={pct[99]}"
    )
    print(f"  suggested: --threshold {pct[1]}")
    print(
        "  dataset shaping (small nets want high-complexity exposure): "
        f"--trim-bottom {50 if pct[50] < 0.45 else 0}"
    )
    return pct


def suggest_gate_thresholds(
    values: dict[str, list[float]], percentile: float = 99.0
) -> dict[str, float]:
    """Derive gate thresholds from metric distributions over trusted GT.

    Returns THRESHOLDS-key -> threshold (entropy_min, blockiness_max, ...).
    Min-direction gates take the (100 - percentile) quantile, max-direction
    gates the percentile quantile; both are rounded to a practical step.
    """
    out: dict[str, float] = {}
    for gate, (direction, step) in _GATES.items():
        data = values.get(gate, [])
        if not data:
            continue
        q = percentile if direction == "max" else 100.0 - percentile
        raw = float(np.percentile(data, q))
        out[f"{gate}_{'min' if direction == 'min' else 'max'}"] = round(
            round(raw / step) * step, 4
        )
    return out


def calibrate_gates(
    input_dir: Path,
    tile_size: int,
    percentile: float,
    icnet_path: Path | None = None,
) -> dict[str, float]:
    """Scan a reference folder and return suggested THRESHOLDS overrides."""
    import json

    import cv2

    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    from lucid.core.metrics import THRESHOLDS

    rng = np.random.default_rng(0)
    values: dict[str, list[float]] = {g: [] for g in _GATES}
    all_crops: list[np.ndarray] = []
    n_measured = 0
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Skipping unreadable %s", p.name)
            continue
        crops = _sample_crops(img, tile_size, rng)
        for crop in crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            m, _ = measure_all(gray)
            for gate, v in zip(_GATES, m):
                values[gate].append(float(v))
            n_measured += 1
        all_crops.extend(crops)
    if not n_measured:
        raise RuntimeError(
            f"No {tile_size}px crops measurable in {input_dir}; images too small?"
        )

    suggested = suggest_gate_thresholds(values, percentile)

    pct = np.percentile
    print(f"\nCalibrated on {n_measured} crops from {len(files)} images (p{percentile:g}):")
    print(f"{'metric':<12} {'p1':>9} {'p50':>9} {'p99':>9} {'current':>9} {'ref-kept':>9}")
    for gate, (direction, _) in _GATES.items():
        d = values[gate]
        if not d:
            continue
        key = f"{gate}_{'min' if direction == 'min' else 'max'}"
        if direction == "min":
            kept = float(np.mean(np.array(d) >= THRESHOLDS[key]) * 100)
        else:
            kept = float(np.mean(np.array(d) <= THRESHOLDS[key]) * 100)
        print(
            f"{gate:<12} {pct(d, 1):>9.2f} {pct(d, 50):>9.2f} {pct(d, 99):>9.2f}"
            f" {THRESHOLDS[key]:>9.2f} {kept:>8.1f}%"
        )
    print("\nIf the reference set matches your target GT quality, adopt:")
    print(json.dumps(suggested, indent=2))
    print("-> paste into THRESHOLDS in lucid/core/metrics.py")

    from lucid.models import get_default_model_path

    model_path = icnet_path or get_default_model_path()
    if model_path is not None and Path(model_path).exists():
        _complexity_report(all_crops, tile_size, Path(model_path))
    else:
        print("\n(complexity distribution skipped: ICNet weights not found)")
    return suggested


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive signal-gate thresholds from a trusted GT reference folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Reference folder of trusted GT images")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="Keep this percentile of the reference distribution")
    parser.add_argument("--icnet", type=Path, default=None,
                        help="ICNet weights path (default: bundled model)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    calibrate_gates(args.input, args.tile_size, args.percentile, args.icnet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
