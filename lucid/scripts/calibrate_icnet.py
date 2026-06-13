"""Offline calibration of ICNet complexity scores across supported tile sizes.

Generates paired observations (raw score at size X, mean score at 512) by
extracting random 1024-crops from a folder of high-resolution images and
scoring them at multiple sizes. Fits a linear mapping per non-512 size and
emits ``lucid/resources/icnet_calibration.json``.

Usage:
    python scripts/calibrate_icnet.py \\
        --input /path/to/high-res/images \\
        --output lucid/resources/icnet_calibration.json \\
        --sizes 128 256 512 1024 \\
        --crops-per-image 6

Requires the pretrained ICNet weights to be already downloaded (run
``lucid`` once on any dataset to populate ``~/.cache/lucid/complexity.pth``)
or pass ``--icnet /path/to/weights.pth`` explicitly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lucid.models import ensure_model_downloaded  # noqa: E402
from lucid.models.icnet import ICNet  # noqa: E402

logger = logging.getLogger("calibrate_icnet")

SUPPORTED_SIZES = (128, 256, 512, 1024)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PairedObservation:
    """One paired observation: mean score at each supported size from the same crop."""

    source_image: str
    crop_xy: tuple[int, int]
    scores: dict[str, float]


def extract_random_crops(
    image: np.ndarray,
    crop_size: int,
    n_crops: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, np.ndarray]]:
    """Return ``n_crops`` (x, y, crop) tuples of size ``crop_size`` from ``image``."""
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        logger.warning("Image %dx%d smaller than crop %d; skipping", w, h, crop_size)
        return []
    out: list[tuple[int, int, np.ndarray]] = []
    for _ in range(n_crops):
        x = int(rng.integers(0, w - crop_size + 1))
        y = int(rng.integers(0, h - crop_size + 1))
        out.append((x, y, image[y : y + crop_size, x : x + crop_size].copy()))
    return out


def _normalize_batch(tiles_bgr: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Preprocess a batch of BGR uint8 tiles to a normalized NCHW float32 tensor."""
    rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in tiles_bgr]
    arr = np.stack(rgb).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
    return tensor


@torch.no_grad()
def score_tiles(
    model: ICNet, tiles_bgr: list[np.ndarray], device: torch.device
) -> np.ndarray:
    """Score a batch of tiles, return a 1-D numpy array of complexity scores."""
    if not tiles_bgr:
        return np.empty(0, dtype=np.float32)
    x = _normalize_batch(tiles_bgr, device)
    out = model(x)
    score = out[0]
    if score.dim() == 0:
        score = score.unsqueeze(0)
    return score.detach().cpu().float().numpy()


def compute_observations(
    input_dir: Path,
    sizes: tuple[int, ...],
    crops_per_image: int,
    weights_path: Path,
    device: torch.device,
    use_half: bool,
    seed: int = 0,
) -> list[PairedObservation]:
    """For each image, draw N random 1024-crops and score them at all sizes.

    A fresh ICNet is instantiated for each size because the runtime model
    asserts that input matches its built-in size. The architecture itself
    is size-flexible (slam interpolates, up_conv uses scale_factor, the head
    uses AdaptiveAvgPool) — only the constructor's ``size1`` parameter
    varies between instantiations.
    """
    rng = np.random.default_rng(seed)
    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    logger.info(
        "Found %d images; sampling %d crops each", len(image_paths), crops_per_image
    )

    # Pre-load one model per size; sharing across sizes is not allowed by the
    # current forward() assert. Memory cost is small (~2x ResNet-18 ≈ 200 MB).
    models: dict[int, ICNet] = {}
    for size in sizes:
        m = ICNet(is_pretrain=False, size1=size, size2=size // 2)
        m.load_state_dict(
            torch.load(str(weights_path), map_location=device, weights_only=True)
        )
        m.to(device).eval()
        if use_half:
            m.half()
        models[size] = m
    logger.info("Loaded %d ICNet instances (one per size)", len(models))

    observations: list[PairedObservation] = []
    for img_path in image_paths:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Could not read %s; skipping", img_path.name)
            continue
        crops = extract_random_crops(bgr, 1024, crops_per_image, rng)
        if not crops:
            continue
        for x_off, y_off, crop1024 in crops:
            scores: dict[str, float] = {}
            for size in sizes:
                model = models[size]
                if size == 1024:
                    tiles = [crop1024]
                else:
                    tiles = []
                    for ty in range(0, 1024, size):
                        for tx in range(0, 1024, size):
                            tiles.append(crop1024[ty : ty + size, tx : tx + size])
                raw = score_tiles(model, tiles, device)
                scores[str(size)] = float(raw.mean()) if raw.size else 0.0
            observations.append(
                PairedObservation(
                    source_image=img_path.name,
                    crop_xy=(int(x_off), int(y_off)),
                    scores=scores,
                )
            )
        logger.info("  %s: %d paired observations", img_path.name, len(crops))
    return observations


def fit_linear(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """Fit y = a*x + b by least squares. Returns (a, b, r_squared, max_abs_error)."""
    if xs.size < 2:
        return 1.0, 0.0, 0.0, float("inf")
    a, b = np.polyfit(xs, ys, 1)
    preds = a * xs + b
    residuals = preds - ys
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    mae = float(np.max(np.abs(residuals)))
    return float(a), float(b), float(r2), mae


def suggested_threshold(target_score: float, a: float, b: float) -> float:
    """Given calibrated-target 0.45, return the raw-score threshold that maps to it.

    Solving ``a*x + b = target`` gives ``x = (target - b) / a``.
    """
    if a == 0:
        return float(target_score)
    return float((target_score - b) / a)


def build_calibration(
    observations: list[PairedObservation],
    reference_size: int = 512,
) -> dict:
    """Fit a linear mapping per non-reference size and build the JSON payload."""
    ref = np.array([o.scores[str(reference_size)] for o in observations], dtype=np.float64)
    models: dict[str, dict] = {}

    models[str(reference_size)] = {
        "raw_to_calibrated": {"type": "identity"},
        "suggested_threshold": 0.45,
        "r_squared": 1.0,
        "max_abs_error": 0.0,
        "n_observations": int(ref.size),
        "notes": "Native training size; default and reference",
    }

    for size in SUPPORTED_SIZES:
        if size == reference_size:
            continue
        raw = np.array([o.scores[str(size)] for o in observations], dtype=np.float64)
        a, b, r2, mae = fit_linear(raw, ref)
        thresh = suggested_threshold(0.45, a, b)
        thresh = float(np.clip(thresh, 0.10, 0.80))
        notes = {
            128: "Smallest supported; for 4x SR with 128 patch size. Drift expected; linear may be loose.",
            256: "Standard 2x SR LR size; ImageNet / face / icon data.",
            1024: "Largest supported; high-res scans. Context branch at native 512 so drift is small.",
        }[size]
        models[str(size)] = {
            "raw_to_calibrated": {"type": "linear", "a": a, "b": b},
            "suggested_threshold": thresh,
            "r_squared": r2,
            "max_abs_error": mae,
            "n_observations": int(raw.size),
            "notes": notes,
        }
    return {
        "calibration_version": "1.0.0",
        "calibrated_on": (
            f"{len({o.source_image for o in observations})} images, "
            f"{len(observations)} paired observations per size"
        ),
        "calibrated_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "reference_size": reference_size,
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder of high-res images")
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to write icnet_calibration.json"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(SUPPORTED_SIZES),
        help="Tile sizes to score at (default: 128 256 512 1024)",
    )
    parser.add_argument(
        "--crops-per-image", type=int, default=6,
        help="Number of random 1024-crops per image (default: 6)",
    )
    parser.add_argument(
        "--icnet", type=Path, default=None,
        help="Path to ICNet .pth weights (default: auto-download)",
    )
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/xpu/mps/cpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.input.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    for s in args.sizes:
        if s not in SUPPORTED_SIZES:
            raise ValueError(f"Size {s} not in supported set {SUPPORTED_SIZES}")

    from lucid.resources import resolve_device

    device = resolve_device(args.device)
    # Calibration runs in FP32 regardless of accelerator. The score drift
    # introduced by FP16 (a few percent) would corrupt the calibration
    # itself; we want the most accurate raw score possible here.
    use_half = False

    model_path = args.icnet or ensure_model_downloaded()
    logger.info("Using device=%s, weights=%s (FP32 for calibration)", device, model_path)

    observations = compute_observations(
        input_dir=args.input,
        sizes=tuple(args.sizes),
        crops_per_image=args.crops_per_image,
        weights_path=model_path,
        device=torch.device(device),
        use_half=use_half,
        seed=args.seed,
    )
    if not observations:
        raise RuntimeError("No observations collected; check input folder and image sizes")

    calibration = build_calibration(observations, reference_size=512)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info("Wrote calibration to %s", args.output)

    print()
    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Source: {calibration['calibrated_on']}")
    print(f"Date:   {calibration['calibrated_date']}")
    print()
    for size_str, m in calibration["models"].items():
        print(
            f"  size={size_str:>4s}  R^2={m['r_squared']:.3f}  "
            f"max_err={m['max_abs_error']:.4f}  "
            f"thresh={m['suggested_threshold']:.2f}  "
            f"n={m['n_observations']}"
        )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
