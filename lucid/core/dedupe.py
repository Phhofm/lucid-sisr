"""ResNet18-based deduplication with HNSW approximate nearest neighbours.

Tiles are embedded as 512-D feature vectors using a pre-trained ResNet-18
(classification head removed). Each tile is compared against the index of
already-accepted tiles using cosine similarity. Tiles above the threshold
are considered redundant and moved to a ``redundant/`` subdirectory.

The index is an HNSW graph (via ``hnswlib``) when available, giving O(log N)
queries and O(1) memory per tile — scales comfortably to ~10M tiles. When
``hnswlib`` is not installed, the module falls back to a brute-force cosine
matmul against an in-memory pool, which is the original behaviour and
acceptable for datasets up to a few hundred thousand tiles.
"""
from __future__ import annotations

import csv
import gc
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Skip tiles larger than this dimension to prevent RAM/CPU stalls during load
MAX_IMAGE_DIM = 4096
FEATURE_DIM = 512

# HNSW graph parameters — tuned for 512-D float32 cosine on a CPU index.
# These are conservative defaults; users with very large pools can tune
# ``M`` and ``ef_construction`` higher for recall at the cost of build time.
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_QUERY = 64

try:
    import hnswlib  # type: ignore[import-not-found]

    _HNSW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on systems without hnswlib
    _HNSW_AVAILABLE = False


def hnswlib_available() -> bool:
    """Return ``True`` if the optional HNSW backend is importable."""
    return _HNSW_AVAILABLE


def _get_feature_extractor(device: str) -> nn.Module:
    """Load a pre-trained ResNet-18 and strip the classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final FC layer; we want the 512-D global avg-pool output
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    # FP16 on real accelerators (matches the ICNet scoring path); MPS keeps FP32.
    if device.startswith(("cuda", "xpu")):
        model.half()
    return model


def _preload_transform() -> transforms.Compose:
    """Standard ImageNet-normalization pipeline, resizing to 224px."""
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class _HNSWIndex:
    """Thin wrapper around hnswlib that keeps accepted-tile vectors in RAM.

    HNSW stores its own copy of the vectors, so we don't need a parallel
    tensor. The pool size is bounded by ``max_elements`` at construction;
    if the cap is hit, the index is rebuilt at 2x size (rare and logged).
    """

    def __init__(self, max_elements: int) -> None:
        if not _HNSW_AVAILABLE:
            raise RuntimeError("hnswlib is not installed")
        self._index = hnswlib.Index(space="cosine", dim=FEATURE_DIM)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=_HNSW_EF_CONSTRUCTION,
            M=_HNSW_M,
        )
        self._index.set_ef(_HNSW_EF_QUERY)
        self._max_elements = max_elements
        self._size = 0
        self._next_id = 0

    def add(self, vectors: np.ndarray) -> None:
        """Append ``vectors`` (shape ``[N, 512]``, dtype float32, L2-normalised)."""
        if vectors.size == 0:
            return
        n = vectors.shape[0]
        if self._size + n > self._max_elements:
            self._grow(needed=self._size + n)
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._index.add_items(vectors, ids)
        self._size += n
        self._next_id += n

    def max_cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        """Return the maximum cosine similarity of each query row to any stored item.

        ``query`` shape ``[Q, 512]``, dtype float32, L2-normalised. Returns a
        length-``Q`` float32 array. An empty index returns all zeros.
        """
        if self._size == 0 or query.size == 0:
            return np.zeros(query.shape[0], dtype=np.float32)
        # hnswlib returns cosine *distance* = 1 - cos_sim, so we flip the sign.
        _labels, distances = self._index.knn_query(query, k=1)
        return (1.0 - distances[:, 0]).astype(np.float32)

    def _grow(self, needed: int) -> None:
        """Grow the index capacity in place, preserving all stored items."""
        new_cap = max(needed, self._max_elements * 2)
        logger.info(
            "HNSW dedupe index growing from %d to %d elements.",
            self._max_elements,
            new_cap,
        )
        self._index.resize_index(new_cap)
        self._max_elements = new_cap


def _brute_force_max_sims(
    features: torch.Tensor, pool: torch.Tensor
) -> torch.Tensor:
    """Reference implementation: O(Batch × Pool) cosine matmul on the active device."""
    if pool.size(0) == 0:
        return torch.zeros(features.size(0), device=features.device)
    sims = torch.mm(features, pool.t())
    return sims.max(dim=1).values


def _trim_bottom(
    tile_paths: list[Path], scores_csv: Path, pct: int, dest_dir: Path
) -> tuple[list[Path], int]:
    """Move the lowest-complexity ``pct``% of tiles to ``dest_dir``.

    BHI finding: removing only the worst tiles (bottom percentile by
    complexity) beats keeping only the best. Tiles missing from the CSV
    are kept. A missing CSV disables the trim.

    Returns:
        (remaining_tile_paths, moved_count).
    """
    if pct <= 0:
        return tile_paths, 0
    try:
        with open(scores_csv, newline="") as f:
            scores = {
                row["tile_name"]: float(row["score_calibrated"])
                for row in csv.DictReader(f)
            }
    except FileNotFoundError:
        logger.warning("trim_bottom: %s not found; skipping percentile trim", scores_csv)
        return tile_paths, 0

    ranked = sorted(
        tile_paths, key=lambda p: scores.get(p.name, float("inf"))
    )
    n_drop = min(len(ranked), len(ranked) * pct // 100)
    dest_dir.mkdir(exist_ok=True)
    for p in ranked[:n_drop]:
        try:
            shutil.move(str(p), str(dest_dir / p.name))
        except OSError as exc:
            logger.warning("trim_bottom: could not move %s: %s", p, exc)
    moved = [p for p in ranked[:n_drop] if not p.exists()]
    logger.info("trim_bottom: dropped %d lowest-complexity tiles", len(moved))
    return ranked[n_drop:], len(moved)


def run_dedupe(
    input_dir: Path,
    threshold: float,
    batch_size: int,
    device_preference: str = "auto",
    io_throttle: float = 0.0,
    scores_csv: Path | None = None,
    trim_bottom: int = 0,
) -> tuple[int, int]:
    """Run deduplication on a folder of tiles.

    Args:
        input_dir: Directory containing tiles to deduplicate.
        threshold: Cosine similarity cutoff (0.96 = very strict, 0.90 = loose).
        batch_size: Number of tiles to process per GPU inference pass.
        device_preference: ``"auto"``, ``"cuda"``, ``"xpu"``, ``"mps"``, or ``"cpu"``.
        io_throttle: Seconds to sleep between batches (helps on slow USB drives).
        scores_csv: Ingest scores CSV used by ``trim_bottom``.
        trim_bottom: Drop the lowest-complexity N% of tiles first (0 = off).

    Note:
        Comparison streams per tile: each tile is matched against all
        tiles accepted so far, including earlier tiles of its own batch.

    Returns:
        ``(kept_count, removed_count)``; trimmed and unreadable/oversized
        tiles are excluded from both.
    """
    from lucid.resources import resolve_device

    device = resolve_device(device_preference)
    model = _get_feature_extractor(device)
    preprocess = _preload_transform()

    tile_paths = sorted(
        p for p in input_dir.rglob("*.png") if "redundant" not in p.parts
    )

    redundant_dir = input_dir / "redundant"
    tile_paths, trimmed = _trim_bottom(
        tile_paths, scores_csv, trim_bottom, redundant_dir
    )
    logger.info("Found %d tiles to dedupe on %s", len(tile_paths), device)

    use_hnsw = _HNSW_AVAILABLE
    if not use_hnsw:
        logger.info(
            "hnswlib not installed; using brute-force cosine matmul. "
            "Install with `pip install lucid-sisr[ann]` for ~50x faster dedupe "
            "and scaling beyond ~200k unique tiles."
        )

    hnsw_index: _HNSWIndex | None = None
    unique_pool = torch.empty((0, FEATURE_DIM), device=device)
    redundant: list[Path] = []

    if use_hnsw:
        # Oversize by 2x to absorb over-counting from duplicates; cap at 10M.
        hnsw_index = _HNSWIndex(
            max_elements=min(max(len(tile_paths) * 2, 100_000), 10_000_000)
        )

    processed = 0
    for i in tqdm(range(0, len(tile_paths), batch_size), desc="Auditing diversity"):
        batch = tile_paths[i : i + batch_size]
        if not batch:
            continue

        # Parallel decode + preprocessing on CPU (I/O bound); features are
        # computed in FP16 on CUDA/XPU, then normalized in FP32.
        def _tile_tensor(p: Path) -> torch.Tensor | None:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
                        return None
                    return preprocess(im.convert("RGB"))
            except Exception as exc:
                logger.warning("Skipping unreadable tile %s: %s", p, exc)
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            decoded = list(pool.map(_tile_tensor, batch))
        tensors = [t for t in decoded if t is not None]
        valid = [p for p, t in zip(batch, decoded) if t is not None]

        if not tensors:
            continue

        processed += len(valid)

        feat = torch.stack(tensors).to(device)
        if device.startswith(("cuda", "xpu")):
            feat = feat.half()
        with torch.no_grad():
            features = model(feat).squeeze()
            if features.dim() == 1:
                features = features.unsqueeze(0)
            features = features.float()
            features = nn.functional.normalize(features, p=2, dim=1)

            # Stream: each tile is compared against everything accepted so
            # far (including earlier tiles of this batch), so duplicates
            # processed together still see each other.
            for idx in range(features.size(0)):
                p = valid[idx]
                f = features[idx].unsqueeze(0)
                if hnsw_index is not None:
                    f_cpu = f.detach().cpu().numpy().astype(np.float32)
                    max_sim = float(hnsw_index.max_cosine_similarity(f_cpu)[0])
                    if max_sim > threshold:
                        redundant.append(p)
                    else:
                        hnsw_index.add(f_cpu)
                else:
                    max_sim = float(_brute_force_max_sims(f, unique_pool)[0])
                    if max_sim > threshold:
                        redundant.append(p)
                    else:
                        unique_pool = torch.cat([unique_pool, f], dim=0)

        del tensors, feat, features
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if io_throttle > 0:
            time.sleep(io_throttle)

    if redundant:
        redundant_dir.mkdir(exist_ok=True)
        logger.info("Moving %d redundant tiles to %s", len(redundant), redundant_dir)
        rename_counters: dict[str, int] = {}
        for p in tqdm(redundant, desc="Moving duplicates"):
            try:
                dest = redundant_dir / p.name
                if dest.exists():
                    base = p.stem
                    if base not in rename_counters:
                        rename_counters[base] = 0
                    rename_counters[base] += 1
                    dest = redundant_dir / f"{base}_{rename_counters[base]}.png"
                shutil.move(str(p), str(dest))
            except Exception as exc:
                logger.warning("Could not move duplicate %s: %s", p, exc)

    skipped = len(tile_paths) - processed
    if skipped > 0:
        logger.warning(
            "%d tiles could not be read or exceeded %dpx and were left in place "
            "(not counted as kept).", skipped, MAX_IMAGE_DIM
        )

    return processed - len(redundant), len(redundant) + trimmed
