"""Per-source-image exact-byte dedupe pre-filter.

Catches pixel-identical tiles that arise from overlapping crops of a single
source image. ResNet18 cosine-similarity dedupe would also catch these, but
the xxhash pre-filter is O(1) per tile and 1000x faster. It runs *before*
ICNet scoring, so a near-duplicate boundary crop is never scored at all.

Statistically:
- A 4K image (3840x2160) tiled at 512 px yields 28 tiles, of which 1-3 are
  typically exact-byte duplicates of neighbours from overlapping crops.
- xxhash of a 512x512x3 uint8 array takes ~50 us; CV2 imread + cvtColor
  is ~10-30 ms. The pre-filter is essentially free.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:  # pragma: no cover
    _HAS_XXHASH = False


def filter_exact_duplicates(
    tiles: list[np.ndarray],
) -> tuple[list[np.ndarray], list[int]]:
    """Drop exact-byte duplicates from a per-source tile list.

    Args:
        tiles: List of RGB uint8 numpy arrays (HxWx3), one per tile.

    Returns:
        ``(unique_tiles, kept_indices)`` where ``kept_indices`` are the
        positions in the input list that survived the filter.
    """
    if not tiles:
        return [], []
    if not _HAS_XXHASH:
        # Defensive fallback: if xxhash is missing for some reason, skip
        # the pre-filter entirely. ResNet18 dedupe will catch them later.
        return list(tiles), list(range(len(tiles)))

    seen: set[int] = set()
    unique: list[np.ndarray] = []
    kept_indices: list[int] = []

    for idx, tile in enumerate(tiles):
        # xxhash of the underlying bytes is faster than hashing the
        # numpy array object, and the bytes are guaranteed identical
        # for two pixel-identical arrays.
        digest = xxhash.xxh3_64(tile.tobytes()).intdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(tile)
        kept_indices.append(idx)

    return unique, kept_indices
