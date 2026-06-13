"""Tests for the per-source xxhash dedupe pre-filter."""
from __future__ import annotations

import numpy as np
import pytest

from lucid.core.hashing import _HAS_XXHASH, filter_exact_duplicates

pytestmark = pytest.mark.skipif(not _HAS_XXHASH, reason="xxhash not installed")


def _tile(value: int, h: int = 16, w: int = 16) -> np.ndarray:
    """Make a constant-fill RGB tile."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = value
    return arr


def test_empty_input_returns_empty():
    unique, kept = filter_exact_duplicates([])
    assert unique == []
    assert kept == []


def test_no_duplicates_keeps_everything():
    a = _tile(0)
    b = _tile(128)
    c = _tile(255)
    unique, kept = filter_exact_duplicates([a, b, c])
    assert len(unique) == 3
    assert kept == [0, 1, 2]


def test_exact_duplicate_dropped():
    a = _tile(42)
    b = _tile(99)  # different
    c = _tile(42)  # duplicate of a
    unique, kept = filter_exact_duplicates([a, b, c])
    # First occurrence (a) is kept, c is dropped
    assert len(unique) == 2
    assert kept == [0, 1]
    # Verify content
    assert np.array_equal(unique[0], a)
    assert np.array_equal(unique[1], b)


def test_first_occurrence_wins():
    """If the same hash appears 3 times, only the first is kept."""
    a = _tile(7)
    b = _tile(7)
    c = _tile(7)
    unique, kept = filter_exact_duplicates([a, b, c])
    assert len(unique) == 1
    assert kept == [0]
    # And it's the *first* instance
    assert unique[0] is a or np.array_equal(unique[0], a)


def test_realistic_boundary_duplicate():
    """A 4K image tiled at 512 px often produces boundary tiles that are
    identical to neighbours. Verify the filter catches them."""
    rng = np.random.default_rng(42)
    # Build a synthetic 1024x1024 image with content
    img = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    # Tile it manually
    tiles = []
    for y in range(0, 1024, 512):
        for x in range(0, 1024, 512):
            tiles.append(img[y:y+512, x:x+512].copy())
    # All 4 tiles should be unique (synthetic data, no overlap)
    unique, kept = filter_exact_duplicates(tiles)
    assert len(unique) == 4
    # Now simulate a 5th tile that duplicates the first
    dup = tiles[0].copy()
    tiles.append(dup)
    unique2, kept2 = filter_exact_duplicates(tiles)
    assert len(unique2) == 4
    assert kept2 == [0, 1, 2, 3]
