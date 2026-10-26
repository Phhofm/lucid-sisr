"""Tests for the device-resolution and HNSW-dedupe helpers."""
from __future__ import annotations

import numpy as np
import pytest

from lucid.core import dedupe
from lucid.resources import (
    detect_accelerator,
    resolve_device,
)


def test_detect_accelerator_returns_known_label():
    """The accelerator label must be one of the known strings."""
    label = detect_accelerator()
    assert label in {
        "CUDA",
        "ROCm (AMD GPU via HIP)",
        "XPU (Intel GPU)",
        "MPS (Apple Silicon GPU)",
        "DirectML (Windows)",
        "CPU",
    }


def test_resolve_device_auto_returns_known_device():
    """``resolve_device('auto')`` must return a real torch device string."""
    device = resolve_device("auto")
    assert device in {"cuda", "xpu", "mps", "cpu"}


def test_resolve_device_cpu_always_works():
    """Explicit CPU must always succeed even without an accelerator."""
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_unknown_falls_back_to_cpu():
    """A garbage preference must log a warning and fall back to CPU."""
    assert resolve_device("not-a-device") == "cpu"


@pytest.mark.parametrize(
    "pref",
    ["cuda", "xpu", "mps", "directml", "cpu"],
)
def test_resolve_device_handles_each_preference(pref: str):
    """Every documented preference string must produce a valid device string."""
    device = resolve_device(pref)
    assert device in {"cuda", "xpu", "mps", "cpu"}


def test_hnswlib_availability_reported():
    """``hnswlib_available()`` returns a bool, regardless of install state."""
    assert isinstance(dedupe.hnswlib_available(), bool)


@pytest.mark.skipif(
    not dedupe.hnswlib_available(),
    reason="hnswlib not installed; skip HNSW-specific tests",
)
def test_hnsw_index_rejects_duplicates():
    """A near-identical vector must match the existing one above 0.9 sim."""
    idx = dedupe._HNSWIndex(max_elements=100)
    rng = np.random.default_rng(seed=42)
    base = rng.normal(size=(1, dedupe.FEATURE_DIM)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    idx.add(base)
    # Identical vector → cosine sim ≈ 1.0 → must be flagged as duplicate
    sims = idx.max_cosine_similarity(base.copy())
    assert sims[0] > 0.99

    # A random vector with low similarity → must be < 0.5
    other = rng.normal(size=(1, dedupe.FEATURE_DIM)).astype(np.float32)
    other /= np.linalg.norm(other, axis=1, keepdims=True)
    sims_other = idx.max_cosine_similarity(other)
    assert sims_other[0] < 0.5


@pytest.mark.skipif(
    not dedupe.hnswlib_available(),
    reason="hnswlib not installed; skip HNSW-specific tests",
)
def test_hnsw_index_empty_returns_zeros():
    """Querying an empty index must return zero similarity, not crash."""
    idx = dedupe._HNSWIndex(max_elements=10)
    rng = np.random.default_rng(seed=0)
    q = rng.normal(size=(3, dedupe.FEATURE_DIM)).astype(np.float32)
    sims = idx.max_cosine_similarity(q)
    assert sims.shape == (3,)
    assert np.all(sims == 0.0)


@pytest.mark.skipif(
    not dedupe.hnswlib_available(),
    reason="hnswlib not installed; skip HNSW-specific tests",
)
def test_hnsw_index_batch_query():
    """Batched query must return one similarity per query row."""
    idx = dedupe._HNSWIndex(max_elements=1000)
    rng = np.random.default_rng(seed=7)
    pool = rng.normal(size=(50, dedupe.FEATURE_DIM)).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    idx.add(pool)

    q = rng.normal(size=(8, dedupe.FEATURE_DIM)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims = idx.max_cosine_similarity(q)
    assert sims.shape == (8,)
    assert np.all((sims >= -1.0) & (sims <= 1.0))


def test_brute_force_max_sims_empty_pool():
    """Brute-force path with an empty pool must return zeros, not crash."""
    import torch

    features = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    pool = torch.empty((0, 16))
    sims = dedupe._brute_force_max_sims(features, pool)
    assert sims.shape == (4,)
    assert torch.all(sims == 0)


def test_brute_force_max_sims_identical_vector():
    """An identical vector in the pool must produce sim ≈ 1.0."""
    import torch

    v = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    pool = v.clone()
    sims = dedupe._brute_force_max_sims(v, pool)
    assert sims.shape == (1,)
    assert float(sims[0]) > 0.99


def _sine_tile(freq: float, amp: float = 60.0) -> np.ndarray:
    """Deterministic structured tile; distinct freqs are far apart in
    ResNet feature space, keeping dedupe tests off the 0.96 boundary."""
    x = np.arange(64)
    tile = np.clip(128 + amp * np.sin(x * freq)[None, :], 0, 255).astype(np.uint8)
    return np.repeat(tile, 64, axis=0)


def test_run_dedupe_trim_bottom_drops_lowest_scoring(tmp_path):
    """``trim_bottom`` must drop the lowest-complexity percentile of the
    surviving tiles, using the scores CSV, before similarity auditing.

    BHI finding: removing only the worst tiles beats keeping only the
    best; this is the percentile prune for it.
    """
    import csv as _csv

    import cv2

    csv_path = tmp_path / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            ["tile_name", "complexity_score", "score_calibrated", "calibration_outlier", "source", "source_image"]
        )
        for i in range(10):
            score = i / 10
            w.writerow([f"t{i}.png", f"{score}", f"{score}", "0", "", ""])

    # Structured, mutually distinct tiles: only the trim rule decides
    for i in range(10):
        cv2.imwrite(str(tmp_path / f"t{i}.png"), _sine_tile(0.5 + 0.1 * i))

    kept, removed = dedupe.run_dedupe(
        input_dir=tmp_path,
        threshold=0.9999,
        batch_size=32,
        device_preference="cpu",
        scores_csv=csv_path,
        trim_bottom=20,
    )
    assert (kept, removed) == (8, 2)
    # The two lowest-complexity tiles (score 0.0 and 0.1) are the ones trimmed
    assert (tmp_path / "redundant" / "t0.png").exists()
    assert (tmp_path / "redundant" / "t1.png").exists()
    assert not (tmp_path / "t0.png").exists()


def test_run_dedupe_flags_duplicates_inside_same_batch(tmp_path):
    """Two byte-identical tiles landing in one batch must collapse to one keeper.

    Regression: max similarities were computed once per batch *before* the
    add-loop, so duplicates processed together in the same batch never saw
    each other and both survived. Comparisons must stream against the pool
    as it grows.
    """
    import cv2

    base = _sine_tile(0.5)
    cv2.imwrite(str(tmp_path / "a.png"), base)
    cv2.imwrite(str(tmp_path / "b.png"), base)  # byte-identical
    cv2.imwrite(str(tmp_path / "c.png"), _sine_tile(1.1))
    cv2.imwrite(str(tmp_path / "d.png"), _sine_tile(1.7))

    kept, removed = dedupe.run_dedupe(
        input_dir=tmp_path, threshold=0.9999, batch_size=32, device_preference="cpu"
    )
    assert (kept, removed) == (3, 1)
    assert (tmp_path / "redundant" / "b.png").exists()


def test_run_dedupe_seed_pool_flags_preexisting_content(tmp_path):
    """A tile whose ResNet-18 embedding matches the seeded pool (e.g.
    v2-corpus content) must be dropped from this run's kept set, even
    though this run never saw it in its own batches."""
    import cv2
    import torch
    from PIL import Image

    base = _sine_tile(0.5)
    cv2.imwrite(str(tmp_path / "a.png"), base)
    cv2.imwrite(str(tmp_path / "b.png"), _sine_tile(1.3))

    fe = dedupe._get_feature_extractor("cpu")
    prep = dedupe._preload_transform()
    with Image.open(tmp_path / "a.png") as im:
        t = prep(im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        seed = torch.nn.functional.normalize(fe(t).squeeze().float()[None], dim=1)

    kept, removed = dedupe.run_dedupe(
        input_dir=tmp_path,
        threshold=0.9999,
        batch_size=8,
        device_preference="cpu",
        seed_embeddings=seed,
    )
    assert (kept, removed) == (1, 1)
    assert (tmp_path / "redundant" / "a.png").exists()
    assert (tmp_path / "b.png").exists()


@pytest.mark.skipif(
    not dedupe.hnswlib_available(),
    reason="hnswlib not installed; skip HNSW-specific tests",
)
def test_hnsw_index_grow_preserves_vectors():
    """T3: After an index grows, vectors added before the grow are still found.

    Regression test: the old _grow implementation rebuilt the index from scratch,
    losing all previously added vectors. This test verifies the fixed version
    uses resize_index() which preserves all items.
    """
    idx = dedupe._HNSWIndex(max_elements=4)
    rng = np.random.default_rng(seed=42)

    # Add 4 vectors - this is within initial capacity
    first_batch = rng.normal(size=(4, dedupe.FEATURE_DIM)).astype(np.float32)
    first_batch /= np.linalg.norm(first_batch, axis=1, keepdims=True)
    idx.add(first_batch)

    # Save one of the first vectors for later query
    query_vec = first_batch[0:1].copy()

    # Add more vectors to trigger a grow
    second_batch = rng.normal(size=(4, dedupe.FEATURE_DIM)).astype(np.float32)
    second_batch /= np.linalg.norm(second_batch, axis=1, keepdims=True)
    idx.add(second_batch)

    # Query with the first vector - should still find it with high similarity
    sims = idx.max_cosine_similarity(query_vec)
    assert sims[0] > 0.99, f"Expected high similarity after grow, got {sims[0]}"
