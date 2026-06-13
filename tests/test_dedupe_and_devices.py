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
