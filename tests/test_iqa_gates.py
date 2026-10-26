"""Tests for the --iqa-gate perceptual tier (pyiqa-backed, cascade).

Stubs replace pyiqa models (the integration works without pyiqa
installed); the production loader is exercised separately.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from lucid.core.filtering import _score_batch, _ScoringContext


def _make_ctx(iqa_gates, clipiqa_model=None, clipiqa_threshold=0.0):
    return _ScoringContext(
        model=torch.nn.Identity(),
        device="cpu",
        use_half=False,
        mean=torch.zeros(1, 3, 1, 1),
        std=torch.ones(1, 3, 1, 1),
        threshold=0.5,  # placeholder: icnet scores stubbed below
        tile_size=512,
        calibrate=False,
        clipiqa_model=clipiqa_model,
        clipiqa_threshold=clipiqa_threshold,
        iqa_gates=iqa_gates,
    )


class _StubMetric:
    """pyiqa-metric duck-type: callable on a batch tensor, results tensor."""

    def __init__(self, scores_fn, lower_better=False):
        self.scores_fn = scores_fn
        self.lower_better = lower_better
        self.calls = []

    def __call__(self, x):
        self.calls.append(x.shape[0])
        return torch.tensor(self.scores_fn(len(x)), dtype=torch.float32)


def _tiles(n: int = 3):
    return [
        (Path(f"tile_{i}.png"), np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        for i in range(n)
    ]


def _run_batch(ctx, n=3):
    tiles = _tiles(n)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
    try:
        import io, csv as _csv

        written = []

        class MockWriter:
            def write(self, name, arr):
                written.append(name)

        ctx.writer = MockWriter()
        scored, record = _score_batch(ctx, tiles, csv_path)
        return scored, record, n
    finally:
        csv_path.unlink(missing_ok=True)


def test_iqa_gate_rejects_low_scores_records_pass_flags():
    scores = [0.2, 0.7, 0.9]  # tile 0 below floor, others above
    gates = [dict(name="topiq_nr", threshold=0.45, lower_better=False,
                  model=_StubMetric(lambda n: scores))]
    ctx = _make_ctx(gates)
    # icnet pass-all via identity model: threshold applies to identity output = input pixels?? —
    # identity output = the [0,255] tensor; icnet gate would fail everything.
    # Use threshold 0.0 = pass-all trick (>= 0 always true):
    ctx.threshold = -1.0
    passed, record, n = _run_batch(ctx, 3)
    assert record["passed"] == [False, True, True]
    assert passed == 2


def test_iqa_gate_cascade_skips_vlm_for_rejected_tiles():
    """A tile failing the previous gate must never reach the next (VLM) gate."""
    topiq_scores = [0.2, 0.7, 0.9]
    top = _StubMetric(lambda n: topiq_scores)
    # qrealign sees ONLY the topiq survivors (tiles 1,2) -> return their
    # scores: tile 1 fails the floor, tile 2 passes
    realign = _StubMetric(lambda n: [1.0, 0.3, 1.0][-n:])
    gates = [
        dict(name="topiq_nr", threshold=0.45, lower_better=False, model=top),
        dict(name="qrealign-mini", threshold=0.4, lower_better=False, model=realign),
    ]
    ctx = _make_ctx(gates)
    ctx.threshold = -1.0
    passed, record, n = _run_batch(ctx, 3)
    # topiq called once with all 3; realign called with only the 2 survivors
    assert top.calls == [3]
    assert realign.calls == [2], realign.calls
    assert record["passed"] == [False, False, True]
    assert passed == 1


def test_iqa_gate_lower_better_direction():
    """niqe-style metrics (lower_better) must pass on score <= threshold."""
    scores = [1.0, 9.0]  # tile 0 good (<= 5), tile 1 bad
    gates = [dict(name="niqe", threshold=5.0, lower_better=True,
                  model=_StubMetric(lambda n: scores))]
    ctx = _make_ctx(gates)
    ctx.threshold = -1.0
    passed, record, _ = _run_batch(ctx, 2)
    assert record["passed"] == [True, False]


def test_iqa_gate_threshold_zero_is_pass_through():
    """A gate with threshold 0 must never reject (disabled semantics)."""
    scores = [0.1, 0.1]
    gates = [dict(name="topiq_nr", threshold=0.0, lower_better=False,
                  model=_StubMetric(lambda n: scores))]
    ctx = _make_ctx(gates)
    ctx.threshold = -1.0
    passed, record, _ = _run_batch(ctx, 2)
    assert record["passed"] == [True, True]


def test_gate_list_builder_merges_clipiqa_legacy():
    from lucid.core.filtering import build_gate_list
    gates = build_gate_list(iqa_gates=[("niqe", 5.0)], clipiqa_threshold=0.3)
    assert gates[0] == ("clipiqa", 0.3)
    assert gates[1] == ("niqe", 5.0)
    assert build_gate_list(None, 0.0) == []
