"""Regression tests for CLIP-IQA gate per-tile score indexing (B1)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from lucid.core.filtering import _score_batch, _ScoringContext


def test_scoring_context_stores_batch_size_and_io_throttle():
    """T2: Verify batch_size and io_throttle are stored in the context."""
    ctx = _ScoringContext(
        model=torch.nn.Identity(),
        device="cpu",
        use_half=False,
        mean=torch.zeros(1, 3, 1, 1),
        std=torch.ones(1, 3, 1, 1),
        threshold=0.5,
        tile_size=512,
        calibrate=False,
        batch_size=7,
        io_throttle=0.005,
    )
    assert ctx.batch_size == 7, f"Expected batch_size=7, got {ctx.batch_size}"
    assert ctx.io_throttle == 0.005, f"Expected io_throttle=0.005, got {ctx.io_throttle}"


def test_scoring_context_defaults():
    """T2: Verify default values for batch_size and io_throttle."""
    ctx = _ScoringContext(
        model=torch.nn.Identity(),
        device="cpu",
        use_half=False,
        mean=torch.zeros(1, 3, 1, 1),
        std=torch.ones(1, 3, 1, 1),
        threshold=0.5,
        tile_size=512,
        calibrate=False,
    )
    assert ctx.batch_size == 32, f"Expected default batch_size=32, got {ctx.batch_size}"
    assert ctx.io_throttle == 0.0, f"Expected default io_throttle=0.0, got {ctx.io_throttle}"


def test_clipiqa_index_uses_loop_position_not_cumulative_counter():
    """B1 regression: CLIP-IQA score must be indexed by loop position, not
    tiles_passed + tiles_clipped, which caused the wrong score to be read
    after any tile was rejected by CLIP-IQA."""
    # Create three 8x8 RGB tiles with Path objects
    tiles = [
        (Path(f"tile_{i}.png"), np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        for i in range(3)
    ]

    # We'll create a context and monkeypatch the model forward to return controlled scores
    icnet_scores = [0.1, 0.9, 0.9]
    clipiqa_scores = [0.9, 0.9, 0.1]

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor(icnet_scores).to(x.device), torch.zeros(1)

    class FakeCLIPModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor(clipiqa_scores).to(x.device)

    fake_model = FakeModel()
    fake_clipiqa = FakeCLIPModel()

    ctx = _ScoringContext(
        model=fake_model,
        device="cpu",
        use_half=False,
        mean=torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        std=torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        threshold=0.5,
        tile_size=512,  # Use 512 for the model size
        calibrate=False,
        clipiqa_model=fake_clipiqa,
        clipiqa_threshold=0.5,
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)

    try:
        written_names = []

        class MockWriter:
            def write(self, name, arr):
                written_names.append(name)

        ctx.writer = MockWriter()

        passed, _ = _score_batch(ctx, tiles, csv_path)

        # Tile 0: ICNet=0.1 (FAIL threshold 0.5) -> rejected
        # Tile 1: ICNet=0.9 (PASS), CLIP-IQA=0.9 (PASS threshold 0.5) -> passes
        # Tile 2: ICNet=0.9 (PASS), CLIP-IQA=0.1 (FAIL threshold 0.5) -> rejected
        # So exactly 1 tile should pass
        assert passed == 1, f"Expected 1 tile passed, got {passed}"
        assert written_names == ["tile_1.png"]
    finally:
        csv_path.unlink(missing_ok=True)
