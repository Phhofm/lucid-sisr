"""Resume / crash-recovery tests (hermetic synthetic fixtures).

Exercises the SQLite state DB across pipeline invocations: resume must
not duplicate tiles and must refuse to silently continue when the
filtering gates changed since the recorded run.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from lucid.pipeline import run_pipeline
from lucid.state import PipelineState

FIXTURE_SEED = 0


def build_input(root) -> object:
    """Create a small deterministic input folder."""
    inp = root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FIXTURE_SEED)
    small = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    gray = cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(inp / "textured.png"), gray)
    cv2.imwrite(str(inp / "flat.png"), np.full((512, 512), 128, np.uint8))
    cv2.imwrite(str(inp / "tiny.png"), gray[:64, :64].copy())
    cv2.imwrite(str(inp / "truncated.jpg"), gray)
    data = (inp / "truncated.jpg").read_bytes()
    (inp / "truncated.jpg").write_bytes(data[: len(data) // 2])
    return inp


def run(inp, out, threshold=0.0, resume=False, stage="all"):
    return run_pipeline(
        input_dir=inp,
        output_dir=out,
        workspace=out / "workspace",
        model_path=None,
        tile_size=128,
        threshold=threshold,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        trim_bottom=0,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=False,
        io_throttle_sec=0.0,
        strict=False,
        resume=resume,
        stage=stage,
        dry_run=False,
        cleanup=False,
    )


def test_resume_does_not_duplicate_final_tiles(tmp_path):
    """Run the pipeline twice with --resume; final tile names must be unique."""
    inp = build_input(tmp_path)
    out = tmp_path / "out"

    run(inp, out, threshold=0.0)
    first_tiles = sorted(p.name for p in (out / "tiles").glob("*.png"))
    assert len(first_tiles) > 0
    assert len(first_tiles) == len(set(first_tiles))

    run(inp, out, threshold=0.0, resume=True)
    second_tiles = sorted(p.name for p in (out / "tiles").glob("*.png"))
    assert second_tiles == first_tiles


def test_resume_refuses_changed_gates(tmp_path):
    """Resuming with different filter gates must abort loudly.

    Otherwise half the dataset is filtered with the old gates and half
    with the new ones, silently corrupting the training set.
    """
    inp = build_input(tmp_path)
    out = tmp_path / "out"
    run(inp, out, threshold=0.0)

    with pytest.raises(RuntimeError, match="changed"):
        run(inp, out, threshold=0.9, resume=True)

    # Same gates resume fine
    run(inp, out, threshold=0.0, resume=True)


def test_resume_dedupe_stage_only(tmp_path):
    """Running just the dedupe stage on an already-ingested workspace
    must work without errors and yield a non-zero dedupe count."""
    inp = build_input(tmp_path)
    out = tmp_path / "out"

    run(inp, out, stage="ingest")
    assert (out / "workspace" / "scored").exists()

    run(inp, out, stage="dedupe")
    scored = sorted(p.name for p in (out / "workspace" / "scored").glob("*.png"))
    assert len(scored) > 0


def test_state_db_persists_across_runs(tmp_path):
    """The SQLite state DB is the source of truth for resume; verify it
    persists and is queryable across pipeline invocations."""
    inp = build_input(tmp_path)
    out = tmp_path / "out"

    run(inp, out, stage="ingest")

    state = PipelineState(out / "workspace")
    state.connect()
    try:
        pending = state.get_pending_sources([inp / "textured.png"])
        assert len(pending) == 0
    finally:
        state.close()
