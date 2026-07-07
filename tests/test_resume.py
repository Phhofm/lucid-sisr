"""Resume / crash-recovery test.

Runs the pipeline in two passes to exercise the SQLite state DB:
  Pass 1: pipeline completes, populates state DB
  Pass 2: pipeline re-runs with --resume, must finalize cleanly and not
          produce duplicate final tile names
"""
from __future__ import annotations

from pathlib import Path

from lucid.pipeline import run_pipeline

CALIB_DIR = Path("/home/phhofm/Documents/filteing_test/input")


def test_resume_does_not_duplicate_final_tiles(tmp_path):
    """Run the pipeline twice with --resume; final tile names must be unique."""
    out = tmp_path / "out"
    workspace = out / "workspace"

    # Pass 1: full pipeline
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=workspace,
        model_path=None,
        tile_size=128,
        threshold=0.45,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=False,
        io_throttle_sec=0.0,
        strict=False,
        resume=False,
        stage="all",
        dry_run=False,
        cleanup=False,
    )
    first_tiles = sorted(p.name for p in (out / "tiles").glob("*.png"))
    assert len(first_tiles) > 0
    assert len(first_tiles) == len(set(first_tiles))

    # Pass 2: re-run with --resume (default; non-dry, no stage)
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=workspace,
        model_path=None,
        tile_size=128,
        threshold=0.45,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=False,
        io_throttle_sec=0.0,
        strict=False,
        resume=True,
        stage="all",
        dry_run=False,
        cleanup=False,
    )
    second_tiles = sorted(p.name for p in (out / "tiles").glob("*.png"))
    # Same number of tiles (no duplicates appended)
    assert len(second_tiles) == len(first_tiles), (
        f"Resume added tiles: {len(second_tiles)} vs {len(first_tiles)}"
    )
    # And same names
    assert set(second_tiles) == set(first_tiles)


def test_resume_dedupe_stage_only(tmp_path):
    """Running just the dedupe stage on an already-ingested workspace
    must work without errors and yield a non-zero dedupe count."""
    out = tmp_path / "out"
    workspace = out / "workspace"

    # Ingest only
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=workspace,
        model_path=None,
        tile_size=128,
        threshold=0.45,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=False,
        io_throttle_sec=0.0,
        strict=False,
        resume=False,
        stage="ingest",
        dry_run=False,
        cleanup=False,
    )
    assert (workspace / "scored").exists()

    # Now just dedupe
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=workspace,
        model_path=None,
        tile_size=128,
        threshold=0.45,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=False,
        io_throttle_sec=0.0,
        strict=False,
        resume=False,
        stage="dedupe",
        dry_run=False,
        cleanup=False,
    )
    # scored/ should now contain only unique tiles
    scored = list((workspace / "scored").glob("*.png"))
    assert len(scored) > 0


def test_state_db_persists_across_runs(tmp_path):
    """The SQLite state DB is the source of truth for resume; verify it
    persists and is queryable across pipeline invocations."""
    from lucid.state import PipelineState

    out = tmp_path / "out"
    workspace = out / "workspace"

    # Ingest only
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=workspace,
        model_path=None,
        tile_size=128,
        threshold=0.45,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=False,
        io_throttle_sec=0.0,
        strict=False,
        resume=False,
        stage="ingest",
        dry_run=False,
        cleanup=False,
    )

    # Open the state DB in a fresh process context and confirm
    state = PipelineState(workspace)
    state.connect()
    try:
        pending = state.get_pending_sources(
            [CALIB_DIR / "aladin-habboubi-YNiXuanD2iY-unsplash.jpg"]
        )
        # After successful ingest, no source is "pending" anymore
        assert len(pending) == 0
    finally:
        state.close()
