"""End-to-end pipeline test on the 7-image calibration folder.

Exercises ingest → dedupe → finalize on real images and asserts the
output structure is what downstream SISR training scripts expect.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lucid.config import LucidConfig
from lucid.pipeline import run_pipeline

CALIB_DIR = Path("/home/phhofm/Documents/filteing_test/input")


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run the full pipeline once per test module to keep wall time down."""
    out = tmp_path_factory.mktemp("pipeline_out")
    workspace = out / "workspace"
    LucidConfig.reset()
    stats = run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=workspace,
        model_path=None,
        tile_size=128,
        threshold=None,  # auto
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
    return out, workspace, stats


def test_pipeline_produces_tiles_folder(pipeline_output):
    out, _, _ = pipeline_output
    tiles_dir = out / "tiles"
    assert tiles_dir.exists()
    tiles = list(tiles_dir.glob("*.png"))
    assert len(tiles) > 0, "Expected at least one final tile"
    # Tile filenames are 5-digit sequential
    for t in tiles[:5]:
        assert t.name[:5].isdigit(), f"Bad tile name: {t.name}"


def test_pipeline_writes_master_lineage(pipeline_output):
    out, _, _ = pipeline_output
    lineage = out / "master_lineage.csv"
    assert lineage.exists()
    rows = list(csv.DictReader(open(lineage)))
    assert len(rows) > 0
    # All columns from Tier 3.4 must be present
    expected = {
        "id", "final_name", "source_dataset", "original_name", "original_path",
        "source_image_min_complexity", "source_image_mean_complexity",
        "source_image_max_complexity", "source_image_tiles_kept",
    }
    assert expected.issubset(rows[0].keys())


def test_pipeline_lineage_stats_are_populated(pipeline_output):
    out, _, _ = pipeline_output
    rows = list(csv.DictReader(open(out / "master_lineage.csv")))
    # At least one row should have a non-empty mean
    populated = [r for r in rows if r["source_image_mean_complexity"]]
    assert len(populated) > 0


def test_pipeline_writes_scores_csv(pipeline_output):
    _, workspace, _ = pipeline_output
    csv_path = workspace / "scores" / "ingest_scores.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        header = f.readline().strip()
    # Tier 3.5 added calibration_outlier column
    assert "calibration_outlier" in header
    assert "score_calibrated" in header
    assert "complexity_score" in header


def test_pipeline_writes_histogram(pipeline_output):
    _, workspace, _ = pipeline_output
    hist = workspace / "scores" / "histogram.txt"
    assert hist.exists()
    body = hist.read_text()
    assert "complexity score histogram" in body
    assert "bin_start" in body


def test_pipeline_writes_threshold_audit(pipeline_output):
    _, workspace, _ = pipeline_output
    audit = workspace / "scored" / "threshold_used.txt"
    assert audit.exists()
    body = audit.read_text()
    assert "tile_size" in body
    assert "threshold_used" in body
    assert "suggested_threshold" in body


def test_pipeline_writes_calibration_outliers(pipeline_output):
    _, workspace, _ = pipeline_output
    out = workspace / "scores" / "calibration_outliers.csv"
    assert out.exists()


def test_pipeline_summary_stats(pipeline_output):
    _, _, stats = pipeline_output
    # 7 source images in the calibration folder
    assert stats["sources_processed"] == 7
    # Some tiles scored (signal filter + ICNet)
    assert stats["tiles_scored"] > 0
    # At least one tile passed both stages
    assert stats["tiles_kept"] > 0


def test_pipeline_cleans_up_intermediates(pipeline_output):
    """Tier 1.1: raw_tiles/ and scored/redundant/ must be removed."""
    _, workspace, _ = pipeline_output
    assert not (workspace / "raw_tiles").exists()
    assert not (workspace / "scored" / "redundant").exists()


def test_pipeline_no_duplicate_final_tiles(pipeline_output):
    out, _, _ = pipeline_output
    tiles = list((out / "tiles").glob("*.png"))
    names = [t.name for t in tiles]
    assert len(names) == len(set(names)), "Duplicate final tile names"


def test_pipeline_with_multiscale_produces_more_tiles(tmp_path):
    """Tier default: multiscale=ON should produce >= as many final tiles
    as multiscale=OFF for the same threshold, on the same inputs."""
    LucidConfig.reset()
    out = tmp_path / "ms"
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=out / "workspace",
        model_path=None,
        tile_size=128,
        threshold=0.45,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=True,
        io_throttle_sec=0.0,
        strict=False,
        resume=False,
        stage="all",
        dry_run=False,
        cleanup=True,
    )
    out_single = tmp_path / "single"
    LucidConfig.reset()
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out_single,
        workspace=out_single / "workspace",
        model_path=None,
        tile_size=128,
        threshold=0.45,
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
        cleanup=True,
    )
    n_ms = len(list((out / "tiles").glob("*.png")))
    n_single = len(list((out_single / "tiles").glob("*.png")))
    assert n_ms >= n_single, f"Multiscale {n_ms} < single-scale {n_single}"


def test_threshold_auto_adjust_per_tile_size(tmp_path):
    """When no --threshold is passed, the orchestrator should use the
    calibration-suggested value, not 0.45."""
    LucidConfig.reset()
    out = tmp_path / "auto"
    run_pipeline(
        input_dir=CALIB_DIR,
        output_dir=out,
        workspace=out / "workspace",
        model_path=None,
        tile_size=128,
        threshold=None,
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
    audit = out / "workspace" / "scored" / "threshold_used.txt"
    assert audit.exists()
    body = audit.read_text()
    # 128px auto-threshold from shipped calibration is ~0.4724
    # (6 decimal places of precision in the audit file).
    assert "threshold_used: 0.472" in body, body
