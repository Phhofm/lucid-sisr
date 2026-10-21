"""End-to-end pipeline test on synthetic fixtures.

Runs ingest → dedupe → finalize on generated images so the suite is
hermetic (no external folder): a textured source, a byte-identical
duplicate of it, a flat image, an oversized-small image, and a
truncated JPEG.
"""
from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from lucid.pipeline import run_pipeline

# Deterministic fixture: 128px random source upscaled 4x passes all six
# signal gates on every one of its 16 tiles (lap_var ~800, blockiness
# ~1.3, entropy ~7.9 — verified against THRESHOLDS).
FIXTURE_SEED = 0


def _textured_gray() -> np.ndarray:
    rng = np.random.default_rng(FIXTURE_SEED)
    small = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)


@pytest.fixture
def input_dir(tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    gray = _textured_gray()
    cv2.imwrite(str(inp / "aaa_dup.png"), gray)  # sorts before textured.png
    cv2.imwrite(str(inp / "textured.png"), gray)  # byte-identical dedupe target
    cv2.imwrite(str(inp / "flat.png"), np.full((512, 512), 128, np.uint8))
    cv2.imwrite(str(inp / "tiny.png"), _textured_gray()[:64, :64].copy())
    cv2.imwrite(str(inp / "truncated.jpg"), gray)
    data = (inp / "truncated.jpg").read_bytes()
    (inp / "truncated.jpg").write_bytes(data[: len(data) // 2])
    return inp


def run(input_dir, output_dir, threshold=0.0, multiscale=False, cleanup=False):
    """Run the full pipeline on the e2e fixtures. threshold=0.0 makes the
    scoring gate pass everything the signal filter kept; clipiqa stays off
    so the test is independent of pyiqa."""
    return run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        workspace=output_dir / "workspace",
        model_path=None,
        tile_size=128,
        threshold=threshold,
        clipiqa_threshold=0.0,
        dedupe_threshold=0.96,
        batch_size=0,
        workers=2,
        max_vram_gb=0.0,
        multiscale=multiscale,
        io_throttle_sec=0.0,
        strict=False,
        resume=False,
        stage="all",
        dry_run=False,
        cleanup=cleanup,
    )


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    inp = tmp_path_factory.mktemp("input_root")
    inp = inp / "input"
    inp.mkdir()
    gray = _textured_gray()
    cv2.imwrite(str(inp / "aaa_dup.png"), gray)
    cv2.imwrite(str(inp / "textured.png"), gray)
    cv2.imwrite(str(inp / "flat.png"), np.full((512, 512), 128, np.uint8))
    cv2.imwrite(str(inp / "tiny.png"), _textured_gray()[:64, :64].copy())
    cv2.imwrite(str(inp / "truncated.jpg"), gray)
    data = (inp / "truncated.jpg").read_bytes()
    (inp / "truncated.jpg").write_bytes(data[: len(data) // 2])

    out = tmp_path_factory.mktemp("pipeline_out")
    stats = run(inp, out)
    return out, out / "workspace", stats


def test_pipeline_produces_tiles_folder(pipeline_output):
    out, _, _ = pipeline_output
    tiles_dir = out / "tiles"
    assert tiles_dir.exists()
    tiles = list(tiles_dir.glob("*.png"))
    # 32 identical-content tiles in; dedupe keeps at most one source's set
    assert 0 < len(tiles) <= 16
    for t in tiles[:5]:
        assert t.name[:5].isdigit(), f"Bad tile name: {t.name}"


def test_pipeline_dedupes_identical_sources(pipeline_output):
    """The two byte-identical sources must collapse to one in the final set."""
    out, _, _ = pipeline_output
    rows = list(csv.DictReader(open(out / "master_lineage.csv")))
    # original_name is the surviving tile's pre-rename name; its prefix
    # carries the source stem. The duplicate source must never survive.
    stems = {r["original_name"].split("_s")[0] for r in rows}
    assert stems == {"aaa_dup"}, stems


def test_pipeline_writes_master_lineage(pipeline_output):
    out, _, _ = pipeline_output
    lineage = out / "master_lineage.csv"
    assert lineage.exists()
    rows = list(csv.DictReader(open(lineage)))
    assert len(rows) > 0
    expected = {
        "id", "final_name", "source_dataset", "original_name", "original_path",
        "source_image_min_complexity", "source_image_mean_complexity",
        "source_image_max_complexity", "source_image_tiles_kept",
    }
    assert expected.issubset(rows[0].keys())


def test_pipeline_writes_scores_csv(pipeline_output):
    _, workspace, _ = pipeline_output
    csv_path = workspace / "scores" / "ingest_scores.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        header = f.readline().strip()
    assert "calibration_outlier" in header
    assert "score_calibrated" in header
    assert "complexity_score" in header


def test_pipeline_summary_stats(pipeline_output):
    _, _, stats = pipeline_output
    # 5 sources: aaa_dup, textured, flat, tiny, truncated
    assert stats["sources_processed"] == 5
    # Only the two identical 512px sources produce tiles: 16 each. The
    # truncated JPEG and the two gate-rejected sources contribute nothing.
    assert stats["tiles_scored"] == 32
    assert stats["tiles_passed"] == 32  # threshold=0.0 passes all signal-passing tiles
    assert 0 < stats["tiles_kept"] <= 16  # dup pair collapses; intra-source near-dupes may also collapse


def test_pipeline_cleans_up_intermediates(pipeline_output):
    _, workspace, _ = pipeline_output
    assert not (workspace / "raw_tiles").exists()
    assert not (workspace / "scored" / "redundant").exists()


def test_pipeline_no_duplicate_final_tiles(pipeline_output):
    out, _, _ = pipeline_output
    tiles = [t.name for t in (out / "tiles").glob("*.png")]
    assert len(tiles) == len(set(tiles)), "Duplicate final tile names"


def test_pipeline_auto_threshold_from_calibration(tmp_path):
    """With no --threshold, the orchestrator uses the calibration-suggested
    value for the tile size (0.472 for 128px), not 0.45."""
    inp = tmp_path / "input"
    inp.mkdir()
    cv2.imwrite(str(inp / "textured.png"), _textured_gray())
    out = tmp_path / "auto"
    run(inp, out, threshold=None, cleanup=False)
    audit = out / "workspace" / "scored" / "threshold_used.txt"
    assert audit.exists()
    assert "threshold_used: 0.472" in audit.read_text(), audit.read_text()


def test_pipeline_with_multiscale_produces_at_least_as_many_tiles(tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    cv2.imwrite(str(inp / "textured.png"), _textured_gray())
    results = {}
    for tag, ms in (("ms", True), ("single", False)):
        out = tmp_path / tag
        run(inp, out, multiscale=ms, cleanup=True)
        results[tag] = len(list((out / "tiles").glob("*.png")))
    assert results["ms"] >= results["single"]
