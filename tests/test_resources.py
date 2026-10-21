"""Tests for the disk-space guard estimate."""
import pytest

import lucid.pipeline as pipeline
from lucid.pipeline import run_pipeline
from lucid.resources import estimate_disk_need_gb


def test_estimate_scales_with_input_bytes():
    # README contract: ~2x input size (staging + scored + tiles copies)
    assert estimate_disk_need_gb(1024**3 * 10) == pytest.approx(20.0)
    assert estimate_disk_need_gb(0) == 0.0


def test_disk_guard_uses_real_input_bytes(tmp_path, monkeypatch):
    """The guard must derive need from actual bytes, not a hardcoded
    25-tiles-per-image multiplier: a folder of pre-tiled tiles previously
    estimated ~2.2 TB (100_866 inputs x 25 tiles) with 340 GB free and
    falsely aborted."""

    inp = tmp_path / "in"
    inp.mkdir()
    for i in range(4):
        (inp / f"t{i}.png").write_bytes(b"x" * (1024 * 1024))  # 4 MB total

    free_calls: list = []

    def fake_free(_path):
        free_calls.append(_path)
        return 0.001  # 1 MB free < the 8 MB need: force the abort branch

    monkeypatch.setattr(pipeline, "get_disk_free_gb", fake_free)
    with pytest.raises(RuntimeError, match="Estimated need"):
        run_pipeline(
            input_dir=inp,
            output_dir=tmp_path / "out",
            workspace=tmp_path / "ws",
            model_path=None,
            tile_size=512,
            threshold=0.0,
            clipiqa_threshold=0.0,
            dedupe_threshold=0.96,
            trim_bottom=0,
            batch_size=0,
            workers=1,
            max_vram_gb=0.0,
            multiscale=False,
            io_throttle_sec=0.0,
            strict=False,
            resume=False,
            stage="all",
            dry_run=False,
            cleanup=False,
        )
    assert len(free_calls) == 1
