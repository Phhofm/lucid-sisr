"""Unit tests for PipelineState (resume tracking)."""
import tempfile
from pathlib import Path

from lucid.state import PipelineState


def test_get_pending_sources_empty():
    """Empty source list should return empty pending list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        state = PipelineState(ws)
        state.connect()
        try:
            pending = state.get_pending_sources([])
            assert pending == []
        finally:
            state.close()


def test_get_pending_sources_respects_completed():
    """Sources marked completed should not be returned as pending."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        state = PipelineState(ws)
        state.connect()
        try:
            src1 = Path("/fake/src1.png")
            src2 = Path("/fake/src2.png")
            src3 = Path("/fake/src3.png")

            state.register_source(src1)
            state.register_source(src2)
            state.register_source(src3)

            state.mark_source_status(src1, "scored")

            pending = state.get_pending_sources([src1, src2, src3])
            assert len(pending) == 2
            assert src1 not in pending
            assert src2 in pending
            assert src3 in pending
        finally:
            state.close()


if __name__ == "__main__":
    test_get_pending_sources_empty()
    test_get_pending_sources_respects_completed()
    print("All state tests passed.")
