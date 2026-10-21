"""Vulture whitelist — names that look unused to static analysis
but are called by the Python runtime, a framework protocol, or a test.

Vulture reads this file as a regular Python module. Any top-level
``from x import y`` statement here will silence the "y is unused"
finding on the source file. Run as:
    vulture lucid/ .vulture-whitelist.py --min-confidence 60
"""
from __future__ import annotations

from lucid.core import __dir__ as _core_dir  # noqa: F401

# --- Python language protocols (PEP 562, dir(), etc.) ---
from lucid.core import __getattr__ as _core_getattr  # noqa: F401
from lucid.core.dedupe import hnswlib_available  # noqa: F401

# --- Used by tests, vulture can't see cross-file into tests/ ---
from lucid.core.hashing import (
    filter_exact_duplicates,  # noqa: F401
    tile_digest,  # noqa: F401
)
from lucid.models.icnet import ConvBlock as ConvBlock  # noqa: F401

# --- PyTorch / nn.Module: `forward` is dispatched via __call__ ---
# (vulture can't see nn.Module.__call__ → forward dispatch)
from lucid.models.icnet import ICNet as _ICNet  # noqa: F401
from lucid.models.icnet import ResNet as ResNet  # noqa: F401
from lucid.models.icnet import _Attention as _Attention  # noqa: F401
from lucid.models.icnet import slam as slam  # noqa: F401
from lucid.resources.icnet_calibration import (
    is_supported_size,  # noqa: F401
    reload_calibration,  # noqa: F401
)
from lucid.state import (
    get_completed_sources,  # noqa: F401
    is_source_done,  # noqa: F401
)
