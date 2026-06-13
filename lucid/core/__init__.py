"""LUCID core pipeline modules.

Top-level imports of the heavy submodules (``dedupe``, ``filtering``,
``finalize``) are deferred to attribute access so that lightweight modules
(``hashing``, ``metrics``, ``validation``) can be imported in environments
without ``torch`` installed (e.g. minimal test venvs, docs builds, type
checkers that just need the function signatures).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .metrics import THRESHOLDS, measure_all
from .validation import is_safe_to_process, load_image_as_rgb, validate_image

if TYPE_CHECKING:
    # Only resolved for type checkers; the runtime __getattr__ below is what
    # actually loads these.
    from .dedupe import run_dedupe
    from .filtering import run_ingest
    from .finalize import run_finalize

__all__ = [
    "measure_all",
    "THRESHOLDS",
    "validate_image",
    "is_safe_to_process",
    "load_image_as_rgb",
    "run_ingest",
    "run_dedupe",
    "run_finalize",
]

# Map of lazily-loaded submodule exports. Using __getattr__ (PEP 562) so
# ``from lucid.core import run_ingest`` still works as long as torch is
# installed, but importing the package itself never requires it.
_LAZY_EXPORTS = {
    "run_dedupe": "lucid.core.dedupe",
    "run_ingest": "lucid.core.filtering",
    "run_finalize": "lucid.core.finalize",
}


def __getattr__(name: str):  # noqa: D401  (PEP 562 protocol)
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module 'lucid.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
