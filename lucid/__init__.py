"""LUCID - Mathematical Integrity for SISR Datasets.

A dataset filtering pipeline that prioritizes technical signal quality
over human-perception metrics for super-resolution training.
"""

__version__ = "3.1.0"

# Lazy imports: only load torch-dependent modules when actually needed.
# This allows `import lucid` to succeed in environments without torch
# (e.g., CI runners that only run non-torch tests).


def __getattr__(name: str):
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    if name == "ensure_model_downloaded":
        from .models import ensure_model_downloaded

        return ensure_model_downloaded
    if name == "get_default_model_path":
        from .models import get_default_model_path

        return get_default_model_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + [
        "run_pipeline",
        "ensure_model_downloaded",
        "get_default_model_path",
    ]
