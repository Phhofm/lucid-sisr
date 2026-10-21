"""Resource detection helpers (VRAM, disk space, drive type, optimal batch sizes)."""
from __future__ import annotations

import logging
from pathlib import Path

import psutil
import torch

logger = logging.getLogger(__name__)


def get_available_vram_gb() -> float:
    """Return available VRAM in GB across the active accelerator, or 0.0 if CPU-only.

    Works for CUDA, Intel XPU, and Apple MPS (MPS reports unified memory, so
    the result is approximate). For HIP/ROCm the value reflects the active
    CUDA-compat device since torch 2.4 unifies the API.
    """
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            return (total - reserved) / (1024 ** 3)
        if _xpu_available():
            # XPU exposes a similar API as of IPEX 2.3
            total = torch.xpu.get_device_properties(0).total_memory
            return total / (1024 ** 3)
    except Exception as e:
        logger.warning("Could not query accelerator memory: %s", e)
    return 0.0


def _xpu_available() -> bool:
    """Return ``True`` if Intel XPU (IPEX) acceleration is available."""
    return (
        hasattr(torch, "xpu")
        and callable(getattr(torch.xpu, "is_available", None))
        and torch.xpu.is_available()
    )


def _mps_available() -> bool:
    """Return ``True`` if Apple Silicon MPS is available."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _directml_available() -> bool:
    """Return ``True`` if Windows DirectML is importable and reports a device.

    DirectML is only useful on Windows and requires the optional
    ``torch-directml`` package; on other platforms or without the package
    this returns ``False`` quickly.
    """
    try:
        import torch_directml  # type: ignore[import-not-found]

        return torch_directml.device_count() > 0
    except Exception:
        return False


def detect_accelerator() -> str:
    """Return a human-readable label for the active compute device.

    Possible values: ``"CUDA"``, ``"ROCm"``, ``"XPU"``, ``"MPS"``, ``"DirectML"``,
    or ``"CPU"``. ROCm is reported as CUDA (since torch unifies the API) but
    the label includes the runtime tag for logging.
    """
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "ROCm (AMD GPU via HIP)"
        return "CUDA"
    if _xpu_available():
        return "XPU (Intel GPU)"
    if _mps_available():
        return "MPS (Apple Silicon GPU)"
    if _directml_available():
        return "DirectML (Windows)"
    return "CPU"


def resolve_device(preference: str = "auto") -> str:
    """Map a user/preference string to an actual torch device string.

    Args:
        preference: ``"auto"`` (use the first available accelerator), or one
            of ``"cuda"``/``"xpu"``/``"mps"``/``"directml"``/``"cpu"``.

    Returns:
        A torch device string (``"cuda"``, ``"xpu"``, ``"mps"``, ``"cpu"``).
        Note: ``"directml"`` is mapped to the DirectML device at call time
        and not returned by this helper; callers wanting DirectML should
        branch on ``detect_accelerator()`` themselves.
    """
    pref = (preference or "auto").lower()
    if pref == "auto":
        return _auto_pick()
    if pref == "cpu":
        return "cpu"
    if pref == "directml":
        logger.warning(
            "DirectML requested but not exposed as a torch device string. "
            "Use detect_accelerator() to check availability and torch-directml "
            "to obtain a device; falling back to CPU for now."
        )
        return "cpu"
    if pref in {"cuda", "xpu", "mps"}:
        return _honour_explicit(pref)
    logger.warning("Unknown device preference %r; falling back to CPU.", preference)
    return "cpu"


def _auto_pick() -> str:
    """Walk the accelerator chain and return the first available device."""
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    if _mps_available():
        return "mps"
    return "cpu"


def _honour_explicit(pref: str) -> str:
    """Return ``pref`` if the corresponding accelerator is available, else CPU."""
    if pref == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if pref == "xpu" and not _xpu_available():
        logger.warning("XPU requested but unavailable; falling back to CPU.")
        return "cpu"
    if pref == "mps" and not _mps_available():
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return "cpu"
    return pref


def recommend_batch_size(tile_size: int, max_vram_gb: float = 0.0) -> int:
    """Recommend a safe ICNet batch size based on available accelerator memory.

    Returns 1 for CPU-only systems. CUDA, ROCm, and XPU all report
    ``total_memory`` via their respective torch APIs, so the same heuristic
    applies across vendors.
    """
    has_accel = (
        torch.cuda.is_available()
        or _xpu_available()
    )
    if not has_accel:
        return 1  # Conservative on CPU / MPS

    vram = max_vram_gb if max_vram_gb > 0 else get_available_vram_gb()
    if vram <= 0:
        return 1

    # Heuristic: ~tile_size^2 * channels(3) * fp16(2 bytes) * model overhead (roughly 20x)
    # plus some extra for the feature maps. We approximate from observation:
    # 512px tiles use ~2GB for a batch of 32.
    mem_per_tile_mb = (tile_size / 512.0) ** 2 * 60.0  # ~60MB per 512 tile in ICNet FP16
    safe_vram_mb = vram * 1024 * 0.85  # leave 15% headroom

    recommended = max(1, int(safe_vram_mb / mem_per_tile_mb))
    # Clamp to multiples of 4, capped at 256.
    return min(256, max(1, (recommended // 4) * 4))


def get_disk_free_gb(path: Path) -> float:
    """Return free space in GB at the given path."""
    try:
        stat = psutil.disk_usage(str(path))
        return stat.free / (1024 ** 3)
    except Exception as e:
        logger.warning("Could not query disk space at %s: %s", path, e)
        return float("inf")


def is_removable_drive(path: Path) -> bool:
    """Attempt to detect if the output path resides on a removable (USB) drive."""
    try:
        # Resolve to a mount point
        resolved = path.resolve()
        for part in psutil.disk_partitions(all=True):
            if resolved.is_relative_to(Path(part.mountpoint)):
                # On Linux, removable drives often appear as 'fuse' or have 'removable' labels.
                # On Windows, one could use win32api but we degrade gracefully.
                if part.fstype in ("fuseblk", "exfat", "vfat", "ntfs"):
                    # Simple heuristic: if the mountpoint is under /media or /mnt on Linux.
                    mp = Path(part.mountpoint)
                    if any(mp.is_relative_to(p) for p in (Path("/media"), Path("/mnt"), Path("/Volumes"))):
                        return True
        return False
    except Exception:
        return False


def recommend_io_throttle(output_path: Path, user_value: float = 0.0) -> float:
    """Return I/O sleep duration in seconds between writes."""
    if user_value > 0:
        return user_value
    if is_removable_drive(output_path):
        return 0.005  # 5ms write spacing for USB
    return 0.0


def estimate_disk_need_gb(input_bytes: float) -> float:
    """Estimate peak workspace size from the actual input volume.

    The pipeline copies kept tiles up to three times (staging -> scored ->
    tiles) plus metadata; ~2x the input bytes is the honest upper bound
    (matches the README contract). The previous tiles-per-image estimator
    assumed 25 tiles per source image and hallucinated terabytes for
    folders whose images already are tiles.
    """
    return (input_bytes * 2) / (1024 ** 3)
