"""Centralized configuration for the LUCID pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir


class LucidConfig:
    """Singleton configuration object holding all user-specified and derived settings."""

    _instance: LucidConfig | None = None

    # --- Directories ---
    input_dir: Path
    output_dir: Path
    workspace: Path | None  # None means auto-generated inside output_dir

    # --- Fixed Config (derived defaults) ---
    tile_size: int = 512
    threshold: float | None = None  # None → auto from calibration table
    dedupe_threshold: float = 0.96
    batch_size: int = 32
    max_workers: int = max(1, os.cpu_count() or 4)
    max_vram_gb: float = 0.0  # 0 means auto-detect
    multiscale: bool = True  # default ON (recommended for SISR GT)
    strict: bool = False
    io_throttle_ms: float = 0.0
    clipiqa_threshold: float = 0.3  # 0.0 = disabled
    dry_run: bool = False
    verbose: bool = False
    resume: bool = False
    cleanup: bool = False
    calibrate: bool = True
    ui_mode: str = "cli"
    log_path: Path | None = None

    @classmethod
    def get(cls) -> LucidConfig:
        if cls._instance is None:
            cls._instance = LucidConfig()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @property
    def cache_dir(self) -> Path:
        return Path(user_cache_dir("lucid", "Phips"))

    @property
    def auto_workspace(self) -> Path:
        if self.workspace is not None:
            return self.workspace
        tag = self.input_dir.name
        return self.output_dir / f"lucid_workspace_{tag}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "threshold": self.threshold,
            "dedupe_threshold": self.dedupe_threshold,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "max_vram_gb": self.max_vram_gb,
            "multiscale": self.multiscale,
            "strict": self.strict,
            "io_throttle_ms": self.io_throttle_ms,
            "clipiqa_threshold": self.clipiqa_threshold,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
            "resume": self.resume,
            "calibrate": self.calibrate,
            "ui_mode": self.ui_mode,
        }
