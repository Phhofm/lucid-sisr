"""Unified pipeline orchestrator - runs all stages in sequence."""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Literal

import torch

from lucid.core import dedupe, filtering, finalize
from lucid.models import ensure_model_downloaded
from lucid.resources import (
    detect_accelerator,
    estimate_disk_need_gb,
    get_disk_free_gb,
    recommend_batch_size,
)
from lucid.state import PipelineState

logger = logging.getLogger(__name__)

PNG_GLOB = "*.png"


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    workspace: Path | None,
    model_path: Path | None,
    tile_size: int,
    threshold: float | None,
    clipiqa_threshold: float,
    dedupe_threshold: float,
    batch_size: int,
    workers: int,
    max_vram_gb: float,
    multiscale: bool,
    io_throttle_sec: float,
    strict: bool,
    resume: bool,
    stage: Literal["all", "ingest", "dedupe", "finalize"],
    dry_run: bool,
    cleanup: bool = False,
    calibrate: bool = True,
    trim_bottom: int = 0,
) -> dict[str, int]:
    """Run the complete LUCID pipeline or a specific stage.

    Args:
        input_dir: Raw image input folder.
        output_dir: Final output folder for filtered tiles.
        workspace: Scratch workspace dir. Auto-creates inside output_dir if None.
        model_path: ICNet weight path; auto-downloads if None.
        tile_size: Tile dimension in pixels (default 512).
        threshold: Minimum ICNet complexity score. If ``None``, the value is
            auto-resolved from the shipped calibration table for the chosen
            ``tile_size`` (R^2 >= 0.94 across all supported sizes).
        clipiqa_threshold: Minimum CLIP-IQA score for artifact filtering.
            Removes ringing/haloring and low-quality tiles. 0.0 = disabled.
        dedupe_threshold: Cosine similarity threshold for deduping (default 0.96).
        batch_size: GPU batch size (0 = auto-detect from VRAM).
        workers: Number of parallel CPU workers for tile extraction.
        max_vram_gb: Cap VRAM usage in GB (0 = use all available).
        multiscale: Enable multi-scale tiling (1.0, 0.75, 0.5, 0.25).
        io_throttle_sec: Seconds to sleep between disk writes (auto-detects USB).
        strict: Abort on first corrupt image.
        resume: Resume a previous run from workspace state DB.
        stage: Run a specific stage only (all, ingest, dedupe, finalize).
        dry_run: Validate inputs and estimate disk usage without writing tiles.
        cleanup: Remove workspace directory after successful completion.
        calibrate: Apply calibration curve to ICNet scores.
        trim_bottom: Drop the lowest-complexity N% of tiles at the dedupe
            stage (0 = off). BHI finding: removing only the worst tiles
            beats keeping only the best. Applied via the scores CSV.

    Returns:
        dict with summary stats: sources_processed, tiles_scored, tiles_passed, tiles_kept.
    """
    stats: dict[str, int] = {
        "sources_processed": 0,
        "tiles_scored": 0,
        "tiles_passed": 0,
        "tiles_kept": 0,
    }

    # Resolve workspace path early (shared across all stages)
    if workspace is None:
        workspace = output_dir / f"lucid_workspace_{input_dir.name}"

    # Detect and announce compute device. The probe covers CUDA, AMD ROCm
    # (via torch.version.hip), Intel XPU (via torch.xpu), Apple MPS, and
    # DirectML on Windows.
    accelerator = detect_accelerator()
    if accelerator == "CPU":
        logger.info("Compute device: CPU (no GPU accelerator detected).")
    elif accelerator.startswith("ROCm"):
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "AMD GPU"
        logger.info("Compute device: %s — %s", accelerator, name)
    elif accelerator.startswith("CUDA"):
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA GPU"
        logger.info("Compute device: %s — %s", accelerator, name)
    else:
        logger.info("Compute device: %s", accelerator)

    # CPU-only warning and safety adjustments
    if accelerator == "CPU":
        logger.warning(
            "No GPU detected. Processing will be slow. "
            "Consider using a GPU or reducing --batch to 2-4."
        )
        torch.set_num_threads(1)
        if batch_size <= 0:
            batch_size = 2
        else:
            batch_size = min(batch_size, 4)

    if model_path is None:
        model_path = ensure_model_downloaded()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Resume: auto-detect existing workspace state DB
    if resume:
        state_db = workspace / ".lucid" / "lucid_state.db"
        if not state_db.exists():
            raise RuntimeError(
                f"Resume requested but no state database found at {state_db}. "
                f"Pass --workspace <path> to the previous workspace directory."
            )

    # Disk space check — count all supported formats to avoid underestimating
    if not dry_run:
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        all_images = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]
        estimated_gb = estimate_disk_need_gb(len(all_images), tile_size, avg_tiles_per_image=25.0)
        check_path = workspace.parent
        free_gb = get_disk_free_gb(check_path)
        if free_gb < estimated_gb * 1.5:
            raise RuntimeError(
                f"Estimated need ~{estimated_gb:.1f} GB but only {free_gb:.1f} GB free on {check_path}. "
                f"Free space or use --workspace on a larger drive."
            )

    # Resolve batch size (if not already clamped above)
    if batch_size <= 0:
        batch_size = recommend_batch_size(tile_size, max_vram_gb)

    # Validate tile size against the supported set, then auto-resolve
    # threshold if not given.
    from lucid.resources.icnet_calibration import (
        SUPPORTED_SIZES,
        suggested_threshold,
    )

    if tile_size not in SUPPORTED_SIZES:
        raise ValueError(
            f"tile_size={tile_size} is not in the supported set "
            f"{sorted(SUPPORTED_SIZES)}. Run `lucid-calibrate` to extend the range."
        )

    if threshold is None:
        threshold = suggested_threshold(tile_size)

    # Run stages
    stages = ["ingest", "dedupe", "finalize"] if stage == "all" else [stage]

    for stg in stages:
        logger.info("=== Starting stage: %s ===", stg.upper())

        if stg == "ingest":
            # Filtering gates for this run; recorded per run so a later
            # --resume cannot silently mix two gate configurations.
            gate_config = {
                "tile_size": tile_size,
                "threshold": threshold,
                "clipiqa_threshold": clipiqa_threshold,
                "dedupe_threshold": dedupe_threshold,
                "trim_bottom": trim_bottom,
                "multiscale": multiscale,
                "calibrate": calibrate,
            }
            state = PipelineState(workspace)
            state.connect()
            try:
                if resume:
                    state.check_run_config(gate_config)
                state.record_run_config(gate_config)
                sources, scored, passed = filtering.run_ingest(
                    state=state,
                    workspace=workspace,
                    input_dir=input_dir,
                    tile_size=tile_size,
                    threshold=threshold,
                    clipiqa_threshold=clipiqa_threshold,
                    model_path=model_path,
                    workers=workers,
                    batch_size=batch_size,
                    multiscale=multiscale,
                    resume=resume,
                    dry_run=dry_run,
                    strict=strict,
                    calibrate=calibrate,
                    io_throttle_sec=io_throttle_sec,
                )
                stats["sources_processed"] = sources
                stats["tiles_scored"] = scored
                stats["tiles_passed"] = passed
            finally:
                state.close()

            # Tier 1.1: discard the per-image stage/ dir. The streaming
            # ingest (v3.0) writes tiles to a per-image subfolder of
            # stage/ that is normally removed as soon as that image is
            # scored. We clean up any leftover subfolders here (e.g. on
            # a crash mid-image, or on a workspace that was started by
            # an older non-streaming version).
            if not dry_run:
                stage_dir = workspace / "stage"
                if stage_dir.exists():
                    n_freed = sum(
                        1 for _ in stage_dir.rglob(PNG_GLOB)
                    )
                    shutil.rmtree(stage_dir, ignore_errors=True)
                    if n_freed:
                        logger.info(
                            "Cleaned up stage/ (%d intermediate tiles removed).",
                            n_freed,
                        )
                # Backward compat: legacy raw_tiles/ from old workspaces
                legacy_raw = workspace / "raw_tiles"
                if legacy_raw.exists():
                    n_freed = sum(
                        1 for _ in legacy_raw.rglob(PNG_GLOB)
                    )
                    shutil.rmtree(legacy_raw, ignore_errors=True)
                    if n_freed:
                        logger.info(
                            "Cleaned up legacy raw_tiles/ (%d tiles).",
                            n_freed,
                        )

        elif stg == "dedupe":
            tile_input = workspace / "scored"
            if not tile_input.exists():
                # Backward compat: older workspaces may have tiles in raw_tiles
                tile_input = workspace / "raw_tiles"
            if tile_input.exists():
                kept, removed = dedupe.run_dedupe(
                    input_dir=tile_input,
                    threshold=dedupe_threshold,
                    batch_size=batch_size,
                    io_throttle=io_throttle_sec,
                    scores_csv=workspace / "scores" / "ingest_scores.csv",
                    trim_bottom=trim_bottom,
                )
                stats["tiles_kept"] = kept
                logger.info("Dedupe: kept %d, removed %d", kept, removed)

            # Tier 1.1: discard scored/redundant/ — these are the dedupe
            # rejects, no longer needed once the unique set has been kept.
            if not dry_run:
                redundant_dir = workspace / "scored" / "redundant"
                if redundant_dir.exists():
                    n_freed = sum(1 for _ in redundant_dir.glob(PNG_GLOB))
                    shutil.rmtree(redundant_dir)
                    if n_freed:
                        logger.info(
                            "Cleaned up scored/redundant/ (%d duplicates removed).",
                            n_freed,
                        )

        elif stg == "finalize":
            tile_input = workspace / "scored"
            if not tile_input.exists():
                tile_input = workspace / "raw_tiles"
            if tile_input.exists():
                finalized = finalize.run_finalize(
                    input_dir=tile_input,
                    output_dir=output_dir,
                    move=True,
                    dry_run=dry_run,
                    dataset_tag=input_dir.name,
                )
                stats["tiles_kept"] = finalized

    # Cleanup workspace if requested and not dry-run
    if cleanup and not dry_run and workspace and workspace.exists():
        logger.info("Cleaning up workspace...")
        shutil.rmtree(workspace)

    # Summary
    logger.info("=== LUCID Pipeline Summary ===")
    logger.info("Sources processed: %d", stats["sources_processed"])
    logger.info("Tiles passed ICNet scoring: %d", stats["tiles_passed"])
    logger.info("Unique tiles finalized: %d", stats["tiles_kept"])

    # Provenance manifest: the full gate configuration + counts next to
    # master_lineage.csv, so a dataset can always be traced to its run.
    if not dry_run:
        import lucid

        manifest = {
            "lucid_version": lucid.__version__,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_dir": str(input_dir),
            "tile_size": tile_size,
            "threshold": threshold,
            "clipiqa_threshold": clipiqa_threshold,
            "dedupe_threshold": dedupe_threshold,
            "trim_bottom": trim_bottom,
            "multiscale": multiscale,
            "calibrate": calibrate,
            "strict": strict,
            "stage": stage,
            "stats": stats,
        }
        manifest_path = output_dir / "run_config.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        logger.info("Wrote run manifest to %s", manifest_path)

    return stats
