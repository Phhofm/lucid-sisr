"""LUCID - Unified CLI entry point.

Single command execution for dataset filtering:
    python -m lucid --input /raw/dataset --output /final/tiles

Creates:
    - output/tiles/ - final filtered tiles (00001_dataset.png, ...)
    - output/master_lineage.csv - provenance mapping
    - .lucid/ workspace with intermediate files and state DB
"""

import argparse
import logging
import sys
from pathlib import Path

import torch.multiprocessing as mp

from lucid.config import LucidConfig
from lucid.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LUCID - SISR Dataset Filtering (Unified Pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "positional_input",
        nargs="?",
        default=None,
        help="Input folder with raw images (positional form of --input)",
    )
    parser.add_argument(
        "positional_output",
        nargs="?",
        default=None,
        help="Output folder for final filtered tiles (positional form of --output)",
    )
    parser.add_argument("--input", default=None, help="Input folder with raw images")
    parser.add_argument(
        "--output",
        default="./lucid_output",
        help="Output folder for final filtered tiles (default: ./lucid_output)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Scratch workspace path (default: auto-generated inside --output)",
    )
    parser.add_argument(
        "--icnet",
        type=Path,
        default=None,
        help="Path to ICNet weights (default: auto-download to platform cache dir)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Minimum complexity score (0.0–1.0). "
            "Default: auto-adjusted per --tile-size from the shipped "
            "calibration table. Pass an explicit value to override."
        ),
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.96,
        help="Cosine similarity threshold above which tiles are considered duplicates",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help=(
            "Tile dimension in pixels. Supported: 128, 256, 512, 1024. "
            "512 is native; other sizes use a calibration curve to map "
            "raw ICNet scores to the 512-trained distribution."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 2 or 4),
        help="CPU worker count",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="GPU batch size (0 = auto-detect from VRAM)",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=0.0,
        help="Max VRAM to use in GB (0 = auto-detect)",
    )
    parser.add_argument(
        "--multiscale",
        dest="multiscale",
        action="store_true",
        default=True,
        help="Enable multi-scale tiling at 1.0/0.75/0.5/0.25 (DEFAULT: on; pass --no-multiscale to disable)",
    )
    parser.add_argument(
        "--no-multiscale",
        dest="multiscale",
        action="store_false",
        help="Disable multi-scale tiling; tile at a single resolution only",
    )
    parser.add_argument(
        "--io-throttle-ms",
        type=float,
        default=0.0,
        help="Milliseconds to sleep between disk writes (0 = auto-detect for USB)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort pipeline on the first corrupt image",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run from the workspace state database",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "ingest", "dedupe", "finalize"],
        default="all",
        help="Run only a specific stage (for debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and estimates without writing tiles",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove .lucid workspace after successful completion",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-calibrate",
        dest="calibrate",
        action="store_false",
        help=(
            "Disable score calibration. Use raw ICNet scores for the "
            "threshold comparison. Default: enabled (recommended)."
        ),
    )
    parser.add_argument(
        "--clipiqa-threshold",
        type=float,
        default=0.3,
        help=(
            "Minimum CLIP-IQA quality score (0.0-1.0) for artifact filtering. "
            "Removes ringing/haloring and low-quality tiles. "
            "Set to 0.0 or pass --no-clipiqa to disable. Default: 0.3"
        ),
    )
    parser.add_argument(
        "--no-clipiqa",
        dest="clipiqa_threshold",
        action="store_const",
        const=0.0,
        help="Disable CLIP-IQA quality filtering",
    )
    return parser


def main() -> int:
    # Set spawn start method before any CUDA init to avoid fork+CUDA deadlocks.
    # Must happen inside main() so importing lucid.__main__ in tests doesn't
    # mutate the global multiprocessing context.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set (e.g. when called multiple times in tests)

    parser = build_parser()
    args = parser.parse_args()

    # Resolve input/output: positional wins when given, else --input/--output
    resolved_input = args.positional_input or args.input
    if resolved_input is None:
        parser.error(
            "an input folder is required: pass it positionally "
            "(e.g. `lucid /path/to/images`) or via --input PATH"
        )
    resolved_output = args.positional_output or args.output

    # Initialize singleton config
    cfg = LucidConfig.get()
    cfg.input_dir = Path(resolved_input)
    cfg.output_dir = Path(resolved_output)
    cfg.workspace = Path(args.workspace) if args.workspace else None
    cfg.tile_size = args.tile_size
    cfg.dedupe_threshold = args.dedupe_threshold
    cfg.batch_size = args.batch
    cfg.max_workers = args.workers
    cfg.max_vram_gb = args.max_vram_gb
    cfg.multiscale = args.multiscale
    cfg.io_throttle_ms = args.io_throttle_ms / 1000.0
    cfg.strict = args.strict
    cfg.resume = args.resume
    cfg.verbose = args.verbose
    cfg.dry_run = args.dry_run
    cfg.cleanup = args.cleanup
    cfg.calibrate = args.calibrate
    cfg.clipiqa_threshold = args.clipiqa_threshold
    # threshold stays None here; the orchestrator resolves it from the
    # calibration table once tile_size is validated.

    # Logging
    log_level = logging.DEBUG if cfg.verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)

    # Validate tile size against the supported set
    from lucid.resources.icnet_calibration import (
        SUPPORTED_SIZES,
    )

    if cfg.tile_size not in SUPPORTED_SIZES:
        logger.error(
            "--tile-size=%d is not in the supported set %s. "
            "Pick one of these or run `lucid-calibrate` to extend the range.",
            cfg.tile_size,
            sorted(SUPPORTED_SIZES),
        )
        sys.exit(2)

    # Auto-adjust threshold per tile size from the calibration table.
    from lucid.resources.icnet_calibration import suggested_threshold

    if cfg.threshold is None:
        cfg.threshold = suggested_threshold(cfg.tile_size)
        logger.info(
            "Auto-threshold for %dpx: %.3f (pass --threshold to override).",
            cfg.tile_size,
            cfg.threshold,
        )

    if cfg.dry_run:
        logger.info("=== DRY RUN === No tiles will be written.")

    try:
        run_pipeline(
            input_dir=cfg.input_dir,
            output_dir=cfg.output_dir,
            workspace=cfg.workspace,
            model_path=args.icnet,
            tile_size=cfg.tile_size,
            threshold=cfg.threshold,
            clipiqa_threshold=cfg.clipiqa_threshold,
            dedupe_threshold=cfg.dedupe_threshold,
            batch_size=cfg.batch_size,
            workers=cfg.max_workers,
            max_vram_gb=cfg.max_vram_gb,
            multiscale=cfg.multiscale,
            io_throttle_sec=cfg.io_throttle_ms,
            strict=cfg.strict,
            resume=cfg.resume,
            stage=args.stage,
            dry_run=cfg.dry_run,
            cleanup=cfg.cleanup,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=cfg.verbose)
        return 1

    logger.info("=== LUCID pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
