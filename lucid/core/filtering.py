"""Tile ingestion with disk-spill architecture for large datasets."""
from __future__ import annotations

import csv
import logging
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..io import BufferedTileWriter
from ..platform import set_process_priority
from ..resources import recommend_batch_size, recommend_io_throttle
from ..state import PipelineState
from .hashing import tile_digest
from .metrics import measure_all
from .validation import load_image_as_rgb

logger = logging.getLogger(__name__)


def _compute_scales(h_orig: int, w_orig: int, tile_size: int, multiscale: bool) -> list[float]:
    """Compute scale factors for multi-scale tiling."""
    scales = [1.0]
    if multiscale:
        scales.extend([0.75, 0.5, 0.25])
        min_dim = min(h_orig, w_orig)
        if min_dim > tile_size:
            scales.append(tile_size / min_dim)
    return sorted({s for s in scales if s > 0}, reverse=True)


def _resample(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Downscale with Pillow BICUBIC — the same kernel as the >8192px
    path in ``validation.py`` (and the MATLAB/benchmark convention).
    A single kernel across the pipeline; no second overshoot profile."""
    im = Image.fromarray(arr)
    return np.array(im.resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8)


def build_gate_list(
    iqa_gates: list[tuple[str, float]] | None, clipiqa_threshold: float
) -> list[tuple[str, float]]:
    """Merge CLI gate specs: legacy --clipiqa first, then --iqa-gate pairs.
    Order matters: cheapest runs first (mini>0 and lower_better probe
    handled by the metric itself). Zero-threshold entries = disabled."""
    gates: list[tuple[str, float]] = []
    if clipiqa_threshold and clipiqa_threshold > 0:
        gates.append(("clipiqa", float(clipiqa_threshold)))
    for name, thr in (iqa_gates or []):
        if thr and thr > 0:
            gates.append((name, float(thr)))
    return gates


def _load_iqa_metric(name: str, device: str):
    """Load one pyiqa metric; returns (model, lower_better) or None."""
    try:
        import pyiqa
    except ImportError:
        logger.warning(
            "pyiqa not installed; skipping IQA gate %s. "
            "Install with: pip install git+https://github.com/chaofengc/IQA-PyTorch.git",
            name,
        )
        return None
    metric = pyiqa.create_metric(name, device=device)
    return metric, bool(getattr(metric, "lower_better", False))


def run_ingest(
    state: PipelineState,
    workspace: Path,
    input_dir: Path,
    tile_size: int,
    threshold: float,
    clipiqa_threshold: float,
    model_path: Path,
    workers: int,
    batch_size: int,
    multiscale: bool,
    resume: bool,
    dry_run: bool = False,
    strict: bool = False,
    calibrate: bool = True,
    io_throttle_sec: float = 0.0,
    iqa_gates: list[tuple[str, float]] | None = None,
) -> tuple[int, int, int]:
    """Run the full ingest pipeline: signal filter -> ICNet score.

    Tile extraction runs in a ProcessPool (CPU-bound signal metrics) and
    tiles are scored by ICNet on the GPU as soon as they land in the
    ``scored/`` directory. The intermediate PNG staging uses a per-image
    subfolder that is removed once the image's tiles are scored, so peak
    disk usage is bounded by one image's worth of tiles (Pass 2.1 + 2.2:
    per-image streaming, no global ``raw_tiles/`` pile-up).

    Args:
        state: Pipeline state tracker.
        workspace: Scratch workspace directory.
        input_dir: Source image folder.
        tile_size: Tile dimension in pixels.
        threshold: Minimum ICNet complexity score.
        clipiqa_threshold: Legacy single-gate alias; use ``iqa_gates``.
            0.0 = disabled.
        model_path: Path to ICNet weights.
        workers: CPU worker count for tile extraction.
        batch_size: GPU batch size for ICNet scoring.
        multiscale: Enable multi-scale tiling.
        resume: Skip already-processed sources.
        dry_run: Validate only, no writes.
        strict: Abort pipeline on the first unreadable/corrupt image.
        calibrate: Apply calibration curve to ICNet scores.
        io_throttle_sec: Seconds to sleep between disk writes (auto-detects USB).
        iqa_gates: Extra pyiqa perceptual gates [(metric_name, threshold)],
            e.g. [("topiq_nr", 0.45), ("qrealign-mini", 0.4)]. Stored order
            = cascade order (cheapest first). Threshold semantics follow
            the metric's own direction (lower_better respected).
            Requires the ``iqa`` extra: pip install -e ".[iqa]".

    Returns:
        (sources_processed, tiles_scored, tiles_passed) for summary stats.
    """

    io_throttle = recommend_io_throttle(workspace, io_throttle_sec)
    # Per-image staging dir; replaced the flat `raw_tiles/` layout in v3.0
    # so disk peaks at one image's tiles, not the full dataset.
    stage_dir = workspace / "stage"
    scored_dir = workspace / "scored"
    scores_csv = workspace / "scores" / "ingest_scores.csv"

    if not dry_run:
        stage_dir.mkdir(parents=True, exist_ok=True)
        scored_dir.mkdir(parents=True, exist_ok=True)
        scores_csv.parent.mkdir(parents=True, exist_ok=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    all_images = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts]

    # Tier 2.4: warn if a large fraction of the source is lossy JPEG. SISR
    # models reproduce whatever is in the GT, so JPEG artifacts become
    # part of the training distribution. Recommend lossless source.
    if all_images:
        jpg_count = sum(
            1 for p in all_images if p.suffix.lower() in {".jpg", ".jpeg"}
        )
        jpg_frac = jpg_count / len(all_images)
        if jpg_frac > 0.5:
            logger.warning(
                "%.0f%% of input (%d/%d) is JPEG. SISR GT works best with "
                "lossless source (PNG/TIFF); JPEG compression artifacts will "
                "be learned by the model. Re-export from RAW/PNG if possible.",
                jpg_frac * 100,
                jpg_count,
                len(all_images),
            )

    if resume:
        sources_to_do = state.get_pending_sources(all_images)
    else:
        for img in all_images:
            state.register_source(img)
        sources_to_do = all_images

    logger.info("Ingesting %d source images (signal filter)...", len(sources_to_do))

    # Set up ICNet model once for streaming scoring. Skip for dry-run.
    effective_batch_size = batch_size if batch_size else recommend_batch_size(tile_size)
    model_ctx = _build_scoring_context(
        threshold=threshold,
        model_path=model_path,
        batch_size=effective_batch_size,
        tile_size=tile_size,
        io_throttle=io_throttle,
        calibrate=calibrate,
        clipiqa_threshold=clipiqa_threshold,
        iqa_gates=iqa_gates,
    ) if not dry_run else None

    # Tier 3.x accumulators (histogram, threshold audit, outliers).
    # Populated incrementally across the streaming loop.
    if not dry_run:
        _init_score_csv(scores_csv, [g[0] for g in build_gate_list(iqa_gates, clipiqa_threshold)])
    all_raw: list[float] = []
    all_cal: list[float] = []
    all_names: list[str] = []
    all_paths: list[str] = []
    # Per-scale tags seen -> [seen, signal_passed, icnet_passed]
    scale_stats: dict[str, list[int]] = {}

    # Phase 1+2 fused: extract one image, score it, delete staging, repeat.
    # This is Pass 2.1 (per-image streaming) + Pass 2.2 (no global raw_tiles).
    sources_processed = 0
    tiles_passed_total = 0
    if not dry_run:
        # ProcessPoolExecutor bypasses the GIL so signal-metric work
        # (numpy ops on 512x512x3 arrays) actually parallelizes across cores.
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_img = {
                executor.submit(
                    extract_tiles_to_staging,
                    img_path,
                    tile_size,
                    stage_dir,
                    multiscale,
                    strict,
                ): img_path
                for img_path in sources_to_do
            }
            for future in tqdm(
                as_completed(future_to_img),
                total=len(future_to_img),
                desc="Extracting tiles",
            ):
                img_path = future_to_img[future]
                try:
                    image_stage, num_passed, err, per_scale = future.result()
                except RuntimeError:
                    raise
                except Exception as exc:
                    if strict:
                        raise RuntimeError(
                            f"--strict: aborting due to worker error on {img_path}: {exc}"
                        ) from exc
                    logger.warning("Worker failed on %s: %s", img_path, exc)
                    state.mark_source_status(img_path, "failed", str(exc))
                    sources_processed += 1
                    continue

                # Aggregate per-scale tile statistics (both on failures
                # with partial stats and on success).
                for tag, (seen, passed) in per_scale.items():
                    _acc = scale_stats.setdefault(tag, [0, 0, 0])
                    _acc[0] += seen
                    _acc[1] += passed

                if err:
                    if strict:
                        raise RuntimeError(
                            f"--strict: aborting due to error on {img_path}: {err}"
                        )
                    state.mark_source_status(img_path, "failed", err)
                    sources_processed += 1
                    if image_stage and image_stage.exists():
                        shutil.rmtree(image_stage, ignore_errors=True)
                    continue

                # Tier 3.6: per-image pass-rate warning
                if num_passed == 0:
                    logger.warning(
                        "0 tiles passed signal filter for %s "
                        "(image may be too small for --tile-size, all-blur, "
                        "or all-sky)",
                        img_path.name,
                    )

                # Score this image's tiles immediately, then drop the staging dir.
                tiles_scored_this = 0
                if image_stage and num_passed > 0:
                    assert model_ctx is not None, "scoring context missing in non-dry-run path"
                    tiles_scored_this, batch_records = _score_staged_tiles(
                        model_ctx,
                        image_stage,
                        scored_dir,
                        scores_csv,
                        source_image=img_path,
                    )
                    all_raw.extend(batch_records["raw"])
                    all_cal.extend(batch_records["cal"])
                    all_names.extend(batch_records["names"])
                    all_paths.extend(batch_records["paths"])
                    tiles_passed_total += tiles_scored_this
                    for name, ok in zip(batch_records["names"], batch_records["passed"]):
                        tag = re.search(r"_s(\d{3})_t", name).group(1)
                        scale_stats.setdefault(f"s{tag}", [0, 0, 0])[2] += int(ok)

                # Cleanup: drop this image's staging subfolder so peak disk
                # is bounded to one image's tiles.
                if image_stage and image_stage.exists():
                    shutil.rmtree(image_stage, ignore_errors=True)

                state.mark_source_status(
                    img_path,
                    "scored" if tiles_scored_this > 0 else "skipped",
                )
                sources_processed += 1
    else:
        # Dry-run: still walk the source list to count, no real work
        sources_processed = len(sources_to_do)

    tiles_passed = tiles_passed_total

    # Tier 3.1 / 3.3 / 3.5: post-run artifacts.
    if all_raw and not dry_run:
        _write_score_artifacts(
            scores_csv=scores_csv,
            scored_dir=scored_dir,
            raw=all_raw,
            cal=all_cal,
            names=all_names,
            paths=all_paths,
            tile_size=tile_size,
            threshold=threshold,
            calibrate=calibrate,
            clipiqa_threshold=clipiqa_threshold,
            scale_stats=scale_stats,
            gate_specs=build_gate_list(iqa_gates, clipiqa_threshold),
        )

    # Remove the empty stage dir if no per-image subfolders left
    if stage_dir.exists() and not any(stage_dir.iterdir()):
        try:
            stage_dir.rmdir()
        except OSError:
            pass

    if clipiqa_threshold > 0:
        logger.info("ICNet + CLIP-IQA scoring complete: %d/%d tiles passed (clipiqa >= %.2f)", tiles_passed, len(all_raw), clipiqa_threshold)
    else:
        logger.info("ICNet scoring complete: %d/%d tiles passed.", tiles_passed, len(all_raw))
    return sources_processed, len(all_raw), tiles_passed


def extract_tiles_to_staging(
    img_path: Path,
    tile_size: int,
    stage_root: Path,
    multiscale: bool = False,
    strict: bool = False,
) -> tuple[Path | None, int, str | None]:
    """Extract tiles from a single image into a per-image staging subfolder.

    Returns:
        (image_stage_dir, num_passed, error). The dir is removed by the
        caller once that image's tiles have been scored.
    """

    set_process_priority()

    img_np = load_image_as_rgb(img_path)
    if img_np is None:
        return None, 0, f"Could not load image: {img_path.name}", {}

    h, w, _ = img_np.shape
    scales = _compute_scales(h, w, tile_size, multiscale)

    image_stage = stage_root / img_path.stem
    image_stage.mkdir(parents=True, exist_ok=True)

    total_passed = 0
    seen_hashes: set[int] = set()
    dup_count = 0
    scale_stats: dict[str, tuple[int, int]] = {}
    for scale in scales:
        tag = f"s{int(scale * 100):03d}"
        tile_img = (
            img_np
            if scale == 1.0
            else _resample(img_np, int(w * scale), int(h * scale))
        )
        th, tw, _ = tile_img.shape
        name_base = f"{img_path.stem}_{tag}"

        seen = signal_passed = 0
        for y in range(th // tile_size):
            for x in range(tw // tile_size):
                tile = tile_img[
                    y * tile_size : (y + 1) * tile_size,
                    x * tile_size : (x + 1) * tile_size,
                ]
                seen += 1
                gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)

                metrics, passes = measure_all(gray)
                if not passes:
                    continue

                # Tier 2.3: drop pixel-identical boundary tiles
                h_digest = tile_digest(tile)
                if h_digest is not None:
                    if h_digest in seen_hashes:
                        dup_count += 1
                        continue
                    seen_hashes.add(h_digest)

                tile_name = f"{name_base}_t{y}_{x}.png"
                cv2.imwrite(
                    str(image_stage / tile_name),
                    cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                )
                signal_passed += 1
        total_passed += signal_passed
        scale_stats[tag] = (seen, signal_passed)

    if dup_count:
        logger.debug(
            "Dropped %d exact-byte duplicate tiles from %s",
            dup_count,
            img_path.name,
        )

    if strict and total_passed == 0 and h >= tile_size and w >= tile_size:
        # strict mode: surface the failure
        return image_stage, 0, f"0 tiles passed signal filter for {img_path.name}", scale_stats

    return image_stage, total_passed, None, scale_stats


def _load_batch_items(
    batch_tiles: list[Path],
) -> list[tuple[Path, np.ndarray]]:
    """Decode a list of PNG paths to BGR→RGB numpy arrays.

    Skips and logs any tile that fails to decode. Returned list is empty
    if every tile in the batch was unreadable.
    """
    items: list[tuple[Path, np.ndarray]] = []
    for t in batch_tiles:
        arr = cv2.imread(str(t), cv2.IMREAD_COLOR)
        if arr is None:
            logger.warning("Could not read tile %s, skipping", t)
            continue
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        items.append((t, rgb))
    return items


class _ScoringContext:
    """Holds the loaded ICNet model and scoring parameters.

    Created once at the start of ``run_ingest`` (Pass 2.1) and reused
    for every per-image staging batch.
    """

    __slots__ = (
        "model",
        "device",
        "use_half",
        "mean",
        "std",
        "threshold",
        "tile_size",
        "calibrate",
        "writer",
        "clipiqa_model",
        "clipiqa_threshold",
        "batch_size",
        "io_throttle",
        "iqa_gates",
    )

    def __init__(self, model, device, use_half, mean, std, threshold,
                 tile_size, calibrate, clipiqa_model=None, clipiqa_threshold=0.0,
                 batch_size=32, io_throttle=0.0, iqa_gates=None):
        self.model = model
        self.device = device
        self.use_half = use_half
        self.mean = mean
        self.std = std
        self.threshold = threshold
        self.tile_size = tile_size
        self.calibrate = calibrate
        self.writer: BufferedTileWriter | None = None
        self.clipiqa_model = clipiqa_model
        self.clipiqa_threshold = clipiqa_threshold
        # Perceptual tier: list of dicts (name, model, threshold, lower_better).
        # Gates run in cascade order; the legacy clipiqa slot auto-registers
        # as the first gate for backward compatibility.
        gates = list(iqa_gates or [])
        if clipiqa_model is not None and clipiqa_threshold and clipiqa_threshold > 0:
            gates.insert(0, dict(
                name="clipiqa", model=clipiqa_model,
                threshold=clipiqa_threshold, lower_better=False,
            ))
        self.iqa_gates = gates
        self.batch_size = batch_size
        self.io_throttle = io_throttle


def _build_scoring_context(
    threshold: float,
    model_path: Path,
    batch_size: int,
    tile_size: int,
    io_throttle: float,
    calibrate: bool,
    clipiqa_threshold: float = 0.0,
    iqa_gates: list[tuple[str, float]] | None = None,
) -> _ScoringContext:
    """Build the ICNet scoring context once for the streaming ingest loop.

    The returned context is reused for every per-image batch. The model
    is moved to the resolved device (CUDA / ROCm / XPU / MPS / DirectML /
    CPU) and cast to FP16 on real accelerators for throughput.
    """
    from ..models.icnet import ICNet
    from ..resources import resolve_device
    from ..resources.icnet_calibration import (
        calibration_status_label,
        suggested_threshold,
    )

    if not calibrate:
        logger.info(
            "Score calibration disabled (--no-calibrate); raw ICNet scores "
            "will be used for the threshold comparison."
        )
    elif tile_size != 512:
        r2_label = calibration_status_label(tile_size)
        logger.info(
            "Tile size %dpx is non-native; using %s calibration. "
            "Suggested --threshold is %.2f (passed %.2f).",
            tile_size,
            r2_label,
            suggested_threshold(tile_size),
            threshold,
        )

    device = resolve_device("auto")
    use_half = device in ("cuda", "xpu")

    model = ICNet(is_pretrain=False, size1=tile_size, size2=tile_size // 2)
    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    model.to(device).eval()
    if use_half:
        model.half()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if use_half:
        mean = mean.half()
        std = std.half()

    # Unified perceptual tier: legacy clipiqa + --iqa-gate list, loaded in
    # cascade order; each gate skipped with a warning when unavailable.
    loaded_gates = []
    for name, thr in build_gate_list(iqa_gates, clipiqa_threshold):
        res = _load_iqa_metric(name, device)
        if res is None:
            continue
        metric, lower_better = res
        loaded_gates.append(
            dict(name=name, model=metric, threshold=thr, lower_better=lower_better)
        )
        logger.info(
            "IQA gate %s: threshold=%s (%s-better)",
            name, thr, "lower" if lower_better else "higher",
        )

    return _ScoringContext(
        model=model,
        device=device,
        use_half=use_half,
        mean=mean,
        std=std,
        threshold=threshold,
        tile_size=tile_size,
        calibrate=calibrate,
        iqa_gates=loaded_gates,
        batch_size=batch_size,
        io_throttle=io_throttle,
    )


def _init_score_csv(scores_csv: Path, gate_names: list[str]) -> None:
    """Write the header row to the scores CSV. Idempotent — truncates
    the file if it already exists."""
    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_csv, "w", newline="") as csv_f:
        csv.writer(csv_f).writerow(
            ["tile_name", "complexity_score", "score_calibrated",
             "calibration_outlier", "source", "source_image"]
            + [f"iqa_{g}" for g in gate_names]
        )


def _score_batch(
    ctx: _ScoringContext,
    items: list[tuple[Path, "np.ndarray"]],  # noqa: F821
    scores_csv: Path,
    source_image: Path | None = None,
) -> tuple[int, dict[str, list]]:
    """Score a single batch of (tile_path, rgb_array) and stream results.

    Returns ``(tiles_passed, record_dict)`` where ``record_dict`` has keys
    ``raw``, ``cal``, ``names``, ``paths`` for the post-run artifacts.
    """
    from ..resources.icnet_calibration import apply_calibration

    record: dict[str, list] = {"raw": [], "cal": [], "names": [], "paths": [], "passed": []}
    if not items:
        return 0, record

    try:
        batch_imgs = [rgb for _, rgb in items]
        stacked = (
            torch.stack([torch.from_numpy(img) for img in batch_imgs])
            .to(ctx.device)
            .permute(0, 3, 1, 2)
        )
        tensors = stacked.half() if ctx.use_half else stacked.float()
        tensors = (tensors / 255.0 - ctx.mean) / ctx.std

        with torch.no_grad():
            scores = (
                ctx.model(tensors)[0].flatten().cpu().float().numpy()
            )
    except Exception as exc:
        logger.warning("Batch scoring failed: %s", exc)
        return 0, record

    # CLIP-IQA scoring: run on the same batch for artifact detection
    # Perceptual tier: gates run in cascade order on granular-tile inputs;
    # a tile that fails a gate is masked out of the later (more expensive)
    # gates' batches.
    alive = list(range(len(items)))
    gate_scores: dict[str, list[float]] = {g["name"]: [0.0] * len(items) for g in ctx.iqa_gates}
    if ctx.iqa_gates and alive:
        with torch.no_grad():
            # gates expect [0,1] float tensors, not normalized
            gate_input = stacked.float() / 255.0
            for g in ctx.iqa_gates:
                if not alive:
                    break
                sub = gate_input[torch.tensor(alive)]
                try:
                    s = g["model"](sub).flatten().cpu().float().numpy()
                except Exception as exc:
                    # Fail closed: an unverified tile must never enter the
                    # dataset. This used to skip the gate and admit the whole
                    # batch unverified (e.g. after CUDA OOM).
                    logger.warning(
                        "IQA gate %s failed (%s); rejecting %d unverified tiles",
                        g["name"], exc, len(alive),
                    )
                    alive = []
                    break
                for pos, val in zip(alive, s):
                    gate_scores[g["name"]][pos] = float(val)
                    ok = (
                        float(val) <= g["threshold"]
                        if g["lower_better"]
                        else float(val) >= g["threshold"]
                    )
                    if not ok:
                        alive = [a for a in alive if a != pos]
    all_alive = set(alive)
    gate_names = list(gate_scores.keys())

    tiles_passed = 0
    with open(scores_csv, "a", newline="") as csv_f:
        writer_csv = csv.writer(csv_f)
        for i, ((t, rgb_arr), sc_raw) in enumerate(zip(items, scores)):
            sc_raw = float(sc_raw)
            sc_cal = (
                apply_calibration(sc_raw, ctx.tile_size)
                if ctx.calibrate
                else sc_raw
            )
            outlier = abs(sc_raw - sc_cal) > 0.10

            # ICNet gate
            icnet_pass = sc_cal >= ctx.threshold
            iqa_ok = i in all_alive

            if icnet_pass and iqa_ok:
                if ctx.writer is not None:
                    ctx.writer.write(t.name, rgb_arr)
                tiles_passed += 1
            writer_csv.writerow(
                [
                    t.name,
                    f"{sc_raw:.6f}",
                    f"{sc_cal:.6f}",
                    "1" if outlier else "0",
                    str(t),
                    str(source_image) if source_image else "",
                ]
                + [f"{gate_scores[g][i]:.6f}" for g in gate_names]
            )
            record["raw"].append(sc_raw)
            record["cal"].append(sc_cal)
            record["names"].append(t.name)
            record["paths"].append(str(t))
            record["passed"].append(bool(icnet_pass and iqa_ok))
    return tiles_passed, record


def _score_staged_tiles(
    ctx: _ScoringContext,
    image_stage: Path,
    scored_dir: Path,
    scores_csv: Path,
    source_image: Path | None = None,
) -> tuple[int, dict[str, list]]:
    """Score every PNG in ``image_stage`` (Pass 2.1 streaming).

    The caller is expected to delete ``image_stage`` after this returns.
    """
    from ..io import BufferedTileWriter

    tile_paths = sorted(image_stage.glob("*.png"))
    if not tile_paths:
        return 0, {"raw": [], "cal": [], "names": [], "paths": [], "passed": []}

    tiles_passed_total = 0
    record: dict[str, list] = {"raw": [], "cal": [], "names": [], "paths": [], "passed": []}
    batch_size = ctx.batch_size

    with BufferedTileWriter(
        scored_dir, throttle_sec=ctx.io_throttle, buffer_size=64
    ) as writer:
        ctx.writer = writer
        for i in range(0, len(tile_paths), batch_size):
            chunk = tile_paths[i : i + batch_size]
            tp, rec = _score_batch(ctx, _load_batch_items(chunk), scores_csv, source_image=source_image)
            tiles_passed_total += tp
            record["raw"].extend(rec["raw"])
            record["cal"].extend(rec["cal"])
            record["names"].extend(rec["names"])
            record["paths"].extend(rec["paths"])
            record["passed"].extend(rec["passed"])
    ctx.writer = None
    return tiles_passed_total, record


def _write_score_artifacts(

    scores_csv: Path,
    scored_dir: Path,
    raw: list[float],
    cal: list[float],
    names: list[str],
    paths: list[str],
    tile_size: int,
    threshold: float,
    calibrate: bool,
    clipiqa_threshold: float = 0.0,
    scale_stats: dict[str, list[int]] | None = None,
    gate_specs: list[tuple[str, float]] | None = None,
) -> None:
    """Tier 3.1 / 3.3 / 3.5: post-run summary artifacts.

    Writes the score histogram (text + optional PNG), the threshold
    audit file, and the top-100 calibration outliers CSV.
    """
    from lucid.resources.icnet_calibration import (
        calibration_r_squared,
        suggested_threshold,
    )

    # ---- Tier 3.1: score histogram ----
    bins = [0] * 20
    for s in cal:
        idx = min(19, max(0, int(s * 20)))
        bins[idx] += 1
    n_total = len(cal)
    n_pass = sum(1 for s in cal if s >= threshold)
    n_rej = n_total - n_pass

    hist_path = scores_csv.parent / "histogram.txt"
    with open(hist_path, "w") as f:
        f.write("# complexity score histogram (score_calibrated)\n")
        f.write(f"# total tiles: {n_total}\n")
        f.write(
            f"# tiles passing threshold ({threshold:.3f}): {n_pass}\n"
        )
        f.write(f"# tiles rejected: {n_rej}\n")
        f.write("# bin_width: 0.05\n\n")
        f.write("bin_start  bin_end  count  bar\n")
        for i, count in enumerate(bins):
            lo = i * 0.05
            hi = (i + 1) * 0.05
            bar = "#" * min(60, int(count / max(1, n_total / 60)))
            marker = (
                " <-- threshold" if threshold >= lo and threshold < hi else ""
            )
            f.write(
                f"  {lo:.2f}     {hi:.2f}  {int(count):5d}  {bar}{marker}\n"
            )
    logger.info("Wrote score histogram to %s", hist_path)

    # Optional PNG plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        centers = [i * 0.05 + 0.025 for i in range(20)]
        ax.bar(centers, bins, width=0.045, color="#4F46E5", edgecolor="white")
        ax.axvline(
            threshold, color="#EF4444", linestyle="--",
            label=f"threshold = {threshold:.3f}",
        )
        ax.set_xlabel("Calibrated complexity score")
        ax.set_ylabel("Tile count")
        ax.set_title(f"LUCID score distribution ({n_total} tiles)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        png_path = scores_csv.parent / "histogram.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        logger.info("Wrote score histogram PNG to %s", png_path)
    except ImportError:
        pass  # matplotlib is optional

    # ---- Tier 3.3: threshold audit file ----
    threshold_path = scored_dir / "threshold_used.txt"
    with open(threshold_path, "w") as f:
        f.write(f"tile_size: {tile_size}\n")
        f.write(f"threshold_used: {threshold:.6f}\n")
        f.write(f"suggested_threshold: {suggested_threshold(tile_size):.6f}\n")
        f.write(
            f"calibration_r_squared: {calibration_r_squared(tile_size):.4f}\n"
        )
        f.write(f"calibration_enabled: {calibrate}\n")
        for gname, gthr in (gate_specs or []):
            f.write(f"iqa_gate_{gname}: on (thr={gthr:.4f})\n")
        f.write(f"tiles_scored: {n_total}\n")
        f.write(f"tiles_passing: {n_pass}\n")
        f.write(f"tiles_rejected: {n_rej}\n")
    logger.info("Wrote threshold audit to %s", threshold_path)

    # ---- Per-scale pass-rate stats: is each multiscale scale earning
    # its compute? (e.g. 0.25x tiles often fail the complexity gate.)
    if scale_stats:
        scale_path = scores_csv.parent / "scale_pass_rates.txt"
        with open(scale_path, "w") as f:
            f.write("# per-scale tile statistics\n")
            f.write("# tag    seen  signal_passed  icnet_passed\n\n")
            for tag, (seen, sig, icn) in sorted(scale_stats.items()):
                rate = sig / seen if seen else 0.0
                f.write(
                    f"  {tag}  {seen:6d}  {sig:6d}  {icn:6d}"
                    f"  ({rate:5.1%} signal)\n"
                )
        logger.info("Wrote per-scale stats to %s", scale_path)

    # ---- Tier 3.5: top-100 calibration outliers ----
    outliers = [
        (abs(r - c), n, p, r, c)
        for n, p, r, c in zip(names, paths, raw, cal)
    ]
    outliers.sort(reverse=True)
    outlier_path = scores_csv.parent / "calibration_outliers.csv"
    with open(outlier_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "abs_error",
                "tile_name",
                "raw_score",
                "calibrated_score",
                "source",
            ]
        )
        for err, n, p, r, c in outliers[:100]:
            w.writerow(
                [f"{err:.4f}", n, f"{r:.6f}", f"{c:.6f}", p]
            )
    outlier_count = sum(1 for e, *_ in outliers if e > 0.10)
    if outlier_count:
        logger.info(
            "%d tiles are calibration outliers (|raw-cal| > 0.10); "
            "top 100 written to %s",
            outlier_count,
            outlier_path,
        )
