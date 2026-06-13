"""Final consolidation and lineage tracking for filtered tiles."""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_finalize(
    input_dir: Path,
    output_dir: Path,
    move: bool = False,
    dry_run: bool = False,
    dataset_tag: str | None = None,
) -> int:
    """Move or copy scored tiles into final output with sequential naming.

    Args:
        input_dir: Directory containing scored/validated tiles.
        output_dir: Destination for final tiles.
        move: If True, move files; otherwise copy.
        dry_run: If True, only count tiles without moving.
        dataset_tag: Source dataset name embedded in final filenames
            (e.g. ``00001_<dataset_tag>.png``). If omitted, recovered
            from the workspace folder name or the original image stem.

    Returns:
        Number of tiles finalized.
    """
    # Find all PNG tiles (exclude redundant folder if present)
    all_pngs = [
        p
        for p in input_dir.rglob("*.png")
        if "redundant" not in p.parts
    ]
    all_pngs = sorted(all_pngs)

    if not all_pngs:
        logger.warning("No tiles found in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = output_dir / "master_lineage.csv"

    # Find the last sequential ID if appending
    start_index = _get_last_index(lineage_path) + 1

    fallback_tag = _resolve_dataset_tag(dataset_tag, input_dir)

    records: list[dict[str, object]] = []
    kept = 0

    for p in tqdm(all_pngs, desc="Finalizing"):
        tag = _tag_for_tile(p, fallback_tag)

        new_name = f"{start_index:05d}_{tag}.png"
        dst = tiles_dir / new_name

        if not dry_run:
            try:
                if move:
                    shutil.move(str(p), str(dst))
                else:
                    shutil.copy2(str(p), str(dst))
            except Exception as exc:
                logger.warning("Could not finalize %s: %s", p, exc)
                continue

        records.append(
            {
                "id": start_index,
                "final_name": new_name,
                "source_dataset": tag,
                "original_name": p.name,
                "original_path": str(p.absolute()),
            }
        )
        start_index += 1
        kept += 1

    # Append to lineage CSV
    if records and not dry_run:
        file_exists = lineage_path.exists()
        # Tier 3.4: enrich with per-image complexity stats from the scores CSV
        # (if available) so users can spot images where every tile was marginal.
        per_image_stats = _collect_per_image_stats(input_dir)
        fields = [
            "id",
            "final_name",
            "source_dataset",
            "original_name",
            "original_path",
            "source_image_min_complexity",
            "source_image_mean_complexity",
            "source_image_max_complexity",
            "source_image_tiles_kept",
        ]
        for rec in records:
            # The scores CSV is keyed by source-image stem; the lineage
            # original_name is the full tile name. Extract the stem for the
            # lookup.
            stem = str(rec["original_name"])
            if "_s" in stem:
                stem = stem.split("_s", 1)[0]
            stats = per_image_stats.get(stem, {})
            rec["source_image_min_complexity"] = str(stats.get("min", ""))
            rec["source_image_mean_complexity"] = str(stats.get("mean", ""))
            rec["source_image_max_complexity"] = str(stats.get("max", ""))
            rec["source_image_tiles_kept"] = str(stats.get("count", ""))
        with open(lineage_path, "a" if file_exists else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)

    logger.info("Finalized %d tiles to %s", kept, output_dir)
    return kept


def _collect_per_image_stats(input_dir: Path) -> dict[str, dict[str, float]]:
    """Aggregate per-image score stats from the workspace scores CSV.

    Returns a mapping of ``original_name -> {min, mean, max, count}``.
    """
    scores_csv = input_dir.parent / "scores" / "ingest_scores.csv"
    if not scores_csv.exists():
        return {}
    by_image: dict[str, list[float]] = {}
    try:
        with open(scores_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("source", "")
                # source path looks like /workspace/raw_tiles/<img_stem>_s100_t0_0.png
                # Pull the image stem (everything before the _sNNN_ marker).
                src_name = Path(src).name
                if "_s" in src_name:
                    stem = src_name.split("_s", 1)[0]
                else:
                    stem = src_name.rsplit(".", 1)[0]
                if not stem:
                    continue
                try:
                    sc = float(row.get("score_calibrated", "nan"))
                except (TypeError, ValueError):
                    continue
                by_image.setdefault(stem, []).append(sc)
    except Exception as exc:
        logger.warning("Could not load scores for per-image stats: %s", exc)
        return {}
    out: dict[str, dict[str, float]] = {}
    for stem, scores in by_image.items():
        out[stem] = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "count": len(scores),
        }
    return out


def _resolve_dataset_tag(explicit: str | None, input_dir: Path) -> str:
    """Pick the best dataset tag when none is passed explicitly.

    Order of preference:
      1. Caller-supplied tag (already a clean dataset name).
      2. The workspace folder name stripped of the ``lucid_workspace_`` prefix.
      3. ``input_dir.name`` if it is *not* one of the internal stage folders
         (``scored``, ``raw_tiles``).
      4. A generic ``dataset`` placeholder so the filenames stay valid.
    """
    if explicit:
        return explicit
    workspace_name = input_dir.parent.name
    if workspace_name.startswith("lucid_workspace_"):
        return workspace_name[len("lucid_workspace_"):]
    if input_dir.name not in ("scored", "raw_tiles"):
        return input_dir.name
    return "dataset"


def _tag_for_tile(tile_path: Path, fallback: str) -> str:
    """Recover the source-image stem for a tile, falling back when ambiguous.

    Tile names follow ``{img.stem}_s{scale}_t{y}_{x}.png``; for a 512-px
    single-scale run the prefix up to ``_s100`` is the source image stem.
    If the stem is missing or uninformative we use the per-stage fallback
    so the lineage CSV never records ``scored`` as a dataset name.
    """
    stem = tile_path.stem
    if "_s" in stem:
        candidate = stem.split("_s", 1)[0].strip()
        if candidate and candidate.lower() not in {"scored", "raw_tiles"}:
            return candidate
    return fallback


def _get_last_index(lineage_path: Path) -> int:
    """Read lineage CSV to find the highest sequential ID."""
    if not lineage_path.exists():
        return 0

    last_id = 0
    try:
        with open(lineage_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last_id = max(last_id, int(row["id"]))
                except (ValueError, KeyError):
                    continue
    except Exception as exc:
        logger.warning("Could not read lineage for index: %s", exc)
    return last_id
