"""Post-process the PD12M run into publishable sets.

Reads OUT/tiles + master_lineage.csv + the per-parquet score ledgers, then:

  1. Recompute the six signal metrics on every final tile (fast, parallel).
  2. Join ICNet + IQA scores from the ingestion ledgers by original name.
  3. Hold out whole SOURCES for validation:
       bench-val  2000 tiles (coreset-diverse, farthest-point)
       train-val   100 tiles (coreset-diverse, disjoint sources)
     Leftover tiles of held-out sources are discarded (no train leakage).
  4. Derive complexity tiers from the remaining train pool:
       full  >= 0.30 | small >= p50 | mini >= p90
  5. Write each set as sequential 000001.png.. with a score CSV.

Usage: python pd12m_postprocess.py [--dry-run]
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

OUT = Path("/media/phhofm/SSD/pd12m_v1")
WS = OUT / "workspace"
TILES = OUT / "tiles"
LINEAGE = OUT / "master_lineage.csv"
LEDGERS = sorted((WS / "scores").glob("ingest_scores.*.csv"))
DRY = "--dry-run" in sys.argv

BENCH_N = 2000
TRAINVAL_N = 100
BENCH_POOL_MIN = int(BENCH_N * 1.15)
TRAINVAL_POOL_MIN = int(TRAINVAL_N * 1.3)
SEED = 42


def _signal(path: str) -> tuple[str, list[float]]:
    import cv2
    import sys as _s

    _s.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core.metrics import measure_all

    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    m, _ = measure_all(g)
    return Path(path).name, m


def source_of(original_name: str) -> str:
    return original_name.split("_s")[0]


def load_ledgers() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for f in LEDGERS:
        for r in csv.DictReader(open(f)):
            out[r["tile_name"]] = {
                "complexity": float(r["score_calibrated"]),
                "topiq_nr": float(r.get("iqa_topiq_nr", 0) or 0),
                "clipiqa+": float(r.get("iqa_clipiqa+", 0) or 0),
                "niqe": float(r.get("iqa_niqe", 0) or 0),
                "qrealign": float(r.get("iqa_qrealign-mini", 0) or 0),
            }
    return out


def embed_paths(paths: list[Path]) -> np.ndarray:
    import torch
    from PIL import Image
    import sys as _s

    _s.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core import dedupe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dedupe._get_feature_extractor(device)
    prep = dedupe._preload_transform()
    half = device == "cuda"
    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = []
            for p in paths[i : i + 64]:
                with Image.open(p) as im:
                    batch.append(prep(im.convert("RGB")))
            t = torch.stack(batch).to(device)
            if half:
                t = t.half()
            f = model(t).squeeze().float()
            if f.dim() == 1:
                f = f.unsqueeze(0)
            f = torch.nn.functional.normalize(f, dim=1)
            vecs.append(f.cpu().numpy())
    return np.vstack(vecs)


def farthest_point(vecs: np.ndarray, k: int, seed: int = SEED) -> list[int]:
    import torch

    n = len(vecs)
    k = min(k, n)
    x = torch.from_numpy(vecs)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    min_d = 1.0 - x @ x[chosen[0]]
    for _ in range(k - 1):
        idx = int(torch.argmax(min_d))
        chosen.append(idx)
        d = 1.0 - x @ x[idx]
        min_d = torch.minimum(min_d, d)
    return chosen


def write_set(name: str, rows: list[dict], metrics: dict[str, list[float]]) -> dict:
    dst = OUT / name
    if not DRY:
        dst.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["original_name"])
    with open(OUT / f"{name}_scores.csv" if not DRY else os.devnull, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "source_id", "complexity", "entropy", "lap_var",
                    "grad_energy", "blockiness", "aliasing", "clipped",
                    "topiq_nr", "clipiqa+", "niqe", "qrealign"])
        for i, r in enumerate(rows, 1):
            new = f"{i:06d}.png"
            src = TILES / r["final_name"]
            if not DRY:
                link_or_copy(src, dst / new)
            sig = metrics[src.name]
            sc = r["scores"]
            w.writerow([new, r["source_id"], f"{sc['complexity']:.4f}",
                        *[f"{v:.3f}" for v in sig],
                        f"{sc['topiq_nr']:.4f}", f"{sc['clipiqa+']:.4f}",
                        f"{sc['niqe']:.4f}", f"{sc['qrealign']:.4f}"])
    return {"set": name, "tiles": len(rows)}


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    lineage = list(csv.DictReader(open(LINEAGE)))
    print(f"lineage rows: {len(lineage)}")
    ledgers = load_ledgers()
    print(f"ledger entries: {len(ledgers)}")

    rows = []
    for r in lineage:
        sc = ledgers.get(r["original_name"])
        if sc is None:
            continue
        rows.append({
            "final_name": r["final_name"],
            "original_name": r["original_name"],
            "source_id": source_of(r["original_name"]),
            "scores": sc,
        })
    print(f"joined rows: {len(rows)}")

    # --- signal metrics on final tiles (parallel) ---
    names = [TILES / r["final_name"] for r in rows]
    with ProcessPoolExecutor(max_workers=10) as pool:
        metrics = dict(pool.map(_signal, [str(p) for p in names], chunksize=64))
    print(f"signal metrics computed: {len(metrics)}")

    # --- source-level validation holdout (whole sources, no leakage) ---
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)
    sources = sorted(by_source)
    rng = np.random.default_rng(SEED)
    rng.shuffle(sources)

    bench_sources: set[str] = set()
    bench_count = 0
    it = iter(sources)
    for s in it:
        if bench_count >= BENCH_POOL_MIN:
            break
        bench_sources.add(s)
        bench_count += len(by_source[s])
    tv_sources: set[str] = set()
    tv_count = 0
    for s in it:
        if tv_count >= TRAINVAL_POOL_MIN:
            break
        tv_sources.add(s)
        tv_count += len(by_source[s])

    def pick(pool_rows: list[dict], k: int) -> list[dict]:
        if len(pool_rows) <= k:
            return pool_rows
        vecs = embed_paths([TILES / r["final_name"] for r in pool_rows])
        idx = farthest_point(vecs, k)
        return [pool_rows[i] for i in idx]

    bench_pool = [r for r in rows if r["source_id"] in bench_sources]
    tv_pool = [r for r in rows if r["source_id"] in tv_sources]
    bench = pick(bench_pool, BENCH_N)
    trainval = pick(tv_pool, TRAINVAL_N)
    val_used = {r["final_name"] for r in bench + trainval}

    train_pool = [r for r in rows
                  if r["source_id"] not in bench_sources
                  and r["source_id"] not in tv_sources]
    leftover = len(bench_pool) - len(bench) + len(tv_pool) - len(trainval)

    comp = np.array([r["scores"]["complexity"] for r in train_pool])
    p50, p90 = float(np.percentile(comp, 50)), float(np.percentile(comp, 90))
    small = [r for r in train_pool if r["scores"]["complexity"] >= p50]
    mini = [r for r in train_pool if r["scores"]["complexity"] >= p90]

    summary = {"total_tiles": len(rows), "train_pool": len(train_pool),
               "small_floor_p50": round(p50, 4), "mini_floor_p90": round(p90, 4),
               "leftover_val_source_tiles_discarded": leftover,
               "bench_sources": len(bench_sources), "trainval_sources": len(tv_sources)}
    summary["sets"] = {}
    for name, rows_ in (("pd12m_train_full", train_pool),
                        ("pd12m_train_small", small),
                        ("pd12m_train_mini", mini),
                        ("pd12m_bench", bench),
                        ("pd12m_trainval", trainval)):
        info = write_set(name, rows_, metrics)
        summary["sets"][name] = info["tiles"]
    (OUT / "sets_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
