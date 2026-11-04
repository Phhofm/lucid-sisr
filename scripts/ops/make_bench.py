"""Phase C: build the UHD-IQA benchmark set from a LUCID run.

Inputs: LUCID output (tiles/ + master_lineage.csv) and its workspace
ingest_scores.csv. Selects up to TARGET tiles with DINOv2 constrained
farthest-point (max 2 per source, similarity ceiling 0.90), writes
gt/x2/x4 and the score sheet, uploads to the bench repo.
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/tmp/kilo")
import release_lib as R

LUCID_OUT = Path("/media/phhofm/SSD/uhdiqa_lucid")
WS = LUCID_OUT / "ws"
OUT = Path("/media/phhofm/SSD/lucid_bench_uhdiqa")
REPO = "Phips/lucid-bench-uhdiqa"
TARGET = 2000
CAP = 2
MIN_SIM = 0.90


def _signal(path: str):
    import cv2

    sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core.metrics import measure_all

    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    m, _ = measure_all(g)
    return Path(path).name, m


def main() -> None:
    scores = {r["tile_name"]: r for r in
              csv.DictReader(open(WS / "scores" / "ingest_scores.csv"))}
    print(f"ledger: {len(scores)}")

    rows = []
    for r in csv.DictReader(open(LUCID_OUT / "master_lineage.csv")):
        orig = r["original_name"]
        sc = scores.get(orig)
        if sc is None:
            continue
        rows.append({
            "path": LUCID_OUT / "tiles" / r["final_name"],
            "source": orig.split("_s")[0],
            "complexity": float(sc["score_calibrated"]),
            "topiq_nr": float(sc.get("iqa_topiq_nr") or 0),
            "clipiqa+": float(sc.get("iqa_clipiqa+") or 0),
            "niqe": float(sc.get("iqa_niqe") or 0),
            "qrealign": float(sc.get("iqa_qrealign-mini") or 0),
        })
    print(f"joined: {len(rows)} tiles / {len({r['source'] for r in rows})} sources")

    emb = R.embed([r["path"] for r in rows], batch=16)
    idx = R.select(emb, [r["source"] for r in rows], target=TARGET, cap=CAP,
                   min_sim=MIN_SIM, seed=7)
    picked = [rows[i] for i in idx]
    print(f"selected: {len(picked)} tiles / {len({r['source'] for r in picked})} sources")

    gt = OUT / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for old in gt.glob("*.png"):
        old.unlink()
    with open(OUT / "bench_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "source_id", "complexity", "entropy", "lap_var",
                    "grad_energy", "blockiness", "aliasing", "clipped",
                    "topiq_nr", "clipiqa+", "niqe", "qrealign"])
        names = [str(r["path"]) for r in picked]
        with ProcessPoolExecutor(max_workers=10) as pool:
            metrics = dict(pool.map(_signal, names, chunksize=32))
        for i, r in enumerate(sorted(picked, key=lambda r: r["source"]), 1):
            new = f"{i:06d}.png"
            dst = gt / new
            try:
                os.link(r["path"], dst)
            except OSError:
                import shutil

                shutil.copy2(r["path"], dst)
            sig = metrics[r["path"].name]
            w.writerow([new, r["source"], f"{r['complexity']:.4f}",
                        *[f"{v:.3f}" for v in sig],
                        f"{r['topiq_nr']:.4f}", f"{r['clipiqa+']:.4f}",
                        f"{r['niqe']:.4f}", f"{r['qrealign']:.4f}"])

    R.make_lr(gt, OUT / "x2", 2)
    R.make_lr(gt, OUT / "x4", 4)
    R.contact_sheet(gt, OUT / "contact_bench.png")

    import subprocess

    subprocess.run([R.HF, "repo", "create", REPO, "--repo-type", "dataset",
                    "--private"], check=False)
    for sub in ("gt", "x2", "x4"):
        R.pack_upload(OUT / sub, REPO, sub, OUT)
    for f in ("bench_scores.csv", "contact_bench.png"):
        R.upload(OUT / f, REPO, f)
    print("BENCH BUILD DONE:", REPO)


if __name__ == "__main__":
    main()