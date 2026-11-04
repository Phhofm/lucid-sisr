"""Build the single canonical validation set: native tiles from PD12M + UHD-IQA.

- PD12M pool: s100 (native) tiles that were source-held-out from v1 training
- UHD-IQA pool: native 512 crops from 4K photos (already filtered)
- one tile per source, DINOv2 farthest-point, max similarity 0.90
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/tmp/kilo")
import release_lib as R

BASE = Path("/media/phhofm/SSD/pd12m_v1")
UHD = Path("/media/phhofm/SSD/uhdiqa_lucid")
OUT = Path("/media/phhofm/SSD/lucid_val_512")
REPO = "Phips/lucid-val-512"


def _signal(path: str):
    import cv2

    sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core.metrics import measure_all

    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    m, _ = measure_all(g)
    return Path(path).name, m


def main() -> None:
    inode = {p.stat().st_ino: p.name for p in (BASE / "tiles").iterdir()}
    lin = {r["final_name"]: r["original_name"]
           for r in csv.DictReader(open(BASE / "master_lineage.csv"))}

    rows = []
    for setdir, csvname in (("pd12m_bench", "pd12m_bench_scores.csv"),
                            ("pd12m_trainval", "pd12m_trainval_scores.csv")):
        meta = {r["tile"]: r for r in csv.DictReader(open(BASE / csvname))}
        for p in (BASE / setdir).iterdir():
            orig = lin.get(inode.get(p.stat().st_ino), "")
            if "_s100_" not in orig or p.name not in meta:
                continue
            m = meta[p.name]
            rows.append({"path": p, "src": "pd12m:" + m["source_id"], "dom": "pd12m",
                         "complexity": float(m["complexity"]),
                         "topiq": float(m["topiq_nr"]), "clipiqa": float(m["clipiqa+"]),
                         "niqe": float(m["niqe"]), "qrealign": float(m["qrealign"])})

    us = {r["tile_name"]: r
          for r in csv.DictReader(open(UHD / "ws" / "scores" / "ingest_scores.csv"))}
    for r in csv.DictReader(open(UHD / "master_lineage.csv")):
        sc = us.get(r["original_name"])
        if sc is None:
            continue
        rows.append({"path": UHD / "tiles" / r["final_name"],
                     "src": "uhd:" + r["original_name"].split("_s")[0], "dom": "uhdiqa",
                     "complexity": float(sc["score_calibrated"]),
                     "topiq": float(sc.get("iqa_topiq_nr") or 0),
                     "clipiqa": float(sc.get("iqa_clipiqa+") or 0),
                     "niqe": float(sc.get("iqa_niqe") or 0),
                     "qrealign": float(sc.get("iqa_qrealign-mini") or 0)})
    print(f"pool: {len(rows)} native tiles / {len({r['src'] for r in rows})} sources")

    emb = R.embed([r["path"] for r in rows], batch=16)
    idx = R.select(emb, [r["src"] for r in rows], target=len(rows), cap=1,
                   min_sim=0.90, seed=42)
    picked = [rows[i] for i in idx]
    print(f"selected: {len(picked)} tiles / {len({r['src'] for r in picked})} sources")

    gt = OUT / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for old in gt.glob("*.png"):
        old.unlink()

    ordered = sorted(picked, key=lambda r: (r["dom"], r["src"]))
    names = [str(r["path"]) for r in ordered]
    with ProcessPoolExecutor(max_workers=10) as pool:
        metrics = dict(pool.map(_signal, names, chunksize=16))

    with open(OUT / "val_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "source_id", "domain", "complexity", "entropy", "lap_var",
                    "grad_energy", "blockiness", "aliasing", "clipped",
                    "topiq_nr", "clipiqa+", "niqe", "qrealign"])
        for i, r in enumerate(ordered, 1):
            new = f"{i:06d}.png"
            try:
                os.link(r["path"], gt / new)
            except OSError:
                shutil.copy2(r["path"], gt / new)
            sig = metrics[r["path"].name]
            w.writerow([new, r["src"].split(":", 1)[1], r["dom"],
                        f"{r['complexity']:.4f}", *[f"{v:.3f}" for v in sig],
                        f"{r['topiq']:.4f}", f"{r['clipiqa']:.4f}",
                        f"{r['niqe']:.4f}", f"{r['qrealign']:.4f}"])

    R.make_lr(gt, OUT / "x2", 2)
    R.make_lr(gt, OUT / "x4", 4)
    R.contact_sheet(gt, OUT / "contact_val.png")

    import subprocess

    subprocess.run([R.HF, "repo", "create", REPO, "--repo-type", "dataset",
                    "--private"], check=False)
    for sub in ("gt", "x2", "x4"):
        R.pack_upload(OUT / sub, REPO, sub, OUT)
    for f in ("val_scores.csv", "contact_val.png"):
        R.upload(OUT / f, REPO, f)
    print("COMBINED VAL DONE:", REPO)


if __name__ == "__main__":
    main()