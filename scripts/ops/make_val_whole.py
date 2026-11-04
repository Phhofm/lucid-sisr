"""Build the whole-scene validation set from the UHD whole-scene run.

Input: LUCID output for short-side-512 center crops (1 tile per photo).
Selection: one tile per photo (inherent), plus cluster/category/green caps
and 0.90 distinctness.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, "/tmp/kilo")
import release_lib as R
from balance_val import CATEGORIES, green_frac, kmeans, classify

OUT = Path("/media/phhofm/SSD/lucid_val_512")
SRC = Path("/media/phhofm/SSD/uhd_ws/out")
REPO = "Phips/lucid-val-512"
CLUSTERS = 60
CAP_PER_CLUSTER = 5
CAP_PER_CATEGORY = 40
GREEN_THR = 0.30
GREEN_CAP = 40
MIN_SIM = 0.90


def _signal(path: str):
    import cv2

    sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core.metrics import measure_all

    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    m, _ = measure_all(g)
    return Path(path).name, m


def main() -> None:
    scores = {r["tile_name"]: r
              for r in csv.DictReader(open(SRC / "ws/scores/ingest_scores.csv"))}
    rows = []
    for r in csv.DictReader(open(SRC / "master_lineage.csv")):
        sc = scores.get(r["original_name"])
        if sc is None:
            continue
        src = r["original_name"].split("_s")[0]
        rows.append({"path": SRC / "tiles" / r["final_name"], "src": src,
                     "complexity": float(sc["score_calibrated"]),
                     "topiq": float(sc.get("iqa_topiq_nr") or 0),
                     "clipiqa": float(sc.get("iqa_clipiqa+") or 0),
                     "niqe": float(sc.get("iqa_niqe") or 0),
                     "qrealign": float(sc.get("iqa_qrealign-mini") or 0)})
    print(f"gated source photos: {len(rows)}")
    cats = classify([r["path"] for r in rows])
    for r, c in zip(rows, cats):
        r["cat"] = c
        r["green"] = green_frac(r["path"])
    print("categories:", dict(Counter(cats).most_common()))

    emb = R.embed([r["path"] for r in rows], batch=16)
    clusters = kmeans(emb, CLUSTERS, seed=0)
    n = len(rows)
    x = torch.from_numpy(emb).cuda()
    max_sim = torch.full((n,), -1.0, device="cuda")
    blocked = torch.zeros(n, dtype=torch.bool, device="cuda")
    used_src: set[str] = set()
    cc: dict[str, int] = {}
    cl: dict[int, int] = {}
    green_count = 0
    picked: list[int] = []
    while True:
        src_ok = torch.tensor([rows[i]["src"] not in used_src for i in range(n)], device="cuda")
        cat_ok = torch.tensor([cc.get(rows[i]["cat"], 0) < CAP_PER_CATEGORY for i in range(n)], device="cuda")
        cl_ok = torch.tensor([cl.get(int(clusters[i]), 0) < CAP_PER_CLUSTER for i in range(n)], device="cuda")
        g_ok = torch.tensor([not (rows[i]["green"] > GREEN_THR and green_count >= GREEN_CAP)
                             for i in range(n)], device="cuda")
        el = (~blocked) & src_ok & cat_ok & cl_ok & g_ok & (max_sim < MIN_SIM)
        if not bool(el.any()):
            break
        masked = torch.where(el, max_sim, torch.full_like(max_sim, 2.0))
        idx = int(torch.argmin(masked))
        picked.append(idx)
        blocked[idx] = True
        used_src.add(rows[idx]["src"])
        cc[rows[idx]["cat"]] = cc.get(rows[idx]["cat"], 0) + 1
        cl[int(clusters[idx])] = cl.get(int(clusters[idx]), 0) + 1
        if rows[idx]["green"] > GREEN_THR:
            green_count += 1
        max_sim = torch.maximum(max_sim, x @ x[idx])
        max_sim[blocked] = 2.0

    sel = [rows[i] for i in sorted(picked)]
    print(f"selected {len(sel)} tiles / {len({r['src'] for r in sel})} photos; green {green_count}")

    gt = OUT / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for old in gt.glob("*.png"):
        old.unlink()
    ordered = sorted(sel, key=lambda r: (r["cat"], r["src"]))
    names = [str(r["path"]) for r in ordered]
    with ProcessPoolExecutor(max_workers=10) as pool:
        metrics = dict(pool.map(_signal, names, chunksize=16))
    with open(OUT / "val_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "source_id", "category", "green_fraction", "complexity",
                    "entropy", "lap_var", "grad_energy", "blockiness", "aliasing",
                    "clipped", "topiq_nr", "clipiqa+", "niqe", "qrealign"])
        for i, r in enumerate(ordered, 1):
            new = f"{i:06d}.png"
            try:
                os.link(r["path"], gt / new)
            except OSError:
                shutil.copy2(r["path"], gt / new)
            sig = metrics[r["path"].name]
            w.writerow([new, r["src"], r["cat"], f"{r['green']:.3f}",
                        f"{r['complexity']:.4f}", *[f"{v:.3f}" for v in sig],
                        f"{r['topiq']:.4f}", f"{r['clipiqa']:.4f}",
                        f"{r['niqe']:.4f}", f"{r['qrealign']:.4f}"])
    R.make_lr(gt, OUT / "x2", 2)
    R.make_lr(gt, OUT / "x4", 4)
    R.contact_sheet(gt, OUT / "contact_val.png")
    for sub in ("gt", "x2", "x4"):
        R.pack_upload(OUT / sub, REPO, sub, OUT)
    for f in ("val_scores.csv", "contact_val.png"):
        R.upload(OUT / f, REPO, f)
    print("WHOLE-SCENE VAL DONE:", REPO)


if __name__ == "__main__":
    main()