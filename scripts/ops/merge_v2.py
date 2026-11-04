"""Merge PD12M v2 tiles + UHD-IQA training tiles into the final finetune tier.

Steps:
  1. collect tiles + scores from both sources
  2. one dedupe pass over the union, seeded with the v1 embedding pool
     (removes near-duplicates of v1 and across the two corpora)
  3. sequential rename 000001.png..
  4. score sheet (domain column), x2/x4 LR, contact sheet, upload

Usage: python merge_v2.py <seed_pool> <repo>
"""
from __future__ import annotations

import csv
import glob
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
sys.path.insert(0, "/tmp/kilo")
from lucid.core import dedupe
import release_lib as R

V2 = Path("/media/phhofm/SSD/pd12m_v2")
UHD = Path("/media/phhofm/SSD/uhd_ws_train/out")
OUT = Path("/media/phhofm/SSD/lucid_v2_final")
STAGE = OUT / "_stage"
GT = OUT / "gt"


def _signal(path: str):
    import cv2

    sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core.metrics import measure_all

    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    m, _ = measure_all(g)
    return Path(path).name, m


def ledger_map(pattern: str) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(f)):
            out[r["tile_name"]] = r
    return out


def collect(root: Path, ledgers: dict[str, dict], domain: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(root / "master_lineage.csv")):
        sc = ledgers.get(r["original_name"])
        if sc is None:
            continue
        rows.append({
            "src": root / "tiles" / r["final_name"],
            "original": r["original_name"],
            "source_id": r["original_name"].split("_s")[0],
            "domain": domain,
            "complexity": float(sc["score_calibrated"]),
            "topiq_nr": float(sc.get("iqa_topiq_nr") or 0),
            "clipiqa+": float(sc.get("iqa_clipiqa+") or 0),
            "niqe": float(sc.get("iqa_niqe") or 0),
            "qrealign": float(sc.get("iqa_qrealign-mini") or 0),
        })
    return rows


def main() -> None:
    seed = torch.from_numpy(np.load(sys.argv[1])).float()
    repo = sys.argv[2]
    import subprocess

    subprocess.run([R.HF, "repo", "create", repo, "--repo-type", "dataset",
                    "--private"], check=False)

    v2_rows = collect(V2, ledger_map(str(V2 / "ws/scores/ingest_scores.*.csv")), "pd12m")
    uhd_rows = collect(UHD, ledger_map(str(UHD / "ws/scores/ingest_scores.csv")), "uhdiqa")
    rows = v2_rows + uhd_rows
    print(f"v2 tiles {len(v2_rows)}, uhd tiles {len(uhd_rows)}, total {len(rows)}")

    GT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    for p in STAGE.glob("*.png"):
        p.unlink()
    for i, r in enumerate(rows):
        dst = STAGE / f"{r['domain']}_{i:07d}.png"
        try:
            os.link(r["src"], dst)
        except OSError:
            shutil.copy2(r["src"], dst)
        r["stage"] = dst

    kept, removed = dedupe.run_dedupe(
        input_dir=STAGE, threshold=0.96, batch_size=64,
        device_preference="auto", seed_embeddings=seed)
    print(f"dedupe: kept {kept}, removed {removed}")

    survivors = {p.name for p in STAGE.glob("*.png")}
    rows = [r for r in rows if r["stage"].name in survivors]
    rows.sort(key=lambda r: (r["domain"], r["source_id"], r["original"]))
    print(f"survivors: {len(rows)}")

    names = [str(r["stage"]) for r in rows]
    with ProcessPoolExecutor(max_workers=10) as pool:
        metrics = dict(pool.map(_signal, names, chunksize=64))

    for old in GT.glob("*.png"):
        old.unlink()
    with open(OUT / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "source_id", "domain", "complexity", "entropy", "lap_var",
                    "grad_energy", "blockiness", "aliasing", "clipped",
                    "topiq_nr", "clipiqa+", "niqe", "qrealign"])
        for i, r in enumerate(rows, 1):
            new = f"{i:06d}.png"
            try:
                os.link(r["stage"], GT / new)
            except OSError:
                shutil.copy2(r["stage"], GT / new)
            sig = metrics[r["stage"].name]
            w.writerow([new, r["source_id"], r["domain"],
                        f"{r['complexity']:.4f}", *[f"{v:.3f}" for v in sig],
                        f"{r['topiq_nr']:.4f}", f"{r['clipiqa+']:.4f}",
                        f"{r['niqe']:.4f}", f"{r['qrealign']:.4f}"])

    R.make_lr(GT, OUT / "x2", 2)
    R.make_lr(GT, OUT / "x4", 4)
    R.contact_sheet(GT, OUT / "contact_v2.png")

    n = len(rows)
    print(f"FINAL: {n} tiles")
    (OUT / "README.md").write_text(f"""---
license: cdla-permissive-2.0
tags: [super-resolution, sisr, training-dataset, finetune]
pipeline_tag: image-to-image
---

# LUCID 512 v2 — final finetune tier

**{n:,} tiles** for the final SISR finetune stage, composed of:
- PD12M round 2 (fresh source slice, disjoint from v1)
- UHD-IQA training split (4K CC0 photos, whole-scene 512)

**Deduped against `lucid-pd12m-512` (v1)**: the dedupe pass was seeded with
all 172,588 v1 tile embeddings, so nothing here near-duplicates v1 content
(cosine < 0.96). It is also deduped internally, across both corpora.

## Layout

```
gt/  512x512
x2/  256x256 bicubic
x4/  128x128 bicubic
scores.csv
```

Shard N of `gt/`, `x2/`, `x4/` shares tile names; `zip()` pairs them.

## Filtering

Same LUCID stack as v1: six signal gates, ICNet complexity >= 0.30,
topiq_nr >= 0.665, clipiqa+ >= 0.45, niqe <= 6.5, qrealign-mini >= 0.78.

`scores.csv` has `domain` (pd12m/uhdiqa) per tile so you can weight or
filter per corpus.
""")
    R.upload(OUT / "README.md", repo, "README.md")
    for sub in ("gt", "x2", "x4"):
        R.pack_upload(OUT / sub, repo, sub, OUT)
    for f in ("scores.csv", "contact_v2.png"):
        R.upload(OUT / f, repo, f)
    print("MERGED V2 DONE:", repo)


if __name__ == "__main__":
    main()