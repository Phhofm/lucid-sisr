"""T3: stratified sample of v3-train tiles through topiq_nr + qrealign-mini.

Reports per-metric distributions, reject rates at the candidate cascade
thresholds (topiq >= 0.55, qrealign >= 0.75), and writes the CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyiqa
import torch

BASE = Path("/media/phhofm/SSD/lucid_v3_pxhere")
OUT = Path("/tmp/kilo/qa_probe")
METRICS = ["topiq_nr", "qrealign-mini"]
N_SAMPLE = 2000

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {m: pyiqa.create_metric(m, device=device) for m in METRICS}
    files = sorted((BASE / "tiles_train").glob("*.png"))
    rng = np.random.default_rng(5)
    sample = [files[i] for i in rng.choice(len(files), size=N_SAMPLE, replace=False)]

    rows = []
    vals = {m: [] for m in METRICS}
    for i, p in enumerate(sample):
        r = {m: float(models[m](str(p))) for m in METRICS}
        rows.append([p.name] + [f"{r[m]:.4f}" for m in METRICS])
        for m in METRICS:
            vals[m].append(r[m])
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{N_SAMPLE}", flush=True)

    with open(OUT / "T3_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile"] + [f"{m}" for m in METRICS])
        w.writerows(rows)

    for m in METRICS:
        v = np.array(vals[m])
        q = np.percentile(v, [1, 5, 25, 50, 75, 95])
        print(f"T3 {m}: p01={q[0]:.3f} p05={q[1]:.3f} p25={q[2]:.3f} "
              f"p50={q[3]:.3f} p75={q[4]:.3f} p95={q[5]:.3f}", flush=True)
    top = np.array(vals["topiq_nr"]); rel = np.array(vals["qrealign-mini"])
    both = (top >= 0.55) & (rel >= 0.75)
    print(f"keep@topiq0.55: {float(np.mean(top >= 0.55))*100:.1f}% | "
          f"keep@qrealign0.75: {float(np.mean(rel >= 0.75))*100:.1f}% | "
          f"keep@cascade(both): {float(np.mean(both))*100:.1f}%", flush=True)
    print("T3-DONE", flush=True)


if __name__ == "__main__":
    main()
