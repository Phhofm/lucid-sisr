"""QA-metric test harness: T1 (val125 reference band) + T2 (degradation
sensitivity matrix) for topiq_nr and qrealign-mini."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch

REF = Path("/media/phhofm/SSD/psisrd_val125/psisrd_val125/hr")
OUT = Path("/tmp/kilo/qa_probe")
OUT.mkdir(exist_ok=True)
METRICS = ["topiq_nr", "qrealign-mini"]
VARIANTS = ["clean", "jpeg30", "jpeg75", "blur9", "blur15", "noise15", "sharpen"]


def degrade(bgr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "clean":
        return bgr
    if kind.startswith("jpeg"):
        q = int(kind[4:])
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind.startswith("blur"):
        s = float(kind[4:])
        return cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), s).clip(0, 255).astype(np.uint8)
    if kind.startswith("noise"):
        s = float(kind[5:])
        rng = np.random.default_rng(4)
        return (bgr.astype(np.float32) + rng.normal(0, s, bgr.shape)).clip(0, 255).astype(np.uint8)
    if kind == "sharpen":
        b = bgr.astype(np.float32)
        return (b + 2.0 * (b - cv2.GaussianBlur(b, (5, 5), 0))).clip(0, 255).astype(np.uint8)
    raise ValueError(kind)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {m: pyiqa.create_metric(m, device=device) for m in METRICS}
    files = sorted(REF.glob("*.png"))

    # T1: reference band on all 125 clean tiles
    t1: dict[str, list[float]] = {m: [] for m in METRICS}
    with open(OUT / "T1_val125.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "metric", "score"])
        for p in files:
            for m in METRICS:
                s = float(models[m](str(p)))
                t1[m].append(s)
                w.writerow([p.name, m, f"{s:.4f}"])
    for m in METRICS:
        v = np.array(t1[m])
        q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        print(f"T1 {m}: n={len(v)} p01={q[0]:.3f} p05={q[1]:.3f} p25={q[2]:.3f} "
              f"p50={q[3]:.3f} p75={q[4]:.3f} p95={q[5]:.3f} p99={q[6]:.3f}", flush=True)

    # T2: degradation sensitivity matrix (20 tiles x 7 variants)
    rng = np.random.default_rng(2)
    t2_files = rng.choice(files, size=20, replace=False).tolist()
    tmp = OUT / "_t2.png"
    mat: dict[str, dict[str, list[float]]] = {v: {m: [] for m in METRICS} for v in VARIANTS}
    for kind in VARIANTS:
        for p in t2_files:
            cv2.imwrite(str(tmp), degrade(cv2.imread(str(p)), kind))
            for m in METRICS:
                mat[kind][m].append(float(models[m](str(tmp))))
        cells = "  ".join(
            f"{m}: p05={np.percentile(mat[kind][m], 5):.3f} mean={np.mean(mat[kind][m]):.3f}"
            for m in METRICS
        )
        print(f"T2 {kind}: {cells}", flush=True)
    with open(OUT / "T2_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", f"{METRICS[0]}_p05", f"{METRICS[0]}_mean",
                    f"{METRICS[1]}_p05", f"{METRICS[1]}_mean"])
        for kind in VARIANTS:
            w.writerow([kind]
                       + [f"{np.percentile(np.array(mat[kind][m]), 5):.4f}"
                          + f", {np.mean(mat[kind][m]):.4f}" for m in METRICS])
    print("wrote", OUT / "T2_matrix.csv", flush=True)


if __name__ == "__main__":
    main()
