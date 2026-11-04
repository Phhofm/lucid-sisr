"""PD12M probe: sample N images from the first metadata parquet, download
from S3, run the full LUCID pipeline (production flags), and report the
per-gate pass funnel.

Usage: python pd12m_probe.py [N]   (default 100)
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from pyarrow import parquet

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REPO = "Spawning/PD12M"
OUT = Path("/media/phhofm/SSD/pd12m_probe")
WORK = OUT / "images"
META = OUT / "metadata"
GATES = ["topiq_nr:0.665", "clipiqa+:0.45", "niqe:6.5", "qrealign-mini:0.78"]
THRESHOLD = "0.30"

OUT.mkdir(parents=True, exist_ok=True)
META.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)


def download_parquet() -> Path:
    print("fetching metadata parquet ...", flush=True)
    return Path(hf_hub_download(REPO, "metadata/pd12m.000.parquet",
                                repo_type="dataset", local_dir=str(META)))


def fetch_one(row) -> str | None:
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        row["mime_type"], "jpg")
    dst = WORK / f"{row['id']}.{ext}"
    if dst.exists():
        return str(dst)
    try:
        r = requests.get(row["url"], timeout=30)
        if r.status_code == 200 and len(r.content) > 10_000:
            dst.write_bytes(r.content)
            return str(dst)
    except Exception:
        pass
    return None


def funnel(csv_path: Path, n_images: int) -> None:
    gates = ["topiq_nr", "clipiqa+", "niqe", "qrealign-mini"]
    thresholds = {"topiq_nr": (0.665, True), "clipiqa+": (0.45, True),
                  "niqe": (6.5, False), "qrealign-mini": (0.78, True)}
    rows = list(csv.DictReader(open(csv_path)))
    print(f"\n=== funnel: {n_images} images, {len(rows)} tiles scored ===")
    alive = rows
    for g in gates:
        col = f"iqa_{g}"
        thr, higher = thresholds[g]
        reached = [r for r in alive if float(r[col]) > 0.0]
        passed = [r for r in reached
                  if (float(r[col]) >= thr if higher else float(r[col]) <= thr)]
        scores = sorted(float(r[col]) for r in reached)
        med = scores[len(scores) // 2] if scores else 0.0
        print(f" {g:14s} reached {len(reached):5d}  passed {len(passed):5d} "
              f"({100*len(passed)/max(len(reached),1):5.1f}%)  median {med:.3f}")
        alive = passed
    print(f" FINAL: {len(alive)} tiles from {len(rows)} candidates")


def main() -> None:
    pq = download_parquet()
    table = parquet.read_table(pq)
    print("columns:", table.column_names, "| rows:", table.num_rows, flush=True)
    df = table.to_pandas().sample(n=min(N, table.num_rows), random_state=0)
    print(f"sampled {len(df)} urls; downloading ...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        paths = [p for p in pool.map(fetch_one, df.to_dict("records")) if p]
    print(f"downloaded {len(paths)} images in {time.time()-t0:.0f}s", flush=True)
    if not paths:
        print("no images downloaded; aborting")
        return

    cmd = [sys.executable, "-m", "lucid", "--input", str(WORK),
           "--output", str(OUT / "out"), "--workspace", str(OUT / "ws"),
           "--stage", "all", "--threshold", THRESHOLD,
           "--dedupe-threshold", "0.96"]
    for g in GATES:
        cmd += ["--iqa-gate", g]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd="/home/phhofm/Documents/GitHub/lucid-sisr")
    print(f"lucid exit={r.returncode} in {time.time()-t0:.0f}s", flush=True)
    csv_path = OUT / "ws" / "scores" / "ingest_scores.csv"
    if csv_path.exists():
        funnel(csv_path, len(paths))
    tiles = OUT / "out" / "tiles"
    if tiles.exists():
        print("final tiles:", len(list(tiles.glob("*.png"))), flush=True)


if __name__ == "__main__":
    main()
