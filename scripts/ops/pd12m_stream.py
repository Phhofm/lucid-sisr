"""Rolling PD12M driver: 2% uniform sample per metadata parquet.

Per parquet: sample N rows -> download from S3 -> lucid ingest -> delete
images. Keeps only scored tiles. Resumable via done_parquets.json.
Stops at a tile budget, then runs one dedupe + finalize.

Env:
  PD12M_PER_PARQUET  images sampled per parquet (default 2000)
  PD12M_BUDGET       stop when scored tiles >= this (default 200000)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from pyarrow import parquet

REPO = "Spawning/PD12M"
OUT = Path("/media/phhofm/SSD/pd12m_v1")
WS = OUT / "workspace"
WORK = OUT / "work"
IMAGES = WORK / "images"
META = OUT / "metadata"
STATE = OUT / "done_parquets.json"
LOG = OUT / "stream.log"

PER_PARQUET = int(os.environ.get("PD12M_PER_PARQUET", "2000"))
BUDGET = int(os.environ.get("PD12M_BUDGET", "200000"))
GATES = ["topiq_nr:0.665", "clipiqa+:0.45", "niqe:6.5", "qrealign-mini:0.78"]
THRESHOLD = "0.30"
RESERVE_GB = 80.0


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def free_gb() -> float:
    return shutil.disk_usage(OUT).free / 1024**3


def scored_count() -> int:
    d = WS / "scored"
    return len(list(d.glob("*.png"))) if d.exists() else 0


def load_state() -> set[str]:
    return set(json.loads(STATE.read_text())) if STATE.exists() else set()


def save_state(done: set[str]) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(done)))
    tmp.rename(STATE)


def fetch_one(row) -> str | None:
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        row["mime_type"], "jpg")
    dst = IMAGES / f"{row['id']}.{ext}"
    for _ in range(2):
        try:
            r = requests.get(row["url"], timeout=30)
            if r.status_code == 200 and len(r.content) > 10_000:
                dst.write_bytes(r.content)
                return str(dst)
        except Exception:
            continue
    return None


def list_parquets() -> list[str]:
    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith("metadata/") and f.endswith(".parquet")]
    return sorted(files)


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(str(c) for c in cmd[:6]) + " ...")
    subprocess.run(cmd, check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    parquets = list_parquets()
    done = load_state()
    log(f"{len(parquets)} parquets, {len(done)} done")

    for idx, pq in enumerate(parquets, 1):
        if pq in done:
            continue
        if scored_count() >= BUDGET:
            log(f"budget reached ({scored_count()} tiles); stopping ingestion")
            break
        if free_gb() < RESERVE_GB:
            log(f"low disk ({free_gb():.0f} GB); stopping ingestion")
            break

        path = Path(hf_hub_download(REPO, pq, repo_type="dataset",
                                    local_dir=str(META)))
        tbl = parquet.read_table(path)
        df = tbl.to_pandas().sample(
            n=min(PER_PARQUET, tbl.num_rows), random_state=idx)
        log(f"parquet {idx}/{len(parquets)}: {len(df)} images sampled")

        with ThreadPoolExecutor(max_workers=12) as pool:
            paths = [p for p in pool.map(fetch_one, df.to_dict("records")) if p]
        log(f"downloaded {len(paths)} images")

        if paths:
            cmd = [sys.executable, "-m", "lucid",
                   "--input", str(IMAGES), "--output", str(OUT),
                   "--workspace", str(WS), "--stage", "ingest",
                   "--threshold", THRESHOLD, "--dedupe-threshold", "0.96"]
            for g in GATES:
                cmd += ["--iqa-gate", g]
            run(cmd)
            scores = WS / "scores" / "ingest_scores.csv"
            if scores.exists():
                scores.rename(WS / "scores" / f"ingest_scores.p{idx:04d}.csv")

        for p in IMAGES.iterdir():
            p.unlink()
        done.add(pq)
        save_state(done)
        log(f"parquet done; scored tiles total: {scored_count()} ({free_gb():.0f} GB free)")

    log("ingestion finished; running dedupe + finalize")
    run([sys.executable, "-m", "lucid", "--input", str(WS / "scored"),
         "--output", str(OUT), "--workspace", str(WS), "--stage", "dedupe"])
    run([sys.executable, "-m", "lucid", "--input", str(WS / "scored"),
         "--output", str(OUT), "--workspace", str(WS), "--stage", "finalize",
         "--simple-names"])
    tiles = OUT / "tiles"
    log(f"FINAL: {len(list(tiles.glob('*.png')))} tiles at {tiles}")


if __name__ == "__main__":
    main()
