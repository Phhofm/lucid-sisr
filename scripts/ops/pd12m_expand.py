"""PD12M expansion driver: takes the NEXT slice of each parquet.

Offset-based state {"metadata/pd12m.000.parquet": 4000} so every run
processes a disjoint, deterministic slice (shuffle rand_state=parquet
index, take [offset:offset+K]). Appends to the same workspace; never
reprocesses a tile. Skips the final dedupe/finalize unless
PD12M_FINALIZE=1 (expansion rounds should not pay that cost).

Env:
  PD12M_PER_PARQUET  images per parquet per round (default 2000)
  PD12M_BUDGET       stop when scored tiles >= this (default 500000)
  PD12M_FINALIZE     "1" to run dedupe+finalize at the end
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

import numpy as np
import requests
from huggingface_hub import hf_hub_download, list_repo_files
from pyarrow import parquet

REPO = "Spawning/PD12M"
OUT = Path(os.environ.get("PD12M_OUT", "/media/phhofm/SSD/pd12m_v1"))
WS = OUT / "workspace"
WORK = OUT / "work"
IMAGES = WORK / "images"
META = OUT / "metadata"
STATE = OUT / "parquet_offsets.json"
LEGACY_STATE = Path(os.environ.get("PD12M_LEGACY_STATE", str(OUT / "done_parquets.json")))
LOG = OUT / "stream.log"

PER_PARQUET = int(os.environ.get("PD12M_PER_PARQUET", "2000"))
BUDGET = int(os.environ.get("PD12M_BUDGET", "500000"))
FINALIZE = os.environ.get("PD12M_FINALIZE", "0") == "1"
SEED_POOL = os.environ.get("PD12M_SEED_POOL")
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


def load_state() -> dict[str, int]:
    if STATE.exists():
        return {k: int(v) for k, v in json.loads(STATE.read_text()).items()}
    if LEGACY_STATE.exists():  # migrate from the boolean driver
        done = json.loads(LEGACY_STATE.read_text())
        return {pq: PER_PARQUET for pq in done}
    return {}


def save_state(state: dict[str, int]) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    state = load_state()

    parquets = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                      if f.startswith("metadata/") and f.endswith(".parquet"))
    log(f"{len(parquets)} parquets; offsets known for {len(state)}")

    for idx, pq in enumerate(parquets, 1):
        if scored_count() >= BUDGET:
            log(f"budget reached ({scored_count()} tiles); stop")
            break
        if free_gb() < RESERVE_GB:
            log(f"low disk ({free_gb():.0f} GB); stop")
            break

        path = Path(hf_hub_download(REPO, pq, repo_type="dataset",
                                    local_dir=str(META)))
        tbl = parquet.read_table(path)
        n = tbl.num_rows
        offset = state.get(pq, 0)
        if offset >= n:
            continue
        perm = np.random.default_rng(idx).permutation(n)
        rows = tbl.to_pandas().iloc[perm[offset : offset + PER_PARQUET]]
        log(f"parquet {idx}/{len(parquets)} offset {offset}: {len(rows)} images")

        with ThreadPoolExecutor(max_workers=12) as pool:
            paths = [p for p in pool.map(fetch_one, rows.to_dict("records")) if p]
        log(f"downloaded {len(paths)} images")

        if paths:
            cmd = [sys.executable, "-m", "lucid",
                   "--input", str(IMAGES), "--output", str(OUT),
                   "--workspace", str(WS), "--stage", "ingest",
                   "--threshold", THRESHOLD, "--dedupe-threshold", "0.96"]
            for g in GATES:
                cmd += ["--iqa-gate", g]
            log("RUN lucid ingest ...")
            subprocess.run(cmd, check=True)
            scores = WS / "scores" / "ingest_scores.csv"
            if scores.exists():
                scores.rename(WS / "scores" / f"ingest_scores.r{offset:06d}_p{idx:04d}.csv")

        for p in IMAGES.iterdir():
            p.unlink()
        state[pq] = offset + PER_PARQUET
        save_state(state)
        log(f"done; scored total {scored_count()} ({free_gb():.0f} GB free)")

    if FINALIZE:
        log("final: dedupe + finalize")
        dedupe_cmd = [sys.executable, "-m", "lucid", "--input", str(WS / "scored"),
                      "--output", str(OUT), "--workspace", str(WS),
                      "--stage", "dedupe"]
        if SEED_POOL:
            dedupe_cmd += ["--seed-pool", SEED_POOL]
            log(f"seeding dedupe with {SEED_POOL}")
        subprocess.run(dedupe_cmd, check=True)
        subprocess.run([sys.executable, "-m", "lucid", "--input", str(WS / "scored"),
                        "--output", str(OUT), "--workspace", str(WS),
                        "--stage", "finalize", "--simple-names"], check=True)
        log(f"FINAL: {len(list((OUT / 'tiles').glob('*.png')))} tiles")


if __name__ == "__main__":
    main()
