"""Pack one published set into ~1 GB webdataset shards and upload them,
deleting each shard after upload (rolling, bounded disk).

Usage: python pd12m_pack_upload.py <set_name> <repo> [repo_subdir]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/media/phhofm/SSD/pd12m_v1")
HF = "/home/phhofm/.local/bin/hf"
PER_SHARD = 2000  # ~1 GB at ~500 kB/tile


def run_retry(cmd: list[str], tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"retry {attempt}: {' '.join(cmd[:3])}", flush=True)
        time.sleep(20 * attempt)
    raise RuntimeError(f"failed: {' '.join(cmd[:3])}")


def main() -> None:
    set_name = sys.argv[1]
    repo = sys.argv[2]
    subdir = sys.argv[3] if len(sys.argv) > 3 else set_name
    src = OUT / set_name
    files = sorted(src.glob("*.png"))
    if not files:
        print(f"{set_name}: no tiles; skip")
        return
    staging = OUT / "_staging" / set_name
    staging.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(0, len(files), PER_SHARD):
        group = files[i : i + PER_SHARD]
        idx = i // PER_SHARD
        tar_path = staging / f"shard-{idx:05d}.tar"
        lst = staging / f"list-{idx:05d}.txt"
        lst.write_text("\n".join(f.name for f in group) + "\n")
        subprocess.run(["tar", "-cf", str(tar_path), "-C", str(src),
                        "-T", str(lst)], check=True)
        lst.unlink()
        run_retry([HF, "upload", repo, str(tar_path),
                   f"{subdir}/{tar_path.name}", "--repo-type", "dataset"])
        tar_path.unlink()
        n += 1
        print(f"{set_name}: uploaded shard {idx:05d} ({n} total)", flush=True)
    staging.rmdir()
    print(f"DONE {set_name}: {n} shards / {len(files)} tiles", flush=True)


if __name__ == "__main__":
    main()
