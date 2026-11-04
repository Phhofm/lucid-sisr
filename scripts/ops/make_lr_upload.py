"""Generate x2/x4 LR shards from local gt tiles and upload, deleting each
shard after upload. Shard N of x2/x4 aligns with shard N of gt.

Usage: python make_lr_upload.py <src_gt_dir> <repo> [subdir_prefix]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

HF = "/home/phhofm/.local/bin/hf"
PER_SHARD = 2000
FACTORS = (("x2", 2), ("x4", 4))


def run_retry(cmd: list[str], tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"retry {attempt}: {' '.join(cmd[:3])}", flush=True)
        time.sleep(20 * attempt)
    raise RuntimeError(f"failed: {' '.join(cmd[:3])}")


def main() -> None:
    src = Path(sys.argv[1])
    repo = sys.argv[2]
    files = sorted(src.glob("*.png"))
    print(f"{len(files)} gt tiles -> x2/x4", flush=True)

    staging = src.parent / "_lr_staging"
    staging.mkdir(exist_ok=True)

    n_shards = 0
    for start in range(0, len(files), PER_SHARD):
        group = files[start : start + PER_SHARD]
        idx = start // PER_SHARD
        tars = []
        for name, factor in FACTORS:
            outdir = staging / f"{name}_{idx:05d}"
            outdir.mkdir(exist_ok=True)
            size = 512 // factor
            for p in group:
                with Image.open(p) as im:
                    im.convert("RGB").resize((size, size), Image.BICUBIC).save(
                        outdir / p.name, optimize=False)
            lst = staging / f"list_{name}_{idx:05d}.txt"
            lst.write_text("\n".join(p.name for p in group) + "\n")
            tar = staging / f"{name}-{idx:05d}.tar"
            subprocess.run(["tar", "-cf", str(tar), "-C", str(outdir), "-T", str(lst)],
                           check=True)
            lst.unlink()
            tars.append((tar, name))

        for tar, name in tars:
            run_retry([HF, "upload", repo, str(tar),
                       f"{name}/{tar.name}", "--repo-type", "dataset"])
            tar.unlink()
        for name, _ in FACTORS:
            outdir = staging / f"{name}_{idx:05d}"
            for p in outdir.iterdir():
                p.unlink()
            outdir.rmdir()
        n_shards += 1
        print(f"shard {idx:05d} done ({n_shards} of {(len(files)+PER_SHARD-1)//PER_SHARD})",
              flush=True)

    staging.rmdir()
    print(f"DONE: {n_shards} shard pairs / {len(files)} tiles", flush=True)


if __name__ == "__main__":
    main()