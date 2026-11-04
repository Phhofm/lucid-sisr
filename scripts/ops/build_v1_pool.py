"""Compute ResNet-18 embedding pool of the v1 gt tiles (for v2 seeding)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
from lucid.core import dedupe

BASE = Path("/media/phhofm/SSD/pd12m_v1")
SRC = BASE / "pd12m_train_full"
OUT = BASE / "v1_pool.npy"


def main() -> None:
    files = sorted(SRC.glob("*.png"))
    print(f"tiles: {len(files)}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dedupe._get_feature_extractor(device)
    prep = dedupe._preload_transform()
    half = device == "cuda"
    chunks = []
    with torch.no_grad():
        for i in range(0, len(files), 128):
            batch = []
            for p in files[i : i + 128]:
                with Image.open(p) as im:
                    batch.append(prep(im.convert("RGB")))
            x = torch.stack(batch).to(device)
            if half:
                x = x.half()
            f = model(x).squeeze().float()
            if f.dim() == 1:
                f = f.unsqueeze(0)
            f = torch.nn.functional.normalize(f, dim=1)
            chunks.append(f.cpu().numpy())
            if (i // 128) % 100 == 0:
                print(f"{i}/{len(files)}", flush=True)
    pool = np.vstack(chunks).astype(np.float32)
    np.save(OUT, pool)
    print("POOL:", pool.shape, "->", OUT, flush=True)


if __name__ == "__main__":
    main()