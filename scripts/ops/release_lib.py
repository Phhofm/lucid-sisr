"""Shared helpers: DINOv2 embeddings, constrained farthest-point selection,
LR generation, packing/upload, contact sheets."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
from PIL import Image

HF = "/home/phhofm/.local/bin/hf"
PER_SHARD = 2000


def embed(paths: list[Path], batch: int = 8) -> np.ndarray:
    """DINOv2 ViT-B/14 embeddings (semantic diversity), L2-normalized."""
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0, img_size=518)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    cfg = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i : i + batch]:
                with Image.open(p) as im:
                    imgs.append(tf(im.convert("RGB")))
            x = torch.stack(imgs).to(device)
            if device == "cuda":
                x = x.half()
            f = model(x).float()
            vecs.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.vstack(vecs).astype(np.float32)


def select(emb: np.ndarray, sources: list[str], target: int,
           cap: int = 1, min_sim: float = 0.90, seed: int = 0) -> list[int]:
    """Farthest-point selection with a per-source cap and a similarity ceiling.

    Each pick is the candidate farthest from everything already selected,
    rejecting candidates that are > min_sim similar to any selected tile.
    """
    n = len(emb)
    if n == 0:
        return []
    x = torch.from_numpy(emb)
    if torch.cuda.is_available():
        x = x.cuda()
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    counts: dict[str, int] = {}
    max_sim = torch.full((n,), -1.0, device=x.device)
    blocked = torch.zeros(n, dtype=torch.bool, device=x.device)

    first = int(rng.integers(n))
    chosen.append(first)
    counts[sources[first]] = counts.get(sources[first], 0) + 1
    blocked[first] = True

    while len(chosen) < target:
        if chosen:
            sims = x @ x[chosen[-1]]
            max_sim = torch.maximum(max_sim, sims)
            max_sim[blocked] = 2.0
        # source cap
        capped = torch.tensor([counts.get(s, 0) >= cap for s in sources],
                              dtype=torch.bool, device=x.device)
        eligible = (~capped) & (~blocked) & (max_sim < min_sim)
        if not bool(eligible.any()):
            break
        masked = torch.where(eligible, max_sim, torch.full_like(max_sim, 2.0))
        idx = int(torch.argmin(masked))
        chosen.append(idx)
        blocked[idx] = True
        counts[sources[idx]] = counts.get(sources[idx], 0) + 1
    return chosen


def make_lr(src: Path, dst: Path, factor: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    size = 512 // factor
    for p in src.glob("*.png"):
        with Image.open(p) as im:
            im.convert("RGB").resize((size, size), Image.BICUBIC).save(dst / p.name)


def pack_upload(set_dir: Path, repo: str, subdir: str, staging_root: Path) -> int:
    files = sorted(set_dir.glob("*.png"))
    staging = staging_root / f"_stage_{subdir.replace('/', '_')}"
    staging.mkdir(parents=True, exist_ok=True)
    n = 0
    for start in range(0, len(files), PER_SHARD):
        group = files[start : start + PER_SHARD]
        idx = start // PER_SHARD
        lst = staging / f"list-{idx:05d}.txt"
        lst.write_text("\n".join(p.name for p in group) + "\n")
        tar = staging / f"shard-{idx:05d}.tar"
        subprocess.run(["tar", "-cf", str(tar), "-C", str(set_dir), "-T", str(lst)],
                       check=True)
        lst.unlink()
        upload(tar, repo, f"{subdir}/{tar.name}")
        tar.unlink()
        n += 1
        print(f"  {subdir}: shard {idx:05d} uploaded", flush=True)
    staging.rmdir()
    return n


def upload(local: Path, repo: str, dest: str, tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        if subprocess.run([HF, "upload", repo, str(local), dest,
                           "--repo-type", "dataset"]).returncode == 0:
            return
        print(f"retry {attempt} {dest}", flush=True)
        time.sleep(20 * attempt)
    raise RuntimeError(f"upload failed: {dest}")


def contact_sheet(set_dir: Path, out: Path, cols: int = 10, rows: int = 6,
                  tile_px: int = 128) -> None:
    files = sorted(set_dir.glob("*.png"))
    if not files:
        return
    rng = np.random.default_rng(7)
    n = min(len(files), cols * rows)
    chosen = [files[i] for i in rng.choice(len(files), size=n, replace=False)]
    img = Image.new("RGB", (cols * tile_px, ((n + cols - 1) // cols) * tile_px),
                    (24, 24, 24))
    for j, p in enumerate(chosen):
        with Image.open(p) as im:
            img.paste(im.convert("RGB").resize((tile_px, tile_px)),
                      ((j % cols) * tile_px, (j // cols) * tile_px))
    img.save(out)
    print("contact sheet:", out, f"({n} tiles)")