"""Content-balanced validation set builder.

Stage 1 (--report): classify the native pool with CLIP and print the
content distribution.
Stage 2 (--build): select with a per-category cap + DINOv2 diversity,
rebuild gt/x2/x4 + val_scores.csv (with category), upload.

Pool:
  - PD12M native s100 tiles, source-held-out from v1 training
  - UHD-IQA native 512 crops (already filtered)
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, "/tmp/kilo")
import release_lib as R

BASE = Path("/media/phhofm/SSD/pd12m_v1")
UHD = Path("/media/phhofm/SSD/uhdiqa_lucid")
OUT = Path("/media/phhofm/SSD/lucid_val_512")
REPO = "Phips/lucid-val-512"
CAP_PER_CATEGORY = 20
CLUSTERS = 60
CAP_PER_CLUSTER = 4
GREEN_THR = 0.30
GREEN_CAP_FIXED = 13
MIN_SIM = 0.90

CATEGORIES = [
    "grass or leaves close-up",
    "tree or forest",
    "flowers or plants",
    "landscape with mountains or fields",
    "sky, clouds or sunset",
    "building, architecture or city street",
    "indoor room or interior",
    "person or portrait",
    "animal",
    "food or drink",
    "vehicle, car or boat",
    "object or product on plain background",
    "text, document or screenshot",
    "texture or abstract pattern",
    "water, sea or river",
]

SHORT = {c: c.split(",")[0].split(" or ")[0] for c in CATEGORIES}


def classify(paths: list[Path], batch: int = 16) -> list[str]:
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    prompts = [f"a photo of {c}" for c in CATEGORIES]
    text = proc(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        t_out = model.get_text_features(**text)
        tfeat = t_out.pooler_output if hasattr(t_out, "pooler_output") else t_out
        if tfeat.shape[-1] == model.text_projection.in_features:
            tfeat = model.text_projection(tfeat)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i : i + batch]:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            inp = proc(images=imgs, return_tensors="pt").to(device)
            i_out = model.get_image_features(**inp)
            f = i_out.pooler_output if hasattr(i_out, "pooler_output") else i_out
            if f.shape[-1] == model.visual_projection.in_features:
                f = model.visual_projection(f)
            f = f / f.norm(dim=-1, keepdim=True)
            sims = f @ tfeat.T
            for j in range(len(imgs)):
                out.append(CATEGORIES[int(sims[j].argmax())])
    return out


def _signal(path: str):
    import cv2

    sys.path.insert(0, "/home/phhofm/Documents/GitHub/lucid-sisr")
    from lucid.core.metrics import measure_all

    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    m, _ = measure_all(g)
    return Path(path).name, m


def green_frac(path: Path) -> float:
    import cv2
    import numpy as np

    im = cv2.imread(str(path))
    b, g, r = [c.astype(np.int16) for c in cv2.split(im)]
    return float(((g > r + 12) & (g > b + 12) & (g > 40)).mean())


def kmeans(emb, k: int, iters: int = 25, seed: int = 0):
    import torch

    x = torch.from_numpy(emb)
    if torch.cuda.is_available():
        x = x.cuda()
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:k].to(x.device)
    c = x[idx].clone()
    for _ in range(iters):
        d = x @ c.T
        assign = d.argmax(dim=1)
        for j in range(k):
            m = assign == j
            if bool(m.any()):
                c[j] = torch.nn.functional.normalize(x[m].mean(dim=0), dim=0)
    return assign.cpu().numpy()


def build_pool() -> list[dict]:
    inode = {p.stat().st_ino: p.name for p in (BASE / "tiles").iterdir()}
    lin = {r["final_name"]: r["original_name"]
           for r in csv.DictReader(open(BASE / "master_lineage.csv"))}
    rows = []
    for setdir, csvname in (("pd12m_bench", "pd12m_bench_scores.csv"),
                            ("pd12m_trainval", "pd12m_trainval_scores.csv")):
        meta = {r["tile"]: r for r in csv.DictReader(open(BASE / csvname))}
        for p in (BASE / setdir).iterdir():
            orig = lin.get(inode.get(p.stat().st_ino), "")
            if "_s100_" not in orig or p.name not in meta:
                continue
            m = meta[p.name]
            rows.append({"path": p, "src": "pd12m:" + m["source_id"], "dom": "pd12m",
                         "complexity": float(m["complexity"]),
                         "topiq": float(m["topiq_nr"]), "clipiqa": float(m["clipiqa+"]),
                         "niqe": float(m["niqe"]), "qrealign": float(m["qrealign"])})
    us = {r["tile_name"]: r
          for r in csv.DictReader(open(UHD / "ws" / "scores" / "ingest_scores.csv"))}
    for r in csv.DictReader(open(UHD / "master_lineage.csv")):
        sc = us.get(r["original_name"])
        if sc is None:
            continue
        rows.append({"path": UHD / "tiles" / r["final_name"],
                     "src": "uhd:" + r["original_name"].split("_s")[0], "dom": "uhdiqa",
                     "complexity": float(sc["score_calibrated"]),
                     "topiq": float(sc.get("iqa_topiq_nr") or 0),
                     "clipiqa": float(sc.get("iqa_clipiqa+") or 0),
                     "niqe": float(sc.get("iqa_niqe") or 0),
                     "qrealign": float(sc.get("iqa_qrealign-mini") or 0)})
    return rows


def main() -> None:
    build = "--build" in sys.argv
    rows = build_pool()
    print(f"pool: {len(rows)} tiles / {len({r['src'] for r in rows})} sources")
    cats = classify([r["path"] for r in rows])
    for r, c in zip(rows, cats):
        r["cat"] = c
    dist = Counter(cats)
    print("\ncontent distribution (pool):")
    for c, n in dist.most_common():
        print(f"  {n:4d}  {c}")

    emb = R.embed([r["path"] for r in rows], batch=16)

    # green budget + visual clusters
    greens = [green_frac(r["path"]) for r in rows]
    for r, gf in zip(rows, greens):
        r["green"] = gf
    clusters = kmeans(emb, CLUSTERS, seed=0)
    n = len(rows)
    green_cap = GREEN_CAP_FIXED

    x = torch.from_numpy(emb)
    if torch.cuda.is_available():
        x = x.cuda()
    max_sim = torch.full((n,), -1.0, device=x.device)
    blocked = torch.zeros(n, dtype=torch.bool, device=x.device)
    used_src: set[str] = set()
    cat_count: dict[str, int] = {}
    clus_count: dict[int, int] = {}
    green_count = 0
    order = []
    while True:
        src_ok = torch.tensor([rows[i]["src"] not in used_src for i in range(n)], device=x.device)
        cat_ok = torch.tensor([cat_count.get(rows[i]["cat"], 0) < CAP_PER_CATEGORY for i in range(n)], device=x.device)
        clus_ok = torch.tensor([clus_count.get(int(clusters[i]), 0) < CAP_PER_CLUSTER for i in range(n)], device=x.device)
        green_ok = torch.tensor([not (rows[i]["green"] > GREEN_THR and green_count >= green_cap) for i in range(n)], device=x.device)
        eligible = (~blocked) & src_ok & cat_ok & clus_ok & green_ok & (max_sim < MIN_SIM)
        if not bool(eligible.any()):
            break
        masked = torch.where(eligible, max_sim, torch.full_like(max_sim, 2.0))
        idx = int(torch.argmin(masked))
        order.append(idx)
        blocked[idx] = True
        used_src.add(rows[idx]["src"])
        cat_count[rows[idx]["cat"]] = cat_count.get(rows[idx]["cat"], 0) + 1
        cl = int(clusters[idx])
        clus_count[cl] = clus_count.get(cl, 0) + 1
        if rows[idx]["green"] > GREEN_THR:
            green_count += 1
        sims = x @ x[idx]
        max_sim = torch.maximum(max_sim, sims)
        max_sim[blocked] = 2.0
    picked = sorted(order)
    sel = [rows[i] for i in picked]
    print(f"green-dominant (>30%) selected: {green_count} (cap {green_cap})")
    print(f"clusters used: {len({int(clusters[i]) for i in picked})} of {CLUSTERS}")
    print(f"\nselected {len(sel)} tiles / {len({r['src'] for r in sel})} sources")
    print("composition (selected):")
    for c, n in Counter(r["cat"] for r in sel).most_common():
        print(f"  {n:4d}  {c}")

    if not build:
        return

    gt = OUT / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    for old in gt.glob("*.png"):
        old.unlink()
    ordered = sorted(sel, key=lambda r: (r["cat"], r["src"]))
    names = [str(r["path"]) for r in ordered]
    with ProcessPoolExecutor(max_workers=10) as pool:
        metrics = dict(pool.map(_signal, names, chunksize=16))
    with open(OUT / "val_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "source_id", "domain", "category", "green_fraction",
                    "complexity", "entropy", "lap_var", "grad_energy", "blockiness",
                    "aliasing", "clipped", "topiq_nr", "clipiqa+", "niqe", "qrealign"])
        for i, r in enumerate(ordered, 1):
            new = f"{i:06d}.png"
            try:
                os.link(r["path"], gt / new)
            except OSError:
                shutil.copy2(r["path"], gt / new)
            sig = metrics[r["path"].name]
            w.writerow([new, r["src"].split(":", 1)[1], r["dom"], r["cat"],
                        f"{r['green']:.3f}",
                        f"{r['complexity']:.4f}", *[f"{v:.3f}" for v in sig],
                        f"{r['topiq']:.4f}", f"{r['clipiqa']:.4f}",
                        f"{r['niqe']:.4f}", f"{r['qrealign']:.4f}"])
    R.make_lr(gt, OUT / "x2", 2)
    R.make_lr(gt, OUT / "x4", 4)
    R.contact_sheet(gt, OUT / "contact_val.png")
    import subprocess

    subprocess.run([R.HF, "repo", "create", REPO, "--repo-type", "dataset",
                    "--private"], check=False)
    for sub in ("gt", "x2", "x4"):
        R.pack_upload(OUT / sub, REPO, sub, OUT)
    for f in ("val_scores.csv", "contact_val.png"):
        R.upload(OUT / f, REPO, f)
    print("REBUILT:", REPO)


if __name__ == "__main__":
    main()