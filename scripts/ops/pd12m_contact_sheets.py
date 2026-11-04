"""Build contact sheets + complexity histogram for the published sets."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/media/phhofm/SSD/pd12m_v1")
SETS = ["pd12m_train_full", "pd12m_train_small", "pd12m_train_mini",
        "pd12m_bench", "pd12m_trainval"]
TILE_PX = 128
COLS = 10
ROWS = 6


def sheet(set_name: str) -> None:
    d = OUT / set_name
    files = sorted(d.glob("*.png"))
    if not files:
        return
    rng = np.random.default_rng(7)
    n = min(len(files), COLS * ROWS)
    chosen = [files[i] for i in rng.choice(len(files), size=n, replace=False)]
    rows = (n + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * TILE_PX, rows * TILE_PX), (24, 24, 24))
    for j, p in enumerate(chosen):
        with Image.open(p) as im:
            img.paste(im.convert("RGB").resize((TILE_PX, TILE_PX)),
                      ((j % COLS) * TILE_PX, (j // COLS) * TILE_PX))
    img.save(OUT / f"contact_{set_name}.png")
    print("sheet:", f"contact_{set_name}.png", f"({n} tiles)")


def histogram() -> None:
    scores = list(csv.DictReader(open(OUT / "pd12m_train_full_scores.csv")))
    comp = np.array([float(r["complexity"]) for r in scores])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(comp, bins=60, color="#3b82f6", alpha=0.85)
    for name, csv_name, color in (
        ("small p50", "pd12m_train_small_scores.csv", "#10b981"),
        ("mini p90", "pd12m_train_mini_scores.csv", "#f59e0b"),
    ):
        rows = list(csv.DictReader(open(OUT / csv_name)))
        floor = min(float(r["complexity"]) for r in rows)
        ax.axvline(floor, color=color, label=f"{name} = {floor:.3f}")
    ax.set_xlabel("ICNet complexity (calibrated)")
    ax.set_ylabel("tiles")
    ax.set_title("PD12M filtered tiles — complexity distribution and tier floors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "complexity_histogram.png", dpi=120)
    print("histogram: complexity_histogram.png")


if __name__ == "__main__":
    for s in SETS:
        sheet(s)
    histogram()
