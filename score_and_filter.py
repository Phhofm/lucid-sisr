#!/usr/bin/env python3
"""Fast score-and-filter: GPU batched clipiqa with parallel I/O.

Scores every PNG tile in a dataset with clipiqa, removes tiles below threshold.
Uses GPU batching (64 images at once) + parallel I/O (2 workers) for ~300 t/s
on an RTX 3060, which is ~14x faster than sequential per-image scoring.

Usage:
    python score_and_filter.py <dataset_dir> [threshold] [batch_size]

Examples:
    python score_and_filter.py /data/my_dataset 0.3
    python score_and_filter.py /data/my_dataset 0.3 128

Requirements:
    pip install pyiqa torch opencv-python numpy
"""
import sys, csv, time, os, torch, cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pyiqa


class TileDataset(Dataset):
    """Load PNG tiles as tensors for batched GPU inference."""
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(str(p))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t


def log(msg):
    print(msg, flush=True)


def main():
    if len(sys.argv) < 2:
        log("Usage: python score_and_filter.py <dataset_dir> [threshold] [batch_size]")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    # Lower CPU priority to avoid freezing the system
    try:
        os.nice(10)
    except OSError:
        pass

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cpu':
        log("WARNING: No CUDA GPU found. Scoring will be slow.")
        batch_size = min(batch_size, 8)

    tiles = sorted(dataset_dir.rglob('*.png'))
    log(f"Found {len(tiles)} tiles, batch_size={batch_size}, threshold>={threshold}")

    metric = pyiqa.create_metric('clipiqa', device=DEVICE)
    log("clipiqa loaded")

    ds = TileDataset(tiles)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2,
                        pin_memory=True, persistent_workers=True, drop_last=False)

    # Warmup (first batch)
    warmup_img = cv2.imread(str(tiles[0]))
    warmup_t = torch.from_numpy(warmup_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    _ = metric(warmup_t.to(DEVICE))
    torch.cuda.synchronize()
    log("warmup done")

    results = []
    t0 = time.time()
    scored = 0

    with torch.no_grad():
        for batch_tensors in loader:
            batch = batch_tensors.to(DEVICE, non_blocking=True)
            scores = metric(batch).squeeze(-1)
            torch.cuda.synchronize()

            for j in range(len(batch_tensors)):
                results.append((str(tiles[scored + j].relative_to(dataset_dir)), scores[j].item()))

            scored += len(batch_tensors)
            if scored % (batch_size * 20) == 0 or scored >= len(tiles):
                elapsed = time.time() - t0
                tps = scored / elapsed
                eta = (len(tiles) - scored) / tps / 60
                log(f"  [{scored}/{len(tiles)}] {tps:.1f} t/s, ETA {eta:.0f}min")

    elapsed = time.time() - t0
    log(f"\nScored {len(tiles)} tiles in {elapsed:.0f}s ({len(tiles)/elapsed:.1f} t/s)")

    # Save scores
    scores_csv = dataset_dir / 'clipiqa_scores.csv'
    with open(scores_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rel_path', 'clipiqa_score'])
        for rel, score in results:
            w.writerow([rel, score])

    scores = [s for _, s in results]
    scores.sort()
    log(f"\nDistribution:")
    log(f"  Min: {scores[0]:.4f}, P10: {scores[int(len(scores)*0.1)]:.4f}, "
        f"Median: {scores[len(scores)//2]:.4f}, P90: {scores[int(len(scores)*0.9)]:.4f}, "
        f"Max: {scores[-1]:.4f}")

    # Filter and remove
    keep = [(f, s) for f, s in results if s >= threshold]
    remove = [(f, s) for f, s in results if s < threshold]

    log(f"\nKeep (>= {threshold}): {len(keep)} tiles")
    log(f"Remove (< {threshold}): {len(remove)} tiles")

    removed_count = 0
    for rel, score in remove:
        p = dataset_dir / rel
        if p.exists():
            os.remove(p)
            removed_count += 1

    log(f"Removed {removed_count} tiles")
    log(f"Remaining: {len(keep)} tiles")

    removed_csv = dataset_dir / 'clipiqa_removed.csv'
    with open(removed_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rel_path', 'clipiqa_score'])
        for rel, score in sorted(remove, key=lambda x: x[1]):
            w.writerow([rel, score])

    # Clean up empty subdirectories
    for d in sorted(dataset_dir.rglob('*'), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    log(f"Done. Scores: {scores_csv}, Removed: {removed_csv}")


if __name__ == '__main__':
    main()
