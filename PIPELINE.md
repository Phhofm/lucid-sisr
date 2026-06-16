# LUCID Filtering Pipeline

A two-stage filtering pipeline for building high-quality SISR training datasets from web-scraped images.

## Why Filter?

Not all web images are suitable for SISR training. Two specific problems degrade model output:

1. **Flat/uninformative tiles** — sky, water, solid colors. The model wastes capacity learning to reconstruct nothing.
2. **Ringing/haloring artifacts** — bright halos around edges from aggressive sharpening. The model learns to reproduce these artifacts, degrading output quality.

Unlike blur or noise (which don't hurt SISR because the LR is just a bicubic downscaled HR), **ringing IS problematic** — it's a structural artifact that the model faithfully learns to reproduce.

## Pipeline Stages

### Stage 1: ICNet Complexity Filtering

**What:** Scores each tile with ICNet (image complexity neural network) and removes low-complexity tiles.

**Threshold:** ≥ 0.6 for high-complexity subset

**Why:** Removes flat/uninformative regions where the model has nothing to learn. ICNet measures local complexity — edges, textures, patterns.

**Effect:** 23,243 → 23,243 tiles (all passed in our case, as we pre-filtered during extraction)

### Stage 2: CLIP-IQA Quality Filtering

**What:** Scores each tile with CLIP-IQA (CLIP-based image quality assessment) and removes low-quality tiles.

**Threshold:** ≥ 0.3 (enabled by default in `python -m lucid`)

**Why:** CLIP-IQA is the best metric for detecting ringing/haloring artifacts:
- **173.7% separation** between ringing and clean images (3.7× better than next best)
- Not trained on human MOS (avoids bias where humans rate sharpened images as "high quality")
- **Fast with GPU batching** (~307 t/s on RTX 3060 with batch_size=64, ~14× faster than sequential)
- Runs as a second scoring head alongside ICNet — same batch, same GPU, no extra pass

**Effect:** ~73% of tiles kept (preserves diversity while removing clear artifacts)

**Usage:**
```bash
# Enabled by default (threshold 0.3)
python -m lucid --input /path/to/images --output /path/to/output

# Custom threshold
python -m lucid --input /path/to/images --output /path/to/output --clipiqa-threshold 0.4

# Disable clipiqa filtering
python -m lucid --input /path/to/images --output /path/to/output --no-clipiqa
```

## Why These Specific Metrics?

We tested 30+ IQA metrics from pyiqa. Results:

| Metric | Separation | Speed | Verdict |
|--------|-----------|-------|---------|
| **clipiqa** | **173.7%** | FAST | **Best choice** |
| clipiqa+ | 89.0% | FAST | Good alternative |
| qualiclip | 86.0% | FAST | Good alternative |
| TOPIQ_NR | 46.7% | FAST | Moderate |
| NIQE | 13.0% | SLOW | **Failed** — wrong direction |
| BRISQUE | 0.8% | FAST | **Failed** — wrong direction |

Key insight: **CLIP-based metrics dominate** because they're not biased by human MOS ratings. Humans sometimes rate sharpened/ringing images as "high quality," so MOS-trained metrics can't cleanly separate ringing from clean.

## Threshold Selection

clipiqa ≥ 0.3 was chosen based on:
- Worst observed ringing artifact scored 0.262
- Clean reference scored 0.349
- Threshold sits between these values
- Keeps 81% of tiles (preserves diversity while removing clear artifacts)

## Three-Stage Training Pipeline

After filtering, we train with three stages:

| Stage | Dataset | Tiles | Resolution | Purpose |
|-------|---------|-------|------------|---------|
| 1. Pretrain | LUCID CC0 v2 | 1.17M | 256×256 | Learn general image statistics |
| 2. Finetune | LUCID CC0 v2 HC | 193K | 256×256 | Focus on complex content |
| 3. Finetune-finetune | LUCID CC0 v2 HC 512 | 18K | 512×512 | Learn high-res details |

## Usage

### Running the pipeline

clipiqa filtering is **built into the LUCID module** — enabled by default:

```bash
# Run full pipeline (ICNet + clipiqa + dedupe)
python -m lucid --input /path/to/images --output /path/to/output

# Custom clipiqa threshold
python -m lucid --input /path/to/images --output /path/to/output --clipiqa-threshold 0.4

# Disable clipiqa filtering
python -m lucid --input /path/to/images --output /path/to/output --no-clipiqa
```

### Re-filtering an existing dataset

For already-extracted tiles, use the standalone script:

```bash
python score_and_filter.py /path/to/dataset 0.3 64
```

The script uses GPU batching + parallel I/O for high throughput. Benchmark on RTX 3060 (12GB VRAM):

| Method | Throughput | Time for 1M tiles |
|--------|-----------|-------------------|
| Sequential (per-image) | ~22 t/s | ~12.6 hours |
| **GPU batched (this script)** | **~307 t/s** | **~54 minutes** |

Output files written to `dataset_dir/`:
- `clipiqa_scores.csv` — all tile scores (rel_path, clipiqa_score)
- `clipiqa_removed.csv` — removed tiles below threshold

### Training

Use the configs in `configs/train/HAT/` with [traiNNer-redux](https://github.com/the-database/traiNNer-redux):

```bash
# Stage 1: Pretrain
python -m traiNNer.train -opt configs/train/HAT/HAT_M_LUCID_Pretrain.yml

# Stage 2: Finetune (update pretrain_network_g path first)
python -m traiNNer.train -opt configs/train/HAT/HAT_M_LUCID_Finetune_HC.yml

# Stage 3: Finetune-finetune (update pretrain_network_g path first)
python -m traiNNer.train -opt configs/train/HAT/HAT_M_LUCID_FinetuneFinetune_HC_512.yml
```

## Ablation Study

See `iqa_ablation_study.md` for the full 30-metric comparison with cross-correlation analysis.
