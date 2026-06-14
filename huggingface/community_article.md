# LUCID: Fast Dataset Filtering for SISR Without Segmentation

[Community Article](/blog/community) 

## Intro

Following my [BHI filtering article](https://huggingface.co/blog/Phips/bhi-filtering), I developed LUCID — a faster, more automated approach to curating super-resolution training datasets.

While BHI combined HyperIQA and IC9600 for filtering, LUCID takes a different philosophical approach: instead of aesthetic quality metrics (which favor sharp, pleasing images), LUCID uses **technical signal quality metrics** that detect hidden degradations invisible to human vision but harmful to SR training.

## The Problem with IQA Metrics

IQA (Image Quality Assessment) metrics like NIQE, HyperIQA, and BRISQUE are trained on human perception data. They prioritize:
- Sharp edges
- High contrast regions
- Visually pleasing textures

But SISR networks learn differently:
- They optimize pixel-level losses (L1, MSE)
- Respond to technical degradation artifacts
- Don't "see" images like humans do

**Important finding**: The community discovered oversharpening artifacts in some BHI-filtered datasets. IQA metrics often *reward* slight oversharpening since it makes images appear sharper — exactly what humans prefer. But these halos introduce false high-frequency content that confuses SR training.

This mismatch led to inconsistent results in my early experiments. Models trained on "aesthetically pleasing" subsets often underperformed those trained on technically clean images.

## Speed Matters for Large Datasets

Previous approaches like [PSISRD](https://huggingface.co/datasets/Phips/PSISRD_val125) used 10+ IQA metrics per image. For large datasets like PD12M, this would take **days** to complete. LUCID uses only signal-based metrics (computable in milliseconds each) plus one ICNet pass, making it practical for million-image datasets.

## LUCID's Technical Approach

LUCID uses six signal-based metrics to detect degradations:

| Metric | Detects | Threshold |
|--------|---------|-----------|
| **Blockiness** | JPEG 8×8 block artifacts | < 45 |
| **Aliasing ratio** | Oversharpening halos | < 0.65 |
| **Noise ratio** | High-frequency noise | < 0.65 |
| **Entropy** | Information content | > 5.2 |
| **Gradient energy** | Edge density | > 0.70 |
| **Laplacian variance** | Focus/sharpness | 80–10000 |

Only tiles passing ALL thresholds proceed to ICNet complexity scoring.

## ICNet Complexity Scoring

The ICNet model (two ResNet-18 branches: detail + context) predicts per-tile complexity scores (0–1). Higher scores indicate more diverse textures and patterns.

**Default threshold**: 0.45 (based on BHI experiments)

This replaces human-centric metrics with a learned measure of what's useful for SR training.

## DiverSEG Inspiration (Without Segmentation)

LUCID draws inspiration from [DiverSEG](https://github.com/tinglyfeng/DiverSeg) but achieves similar results WITHOUT semantic segmentation:

| Aspect | DiverSEG | LUCID |
|---|---|---|
| Segmentation | Required | Not needed |
| Processing time | Slow (~10–20×) | Fast |
| Quality outcome | Baseline | Equivalent (validated) |

**Validation**: Tested on ImageNet+PASS with LUCID filtering → HAT-M training reached **identical PSNR/SSIM curves** as DiverSEG-IP, proving segmentation-free filtering matches segmentation-aware curation.

## Installation & Quick Start

```bash
pip install -e .
python -m lucid --input /path/to/raw_images --output /path/to/filtered_dataset
```

That's it! The ICNet model auto-downloads on first run. The pipeline runs all three stages (ingest → dedupe → finalize) automatically in a single command.

### Output Structure

```
/path/to/filtered_dataset/
├── tiles/              # Final filtered tiles (00001_dataset.png, ...)
└── master_lineage.csv  # Provenance: each tile → source image
```

### Advanced Usage

```bash
# Resume after interruption
python -m lucid --input /images --output /filtered --resume

# Process on fast internal SSD, output to USB
python -m lucid --input /images --output /usb/out --workspace /tmp/lucid

# Dry run: validate inputs and estimate disk space
python -m lucid --input /images --output /filtered --dry-run --verbose

# Launch the GUI (if tkinter is available)
lucid-gui
```

## Key Differences from BHI

| Aspect | BHI | LUCID |
|---|---|---|
| IQA metrics | HyperIQA + IC9600 | Signal metrics only |
| Focus | Aesthetic quality | Technical cleanliness |
| Multi-scale | No | Yes (optional `--multiscale`) |
| Auto-model download | No | Yes |
| Resume / crash recovery | Manual | SQLite state DB |
| Disk space guard | No | Yes |
| USB drive support | No | Auto I/O throttling |

## Why Technical Cleanliness Matters

Most online datasets have hidden issues:
- **JPEG block boundaries** (every 8 pixels) — invisible but confuse SR
- **Oversharpening halos** — false high-frequency content
- **Compression noise** — degrades learning signals

Unlike IQA metrics that reward sharp aesthetics, LUCID detects these technical issues to provide cleaner training signals.

## Validation Results

- **DF2K**: 69% reduction, better metrics
- **ImageNet+PASS**: Reached identical PSNR/SSIM curves as DiverSEG-IP
- **LSDIR**: 64.9% reduction, maintained performance

## Robustness Features

LUCID is designed to be used easily on any machine:

- **Resume / Crash Recovery**: SQLite state DB tracks every source image. Re-run with `--resume` to continue from where you left off.
- **Disk-Space Guard**: Estimates peak usage before processing and warns if insufficient space.
- **Large Image Handling**: Images > 8192 px are auto-downscaled using **Bicubic interpolation** (`Image.BICUBIC`) instead of Lanczos to prevent generating false high-frequency ringing halos that confuse SISR network training. This choice reflects a careful consideration of downsampling algorithms: Lanczos introduces sharpening artifacts, Mitchell-Netravali may be cleaner but requires custom implementation, and even bicubic shows metric differences between Pillow and MATLAB implementations. After experimenting with Magic Kernel Sharp and DPID "Rapid, Detail-Preserving Image Downscaling", I chose the standard, well-tested Bicubic implementation for its balance of technical cleanliness and reliability.
- **External Drive Support**: Auto-detects USB drives and throttles writes to prevent saturating the controller.
- **GPU / CPU Fallback**: Runs on GPU with FP16 when available; falls back to CPU automatically.
- **Corrupt Image Detection**: Eager decode catches truncated JPEGs; skipped by default, abort with `--strict`.

## Available Dataset: lucid_cc0

A CC0-licensed subset of LUCID-filtered tiles is available on HuggingFace:

**[`Phips/lucid_cc0`](https://huggingface.co/datasets/Phips/lucid_cc0)**

| Property | Value |
|----------|-------|
| Tiles | 231,788 (512×512 PNG, ~32 GB) |
| Sources | [PD12M](https://huggingface.co/datasets/Spawning/PD12M) (CC0, 88%) + [UHD-IQA](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) (CC0, 12%) |
| Complexity threshold | ≥ 0.6 |
| License | CC0-1.0 (public domain) |

Includes `provenance.csv` with per-tile source attribution and complexity scores.

### Why threshold 0.6?

Higher complexity thresholds lead to faster PSNR/SSIM convergence in early training. At 0.6, LUCID produces a dataset that matches DiverSeg-IP's performance on 4× HAT-M + Urban100 with 232K tiles instead of 526K.

## Validation Proof

### PSNR/SSIM Curves — All Experiments Compared

![PSNR Comparison](training_logs/screenshots/1.png)

### elite-complex-050 Matches DiverSeg-IP on 4× HAT-M + Urban100

![Elite matches DiverSeg](training_logs/screenshots/2.png)

### Complexity Threshold Effect on Validation Metrics

![Threshold Effect](training_logs/screenshots/3.png)

## Reproducibility

All training configs and logs are included in the dataset repository.

### Training Configs

All configs are 99% identical (pure bicubic pipeline, Charbonnier loss, AdamW 2e-4, 800K iterations). Only `dataroot_gt` differs:

| Config | Dataset | Tiles |
|--------|---------|-------|
| `HAT_M_Probe_Diverseg.yml` | diverseg-ip | 526K |
| `HAT_M_Probe_El_Co_050.yml` | elite-complex-050 | 342K |
| `HAT_M_Probe_CC0_06.yml` | lucid_cc0 | 232K |

### Training Logs

TensorBoard `.tfevents` converted to human-readable PNG + CSV:

```
training_logs/
├── diverseg/          # {psnr,ssim,loss}.{png,csv}
├── elite_complex_050/ # {psnr,ssim,loss}.{png,csv}
├── cc0_06/            # {psnr,ssim,loss}.{png,csv}
├── comparison_psnr.png
├── comparison_ssim.png
└── screenshots/       # Validation comparison plots
```

## LUCID-CC0 v2: Large-Scale Training Data

The v1 dataset proved the concept, but at 231K tiles (32 GB) from PD12M + UHD-IQA, it's limited for large-scale training. **LUCID-CC0 v2** is a massive upgrade: ~200 GB of CC0-licensed tiles sourced entirely from PxHere, processed through LUCID's signal-based filtering pipeline.

### Three-Stage Training Pipeline

v2 is designed around a progressive three-stage training strategy, each stage building on the previous:

| Stage | Dataset | Resolution | Tiles | Purpose |
|-------|---------|------------|-------|---------|
| 1. Pretrain | `lucid-cc0-v2` | 256×256 | ~1.3M | Train from scratch on diverse, clean data |
| 2. Finetune | `lucid-cc0-v2-hc` | 256×256 | ~271K | Focus on high-complexity tiles (≥ 0.85) |
| 3. Finetune-finetune | `lucid-cc0-v2-hc-512` | 512×512 | — | High-resolution refinement, complexity ≥ 0.85 |

**Stage 1** (`lucid-cc0-v2`): Pretrain a model from scratch on ~1.3M LUCID-filtered tiles. The large tile count provides broad coverage of natural image statistics.

**Stage 2** (`lucid-cc0-v2-hc`): Finetune on a high-complexity subset (~271K tiles, complexity ≥ 0.85). This sharpens the model's ability to reconstruct fine textures and diverse patterns.

**Stage 3** (`lucid-cc0-v2-hc-512`): Finetune again on 512×512 high-complexity tiles. The increased resolution lets the model learn finer detail reconstruction at scale.

### MATLAB-Compatible Downscaling

LR images are generated using bicubic downscaling that matches MATLAB's `imresize` — the standard reference in SR research. This ensures metrics computed during evaluation are directly comparable to published results.

### Datasets

All datasets are CC0-1.0 licensed (public domain) and available on HuggingFace:

| Dataset | HuggingFace Link | Description |
|---------|------------------|-------------|
| `lucid-cc0-v2` | [Phips/lucid-cc0-v2](https://huggingface.co/datasets/Phips/lucid-cc0-v2) | Full v2 dataset (~1.3M tiles, 256×256) |
| `lucid-cc0-v2-hc` | [Phips/lucid-cc0-v2-hc](https://huggingface.co/datasets/Phips/lucid-cc0-v2-hc) | High-complexity subset (~271K tiles, 256×256) |
| `lucid-cc0-v2-hc-512` | [Phips/lucid-cc0-v2-hc-512](https://huggingface.co/datasets/Phips/lucid-cc0-v2-hc-512) | High-complexity 512×512 tiles |

## Future Work

- Multi-dataset merging strategies
- Per-dataset threshold profiles
- Integration with neosr training pipeline

You made it to the end! Thank you for reading :)
