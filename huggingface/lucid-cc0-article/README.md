# LUCID: Fast Dataset Filtering for SISR Without Segmentation

[Community Article](/blog/community)

This article documents my research journey from BHI filtering to LUCID — a fast, signal-based approach to curating super-resolution training datasets. The goal: find training images that are *technically clean*, not just aesthetically pleasing.

**Quick links:**
- **[LUCID source code](https://github.com/phhofm/lucid)** — filter your own datasets
- **[`Phips/lucid_cc0`](https://huggingface.co/datasets/Phips/lucid_cc0)** — 231K CC0-licensed filtered tiles (ready to use)
- **[BHI filtering article](https://huggingface.co/blog/Phips/bhi-filtering)** — the earlier work that started this

## The Problem

Single image super-resolution (SISR) networks are only as good as their training data. Most datasets are scraped from the web, which introduces hidden problems:

- **JPEG compression artifacts** — 8×8 block boundaries baked into every compressed image
- **Oversharpening halos** — false high-frequency content from social media filters and post-processing
- **Downsampling implementation differences** — Pillow, OpenCV, and MATLAB bicubic produce different results. This means validation metric scores (PSNR, SSIM) aren't directly comparable across papers unless they used the exact same downsampling code. I discovered this when I couldn't reproduce published metrics on standard validation sets like Urban100 — papers report "PSNR" but actually use PSNR-Y (luminance only), and different image dimensions lead to different rounding at boundaries.
- **Inconsistent validation** — even the standard test sets (Set5, Set14, BSD100, Urban100) have HR images with dimensions not evenly divisible by the downsample scale, causing pixel-level differences between implementations.

These issues are invisible to human eyes but directly affect how SISR networks learn.

## BHI: The First Attempt

My first approach was [BHI filtering](https://huggingface.co/blog/Phips/bhi-filtering) — combining three metrics:

- **Blockiness** — detects JPEG 8×8 grid artifacts
- **HyperIQA** — perceptual quality (trained on human preference data)
- **IC9600** — image complexity/texture richness

The results were promising. On DF2K and ImageNet, BHI-filtered subsets matching the validation metrics of the full dataset with drastically fewer tiles. The blockiness metric was especially impactful — ImageNet is riddled with JPEG artifacts, and removing those tiles alone removed 97% of the data.

But BHI had limitations: HyperIQA and IC9600 are heavy neural network models, slow to run on large datasets. And there was a deeper problem I hadn't anticipated.

## The IQA Metrics Rabbit Hole

I wanted to understand IQA metrics better — which ones are reliable, which overlap, and whether a larger set of metrics could catch problems BHI missed.

### Exploring Metric Selection

I opened discussions on [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) about metric selection ([#247](https://github.com/chaofengc/IQA-PyTorch/issues/247)) and was surprised to find that even the benchmarks show significant disagreement between metrics.

I tested this myself: running 68 different metrics on the same set of SISR model checkpoints at different training iterations. The results were divergent — PSNR said iteration 70k was best, SSIM said 10k, LPIPS said 30k, DISTS said 90k. Each metric was measuring something different, and none agreed on which output was "best."

### PSISRD: 10 Metrics, Maximum Coverage

I created [PSISRD](https://huggingface.co/datasets/Phips/PSISRD_val125) — extending BHI from 3 metrics to 10, chosen to minimize overlap and catch different types of degradation:

```python
DEFAULT_THRESHOLDS = {
    'brisque': 20,          # Pixel-level artifacts (<=)
    'hyperiqa': 0.7,        # Perceptual quality (>=)
    'maniqa': 0.6,          # Holistic quality (>=)
    'clipiqa_plus': 0.38,   # Semantic plausibility (>=)
    'topiq_nr': 0.6,        # Naturalness (>=)
    'musiq': 65,            # Multi-scale quality (>=)
    'tres': 80,             # Transformer-based quality (>=)
    'arniqa': 0.65,         # Agnostic quality (>=)
    'blockiness': 10,       # Compression artifacts (<)
    'complexity': 0.2       # Texture detail (>=)
}
```

This caught oversharpening artifacts that BHI missed — someone in the community flagged that BHI-filtered datasets had oversharpening artifacts, and I confirmed it during evaluation. IQA metrics reward sharpness because humans prefer it on screens, but those halos are false content that confuses SR training.

### The Problem

PSISRD worked, but it had two fatal flaws for large-scale use:

1. **Speed** — running 10 neural network metrics per image took many nights on large datasets. PD12M alone would take days.
2. **Diminishing returns** — only a tiny fraction of images survived all 10 thresholds.

I was overengineering it.

## The Key Insight: Technical Cleanliness > Aesthetic Quality

The breakthrough was realizing that online datasets have a systematic bias: **social media platforms and image sharing sites favor sharpened, high-contrast content**. People edit their photos with sharpening filters, increase contrast, boost saturation. This is what looks good on screens — and it's what gets uploaded and scraped into training datasets.

IQA metrics are trained on human preference data, so they inherit this bias. They *reward* the very artifacts that harm SR training. The metrics that were supposed to help me find "good" training images were actually finding the opposite.

SISR networks don't care about aesthetics. They optimize pixel-level losses (L1, MSE) and respond to technical signal quality. What matters for training is:

- Clean, artifact-free images (no JPEG blocks, no compression noise)
- Natural image statistics (no artificial sharpening halos)
- Genuine texture complexity (not post-processed sharpness)

This is why LUCID exists: to detect technical degradations using fast, signal-based metrics rather than expensive, aesthetic-biased neural networks.

## LUCID: Signal-Based Filtering

LUCID uses six signal-based metrics that are computable in milliseconds each — no neural networks required for the filtering stage:

| Metric | Detects | Threshold |
|--------|---------|-----------|
| **Blockiness** | JPEG 8×8 block artifacts | < 45 |
| **Aliasing ratio** | Oversharpening halos | < 0.65 |
| **Noise ratio** | High-frequency noise | < 0.65 |
| **Entropy** | Information content | > 5.2 |
| **Gradient energy** | Edge density | > 0.70 |
| **Laplacian variance** | Focus/sharpness | 80–10000 |

Only tiles passing ALL thresholds proceed to ICNet complexity scoring.

### ICNet Complexity Scoring

The ICNet model (two ResNet-18 branches: detail + context) predicts per-tile complexity scores (0–1). Higher scores indicate more diverse textures and patterns — the kind of content that actually helps SR networks learn.

**Default threshold**: 0.45 (based on BHI experiments, though higher thresholds like 0.6 produce even cleaner datasets)

This replaces human-centric quality metrics with a learned measure of what's useful for SR training.

## DiverSeg-IP Validation

The community found that [DiverSEG-IP](https://github.com/tinglyfeng/DiverSeg) — which uses semantic segmentation to curate training data — produced strong validation results. I trained on LUCID-filtered datasets from the same sources (ImageNet+PASS) and compared:

| Aspect | DiverSEG-IP | LUCID |
|---|---|---|
| Segmentation | Required | Not needed |
| Processing time | Slow (~10–20×) | Fast |
| Quality outcome | Baseline | Equivalent |

**Result**: LUCID-filtered ImageNet+PASS reached **identical PSNR/SSIM curves** as DiverSEG-IP on 4× HAT-M + Urban100 — proving that signal-based filtering matches segmentation-aware curation without the overhead.

## The lucid_cc0 Dataset

A CC0-licensed subset of LUCID-filtered tiles is available on HuggingFace:

**[`Phips/lucid_cc0`](https://huggingface.co/datasets/Phips/lucid_cc0)**

| Property | Value |
|----------|-------|
| Tiles | 231,788 (256×256 PNG, ~32 GB) |
| Sources | [PD12M](https://huggingface.co/datasets/Spawning/PD12M) (CC0, 88%) + [UHD-IQA](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html) (CC0, 12%) |
| Complexity threshold | ≥ 0.6 |
| License | CC0-1.0 (public domain) |

Includes `metadata.csv` with per-tile source attribution and complexity scores.

### Why threshold 0.6?

Higher complexity thresholds lead to faster PSNR/SSIM convergence in early training. At 0.6, LUCID produces a dataset that matches DiverSeg-IP's performance on 4× HAT-M + Urban100 with 232K tiles instead of 526K.

**Note**: This CC0 dataset uses different source images than DiverSeg-IP (which uses ImageNet+PASS), so the absolute quality differs. The point is that LUCID's *filtering methodology* produces equivalent training outcomes regardless of source — you can filter any large dataset with the tool.

## How to Use LUCID

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

### Robustness Features

- **Resume / Crash Recovery**: SQLite state DB tracks every source image. Re-run with `--resume` to continue from where you left off.
- **Disk-Space Guard**: Estimates peak usage before processing and warns if insufficient space.
- **Large Image Handling**: Images > 8192 px are auto-downscaled using Bicubic interpolation instead of Lanczos to prevent generating false high-frequency ringing halos that confuse SISR network training.
- **External Drive Support**: Auto-detects USB drives and throttles writes to prevent saturating the controller.
- **GPU / CPU Fallback**: Runs on GPU with FP16 when available; falls back to CPU automatically.
- **Corrupt Image Detection**: Eager decode catches truncated JPEGs; skipped by default, abort with `--strict`.

## Validation Results

All configs are 99% identical (pure bicubic pipeline, Charbonnier loss, AdamW 2e-4, 800K iterations on 4× HAT-M). Only `dataroot_gt` differs:

| Config | Dataset | Tiles |
|--------|---------|-------|
| `HAT_M_Probe_Diverseg.yml` | diverseg-ip | 526K |
| `HAT_M_Probe_El_Co_050.yml` | elite-complex-050 | 342K |
| `HAT_M_Probe_CC0_06.yml` | lucid_cc0 | 232K |

### PSNR/SSIM Curves — All Experiments Compared

![PSNR Comparison](training_logs/screenshots/1.png)

### elite-complex-050 Matches DiverSeg-IP on 4× HAT-M + Urban100

![Elite matches DiverSeg](training_logs/screenshots/2.png)

### Complexity Threshold Effect on Validation Metrics

![Threshold Effect](training_logs/screenshots/3.png)

## Reproducibility

All training configs, logs, and screenshots are in the [`lucid-cc0-article`](https://huggingface.co/datasets/Phips/lucid-cc0-article) repository.

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

For standardized validation across models, see [BHI100](https://huggingface.co/datasets/Phips/BHI100) — a set with consistent 480×480 HR images and published downsampling implementations, designed to eliminate the validation inconsistencies between different downsampling algorithms.

---

This is hobby research — I work on whatever I find interesting and publish when I have findings to share. If you filter your own dataset with LUCID, I'd love to hear about it.
