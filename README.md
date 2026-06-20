# LUCID: SISR Dataset GT Filtering & Preparation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Phhofm/lucid-sisr/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Created by [Philip Hofmann](https://github.com/Phhofm)**

A fast, single-command pipeline for filtering SISR (Single Image Super Resolution) training datasets. Focuses on degradation-free, structurally novel data rather than human-perception scores.

```
Input images ──► Signal filter ──► ICNet scoring ──► ResNet18 dedupe ──► Clean tiles
```

---

## One-Click Start

To install and run LUCID immediately without typing any commands, we provide automated one-click launcher scripts:

* **Linux & macOS**: Double-click or run `./run.sh` in your terminal.
* **Windows**: Double-click `run.bat`.

> **What these do automatically:**
> 1. **Checks Python**: Verifies Python 3.10+ is installed. If missing, it will automatically offer to install it (via Homebrew on Mac, APT on Ubuntu/Debian, or the Microsoft Store on Windows).
> 2. **Sets up Virtual Environment**: Automatically creates a local `.venv` inside the project folder.
> 3. **Installs Dependencies**: Automatically installs OpenCV, PyTorch, TorchVision, Pillow, and tqdm.
> 4. **Launches GUI**: Seamlessly boots up the gorgeous dark-themed desktop dashboard!

---

## Command Line Quick Start

### Install from source

```bash
git clone https://github.com/Phhofm/lucid-sisr.git
cd lucid-sisr
pip install -e .
# Optional: PyTorch with CUDA for GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Run

```bash
python -m lucid --input /path/to/raw_images --output /path/to/filtered_dataset
```

That's it. LUCID will:

1. **Auto-download** the ICNet model on first run (bundled in the repo; falls back to [GitHub releases](https://github.com/umzi2/Dataset_Preprocessing/releases) if not found).
2. **Tile** images into 128/256/512/1024 px patches (512 is native; other sizes are auto-calibrated — see *Tile size & calibration* below).
3. **Filter** tiles using signal metrics (entropy, blockiness, gradient energy, noise, aliasing, sharpness).
4. **Score** tiles with a learned complexity model (ICNet) on GPU.
5. **Deduplicate** remaining tiles using ResNet18 fingerprints.
6. **Output** sequentially named, ready-to-train tiles with full provenance.

---

## Installation

### Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.12+ |
| PyTorch | 2.0+ | 2.4+ (latest stable) |
| RAM | 4 GB | 16 GB+ (for large images) |
| Disk | 2× input size free | NVMe SSD |
| GPU | Optional (CPU fallback) | NVIDIA with 6 GB+ VRAM |

```bash
# 1. Clone or download the repo
git clone https://github.com/Phhofm/lucid.git
cd lucid

# 2. Install (creates the `lucid` and `lucid-gui` commands)
pip install -e .

# 3. Install PyTorch with CUDA for GPU acceleration (NVIDIA GPU recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. (Optional, recommended for >200k unique tiles) HNSW-backed dedupe
pip install lucid-sisr[ann]

# 5. (Optional) Intel ARC / discrete Intel GPU support
pip install lucid-sisr[xpu]

# 6. (Optional) Windows DirectML (last-resort AMD/Intel iGPU acceleration)
pip install lucid-sisr[directml]
```

#### GPU Support Matrix

| Platform | Status | Notes |
|---|---|---|
| NVIDIA (CUDA) | First-class | FP16 enabled, VRAM auto-detect, recommended path |
| AMD (ROCm / HIP) | First-class | PyTorch unifies under `torch.cuda`; install the matching ROCm torch wheel |
| Apple Silicon (MPS) | First-class | FP32 path; unified memory |
| Intel ARC / discrete (XPU) | First-class | Requires `pip install lucid-sisr[xpu]` (Intel Extension for PyTorch) |
| Windows DirectML | Available | Requires `pip install lucid-sisr[directml]`; slower than vendor SDKs |
| CPU | Always available | Batch auto-clamped to 1–4; expect ~10× slower scoring |

LUCID auto-detects the active accelerator at startup and logs which one is in use:
`Compute device: CUDA — NVIDIA GeForce RTX 5070` (or `ROCm`, `XPU`, `MPS`, `DirectML`, `CPU`).

> **Note:** LUCID auto-detects GPU and falls back to CPU automatically. For NVIDIA GPU acceleration, install PyTorch with CUDA support using the command above. Apple Silicon (MPS) and CPU-only machines are fully supported but will process slower. The GUI (`lucid-gui`) requires `python3-tk` (e.g., `sudo apt install python3-tk` on Ubuntu/Debian). Headless servers have no display server.

---

## Usage

### Single Command

```bash
python -m lucid --input /path/to/raw_images --output /path/to/filtered_dataset
```

The pipeline runs all three stages (ingest → dedupe → finalize) automatically.

### Output Structure

```
/path/to/filtered_dataset/
├── tiles/              # Final filtered tiles
│   ├── 00001_datasetA.png
│   ├── 00002_datasetA.png
│   └── ...
└── master_lineage.csv  # Provenance: tile → source image mapping
```

After successful completion, a `lucid_workspace_<input-name>/` scratch folder is left inside the output directory (it contains a hidden `.lucid/lucid_state.db`, the intermediate `raw_tiles/`, `scored/`, and `scores/` subdirs, plus a `scored/redundant/` subfolder with duplicates). Remove it with `--cleanup` or delete it manually.

### Full CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | *(required)* | Folder with source images (scanned recursively) |
| `--output PATH` | `./lucid_output` | Destination for final filtered tiles |
| `--workspace PATH` | *(auto)* | Scratch workspace dir; auto-creates inside `--output` if omitted |
| `--icnet PATH` | *(auto-download)* | Path to ICNet `.pth` weights; downloads automatically on first run |
| `--tile-size INT` | `512` | Tile dimension in pixels. Supported: **128, 256, 512, 1024**. 512 is native; other sizes use a per-size linear calibration curve (R² ≥ 0.94, shipped in `lucid/resources/icnet_calibration.json`) to map raw ICNet scores to the 512-trained distribution. |
| `--no-calibrate` | off | Disable the calibration transform. Use raw ICNet scores for the threshold comparison. Default: enabled. |
| `--threshold FLOAT` | `0.45` | Minimum ICNet complexity score (0.0–1.0) |
| `--dedupe-threshold FLOAT` | `0.96` | Cosine similarity above which tiles are considered duplicates |
| `--workers INT` | `cpu_count - 2` | Parallel CPU workers for tile extraction |
| `--batch INT` | `auto` | GPU batch size for ICNet scoring (0 = auto-detect from VRAM) |
| `--max-vram-gb FLOAT` | `0.0` | Cap VRAM usage in GB (0 = use all available) |
| `--multiscale` | *(off)* | Tile at 1.0×, 0.75×, 0.5×, 0.25× scales |
| `--io-throttle-ms FLOAT` | `auto` | ms sleep between disk writes (auto-detects USB drives) |
| `--strict` | *(off)* | Abort on the first corrupt/unreadable image |
| `--resume` | *(off)* | Continue a previous run from the workspace state DB |
| `--stage STAGE` | `all` | Run a single stage: `all`, `ingest`, `dedupe`, or `finalize` |
| `--cleanup` | *(off)* | Remove `lucid_workspace_<input>/` scratch folder after successful completion |
| `--dry-run` | *(off)* | Validate inputs and estimate disk usage without writing |
| `-v, --verbose` | *(off)* | Enable debug-level logging |

### Common Patterns

```bash
# Resume after interruption (power loss, disk full, Ctrl+C)
python -m lucid --input /images --output /filtered --resume

# Process on fast internal SSD, output to USB drive
python -m lucid --input /images --output /usb/output --workspace /tmp/lucid_scratch

# Run only ingest and dedupe, skip final rename (for debugging)
python -m lucid --input /images --output /filtered --stage dedupe

# Dry run: check if you have enough disk space before committing
python -m lucid --input /images --output /filtered --dry-run --verbose

# Launch the Tkinter GUI (if available on your system)
lucid-gui
```

---

## How It Works

### Tile size & calibration

ICNet was trained at **512×512 px**. Running it at other sizes introduces
predictable drift in the raw score distribution, but the drift is
linear and the same images preserve their relative ordering. LUCID
ships a per-size calibration table (`lucid/resources/icnet_calibration.json`)
that maps raw scores to the 512-trained distribution:

| Tile size | Calibration status | Suggested `--threshold` | R² | Max abs. error |
|---|---|---|---|---|
| **128 px** | Approximate | `0.47` | 0.94 | 0.087 |
| **256 px** | Calibrated | `0.45` | 0.99 | 0.048 |
| **512 px** | Native | `0.45` | 1.00 | 0.000 |
| **1024 px** | Calibrated | `0.45` | 0.99 | 0.040 |

- Calibration is on by default. Pass `--no-calibrate` to use the raw scores as-is.
- The CSV `scored/ingest_scores.csv` has both `complexity_score` (raw) and
  `score_calibrated` (mapped) columns.
- To regenerate the calibration table from your own representative images, run:

  ```bash
  lucid-calibrate --input /path/to/representative/folder
  ```

  This will rescore 6 random 1024-crops per image at each supported size
  in FP32, fit a linear regression against the 512-px scores, and overwrite
  the JSON in the installed `lucid/resources/` directory.

### Choosing the right `--threshold`

The threshold controls what fraction of tiles survive ICNet scoring. The
right value depends on your downstream task:

| Use case | Suggested threshold | What you keep |
|---|---|---|
| Aggressive: small, high-quality dataset | `0.50 – 0.55` | Only very-detailed textures. ~10–25% of input tiles. |
| **Balanced (default)** | `0.45` | Solid detail and context mix. ~30–50% of input tiles. |
| Permissive: large dataset, room to prune later | `0.40 – 0.42` | Captures more variety, including low-contrast regions. ~50–70% of input tiles. |
| Research / pre-training: maximize coverage | `0.30 – 0.35` | Keeps nearly everything that survives the signal filter. ~80%+ of input tiles. |

**Heuristics for choosing:**

- **For LSDIR / DF2K / high-resolution SISR training** the `0.42 – 0.45` band
  tends to be the sweet spot. Below `0.40` the model starts seeing too many
  near-flat tiles and the loss landscape flattens.
- **For 4×–8× super-resolution with aggressive downsampling** you can push
  the threshold lower (`0.38 – 0.42`) because low-frequency content still
  provides useful training signal for the upsampler.
- **For 2× / light upscaling** prefer the higher end (`0.48 – 0.55`) — the
  model needs sharp, information-dense tiles to learn fine detail.

**How to inspect what you'll get before running on a huge folder:**

```bash
# Dry-run on a representative sample
python -m lucid --input ./sample/ --output /tmp/lucid_preview --dry-run --tile-size 512

# Or run the full ingest on a small folder and look at:
#   workspace/scored/threshold_used.txt   (audit trail: what threshold + R²)
#   workspace/scores/histogram.txt       (score distribution)
#   workspace/scores/calibration_outliers.csv  (top 100 score drift)
#   master_lineage.csv                    (per-image pass-rate)
```

**Quick sweep on a small folder:**

```bash
for t in 0.40 0.42 0.45 0.48 0.50; do
  python -m lucid --input ./sample/ --output /tmp/lucid_t${t} --threshold $t
  echo "t=$t: $(ls /tmp/lucid_t${t}/tiles/ | wc -l) tiles"
done
```

Pick the threshold that yields the count you want, then run on the full
dataset. The `_init_score_csv` header and `threshold_used.txt` are stable
across runs, so you can re-run and compare without losing provenance.

### Stage 1: INGEST — Signal Filter + ICNet Scoring

Each source image is loaded, validated for corruption, and split into tiles. Tiles are filtered in two tiers:

**Tier 1 — Signal Metrics (CPU, per-tile):**

| Metric | Threshold | What it detects |
|---|---|---|
| Entropy | ≥ 5.2 bits | Flat/empty regions |
| Laplacian variance | 80–10,000 | Blur or oversharpening |
| Gradient energy | ≥ 0.70 | Lack of texture |
| Blockiness | ≤ 45.0 | JPEG compression artifacts |
| Noise ratio | ≤ 0.65 | Sensor/compression noise |
| Aliasing ratio | ≤ 0.65 | Edge-enhancement halos |

**Tier 2 — ICNet Complexity (GPU):**  
Tiles that pass signal filters are scored by a pre-trained ICNet model (two ResNet-18 branches: detail + context). The model outputs a per-pixel complexity map and a scalar score. Tiles below `--threshold` (default 0.45) are discarded.

### Stage 2: DEDUPE — Diversity Audit

Remaining tiles are embedded using a pre-trained ResNet-18 (classification head removed, 512-D feature vector). Each new tile is compared against the pool of already-accepted tiles using cosine similarity. Tiles above `--dedupe-threshold` (default 0.96) are moved to `scored/redundant/`.

### Stage 3: FINALIZE — Consolidation

Unique tiles are copied to `output/tiles/` with sequential naming (`00001_dataset.png`, `00002_dataset.png`, …) and a `master_lineage.csv` is written mapping every output tile back to its source image, original filename, and absolute path.

---

## Robustness Features

### Crash Recovery / Resume

Every source image is tracked in an SQLite state database (`lucid_workspace_<input>/.lucid/lucid_state.db`). If the pipeline is interrupted (power loss, `Ctrl+C`, disk full), re-run with `--resume` to pick up exactly where it left off.

### Disk-Space Guard

Before processing, LUCID estimates the peak disk usage and compares it to free space on the target drive. If insufficient, it aborts with a clear error message.

### Large Image Handling

Images exceeding 8192 px on any dimension are automatically downscaled before tiling to prevent RAM exhaustion. To protect the ground-truth (GT) purity of your SISR training tiles, we use **Bicubic interpolation** (`Image.BICUBIC`) instead of Lanczos, preventing artificial **oversharpening halos or ringing artifacts** that would mislead the super-resolution network during training. This choice reflects careful consideration: Lanczos introduces sharpening artifacts, Mitchell-Netravali may be cleaner but requires custom implementation, and even bicubic shows metric differences between Pillow and MATLAB implementations. After experimenting with Magic Kernel Sharp and DPID, I chose the standard, well-tested Bicubic implementation for its balance of technical cleanliness and reliability. The PIL decompression bomb limit is capped at 200 MP (not disabled).

### External / USB Drive Support

- `--io-throttle-ms` auto-detects removable drives and adds a small delay between disk writes to prevent saturating the USB controller
- `--workspace` lets you keep all intermediate files on a fast internal drive while writing the final output to a slow drive

### GPU / CPU Fallback

LUCID auto-detects the active accelerator (CUDA → ROCm → XPU → MPS → DirectML → CPU)
and logs which device is in use at startup. The `--max-vram-gb` flag is available
to override the auto-detected value on exotic hardware.

- **With CUDA, ROCm, or XPU:** ICNet and ResNet-18 run on GPU with FP16 (half-precision) for maximum throughput
- **With MPS (Apple Silicon):** FP32 path; unified memory; ~3–5× slower than NVIDIA
- **Without any GPU:** Everything runs on CPU with batch size automatically clamped to 1–4 and a warning printed
- **Dedupe scale:** with `pip install lucid-sisr[ann]` the dedupe index uses HNSW (O(log N) queries, scales to ~10M tiles). Without it, a brute-force cosine matmul is used; adequate for ≤200k unique tiles

**Tested on:**
- Linux + NVIDIA RTX 5070 (CUDA 12.x) — reference path
- Linux + AMD Radeon (ROCm 6.x torch wheel) — same code path as CUDA
- macOS Apple Silicon (MPS) — FP32
- Windows + Intel ARC (XPU) — via `pip install lucid-sisr[xpu]`
- Windows + AMD/Intel iGPU (DirectML) — via `pip install lucid-sisr[directml]`; slower
- Any CPU — always works; expect ~10× slower than NVIDIA GPU

### Corrupt Image Detection

`Image.open(...).load()` is used (eager decode) with `LOAD_TRUNCATED_IMAGES = False` to catch truncated JPEGs and other corrupted files. In normal mode, corrupt images are logged and skipped. With `--strict`, the pipeline aborts on the first corrupt image.

---

## Performance Tips

| Scenario | Recommendation |
|---|---|
| **Fast NVMe SSD** | Default settings; `--workers 12+` for many cores |
| **External USB HDD** | `--workspace /tmp/lucid` (fast internal scratch), `--io-throttle-ms 5` |
| **Low VRAM GPU (< 6 GB)** | `--tile-size 256 --batch 4` |
| **CPU-only machine** | Expect ~10× slower ICNet scoring; reduce `--batch` to 2 |
| **Many small images** | Increase `--workers` to match core count (default: `cpu_count - 2`) |
| **Millions of source images** | Run `--dry-run` first to estimate disk space and tile count |
| **Don't lose progress** | Always add `--resume` so you can continue after interruption |

---

## Personal Journey & Evolution

My path to LUCID came from years of experimenting with SISR dataset creation, evolving from IQA-dependent approaches to mathematically principled curation:

- **Early IQA Exploration**: Questioned metric limitations on PyIQA  
  [Issue #247](https://github.com/chaofengc/IQA-PyTorch/issues/247) · [Issue #182](https://github.com/chaofengc/IQA-PyTorch/issues/182)

- **BHI Dataset**: First principled SISR curation attempt  
  [HuggingFace Dataset](https://huggingface.co/datasets/Phips/BHI) · [Methodology Blog](https://huggingface.co/blog/Phips/bhi-filtering)

- **BHI100 Validation**: Standardized evaluation addressing downscaling inconsistencies (Pillow vs MATLAB)  
  [Results Code](https://github.com/Phhofm/bhi100-sisr-iqa-metrics) · [Live Dashboard](https://phhofm.github.io/bhi100-sisr-iqa-metrics/)

- **PSISRD Validation**: Comprehensive IQA metric evaluation (125+ metrics)  
  [HuggingFace Dataset](https://huggingface.co/datasets/Phips/PSISRD_val125)

**Key Insights from Journey:**
1. **Computational infeasibility**: Filtering with thousands of IQA models revealed days-long processing times for large datasets
2. **Conceptual mismatch**: Human-pleasing metrics (MOS, NIQE, etc.) don't necessarily optimize SR network learning — observed through inconsistent "best" checkpoint selections across different metrics
3. **Degradation sensitivity**: Even standard datasets like Urban100 gave different results based on downscaling method (Pillow vs MATLAB), motivating standardized validation sets
4. **Key realization**: What helps SISR training isn't what looks pleasing to humans, but what provides meaningful structural gradients during optimization

This led to LUCID's core premise: **Focus on degradation-free, structurally novel data that provides consistent learning signals rather than human-perception scores**.

---

## Why Filter for Technical Cleanliness?

Most online datasets (ImageNet, PASS, etc.) are lossy JPEG compressed and often edited/sharpened before posting. This introduces hidden degradations that human vision ignores but confuse SR models:

- **JPEG blockiness** — 8×8 block boundary artifacts invisible to human vision but confusing for SR reconstruction
- **Oversharpening halos** — Subtle edge enhancement artifacts from photo editing that introduce false high-frequency content
- **Compression noise** — High-frequency noise patterns that degrade training signals

Unlike IQA metrics (NIQE, HyperIQA, BRISQUE) that favor sharp aesthetic images, LUCID detects these technical degradations to provide cleaner learning signals for super-resolution.

---

## DiverSEG Inspiration (Without Segmentation Cost)

LUCID draws inspiration from DiverSEG-IP but achieves similar results WITHOUT semantic segmentation:

| Aspect | DiverSEG-IP | LUCID |
|---|---|---|
| Segmentation | Required (slow) | Not needed (fast) |
| Processing time | ~10–20× slower | Fast, scalable |
| Quality outcome | Baseline | Equivalent (validated) |

**Validation**: Tested on ImageNet+PASS with LUCID filtering → HAT-M training reached **identical PSNR/SSIM curves** as DiverSEG-IP, proving segmentation-free filtering can match segmentation-aware curation at a fraction of the computational cost.

---

## Key Learnings from BHI Work

From the BHI filtering experiments ([Blog Post](https://huggingface.co/blog/Phips/bhi-filtering)):

- **Blockiness < 30**: Critical for handling JPG compression artifacts detrimental to SR training
- **HyperIQA ≥ 0.2**: Removing only the worst tiles (bottom 20%) improved metrics more than keeping only the best
- **IC9600 ≥ 0.4**: Higher complexity filtering showed benefits, though optimal threshold varied by dataset
- **Surprising finding**: Models trained on 69% reduced datasets (BHI-filtered DF2K) often outperformed full datasets in PSNR/SSIM/DISTS metrics
- **Dataset quantity matters**: Extreme reduction (like ImageNet-BHI at 2.3%) hurt performance, suggesting an ideal retention rate exists

---

## Philosophical Foundation

LUCID represents the culmination of realizing that:

1. SISR networks optimize mathematical loss functions, not human opinion scores
2. Degradation-free, structurally novel data provides more consistent learning signals
3. Diversity prevents overfitting to common textures
4. Traceability enables scientific ablation studies
5. System safety (`os.nice(15)`) allows background processing without disrupting workflow

This approach shifts focus from *"What looks good?"* to *"What helps the network learn best?"*

---

## Credits

- **IC9600 Complexity Model**: The ICNet model used for tile complexity scoring is based on [IC9600](https://github.com/tinglyfeng/IC9600) by [Tingly Feng](https://github.com/tinglyfeng). Credit where credit is due.
- **LUCID Pipeline**: Created by [Philip Hofmann](https://github.com/Phhofm).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **ICNet download fails** | The model is bundled in the repo at `lucid/resources/icnet.pth`. If missing, manually download from [IC9600 releases](https://github.com/umzi2/Dataset_Preprocessing/releases) and pass `--icnet /path/to/ic9600.pth` |
| **`--tile-size` rejected** | Only 128/256/512/1024 are supported. Run `lucid-calibrate --input <folder>` to add new sizes to the calibration table. |
| **Bad score distribution at non-512 sizes** | The shipped calibration fits a 7-image reference set. For domain-specific data, regenerate with `lucid-calibrate --input <your-images>` and reinstall. |
| **CUDA out of memory** | Reduce `--batch` (e.g., `--batch 4`) or `--tile-size` (e.g., `--tile-size 256`). Use `--max-vram-gb` to cap usage |
| **Resume doesn't work** | Ensure `--workspace` matches the path used in the original run. The state DB lives at `lucid_workspace_<input>/.lucid/lucid_state.db` |
| **No images found** | Check file extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff` are supported (case-insensitive) |
| **Processing very slow** | If no GPU is available, CPU scoring is ~10× slower. Ensure `--workers` matches your core count. Set `--workspace` on an SSD, not HDD |
| **Dedupe stage OOM on large datasets** | Install the HNSW backend: `pip install lucid-sisr[ann]`. Brute-force is fine up to ~200k unique tiles; HNSW scales to ~10M+ |
| **No GPU detected but I have an AMD/Intel card** | AMD ROCm: install the ROCm torch wheel matching your GPU. Intel XPU: `pip install lucid-sisr[xpu]`. Windows DirectML: `pip install lucid-sisr[directml]` |
| **Disk full mid-run** | Free space and re-run with `--resume`. The state DB ensures already-processed images are skipped |
| **`lucid-gui` crashes** | Requires tkinter. Install `python3-tk` (Ubuntu/Debian) or `tk` (Arch). Headless servers have no display server |

---

## License

MIT License — see [LICENSE](LICENSE) file.
