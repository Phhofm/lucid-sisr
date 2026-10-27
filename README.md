# LUCID — SISR Dataset GT Filtering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Phhofm/lucid-sisr/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ci](https://github.com/Phhofm/lucid-sisr/actions/workflows/ci.yml/badge.svg)](https://github.com/Phhofm/lucid-sisr/actions/workflows/ci.yml)

**By [Philip Hofmann](https://github.com/Phhofm)** — one command turns raw images into clean, deduplicated SISR ground-truth tiles.

```
raw images ──► signal filter ──► ICNet complexity ──► ResNet18 dedupe ──► clean tiles
```

---

## 1. Install

```bash
git clone https://github.com/Phhofm/lucid-sisr.git
cd lucid-sisr
pip install -e .

# NVIDIA GPU (recommended, optional):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Large datasets (>200k unique tiles, optional):
pip install lucid-sisr[ann]
```

No command line? `./run.sh` (Linux/macOS) or `run.bat` (Windows) installs everything and opens the GUI.
AMD/Intel/Apple GPUs: see [Troubleshooting](#9-troubleshooting).

---

## 2. Run

```bash
lucid --input /path/to/raw_images --output /path/to/dataset
```

That's the whole pipeline. It:

1. Tiles images at your `--tile-size` (default 512; multiscale adds 0.75×/0.5×/0.25×).
2. Rejects degraded tiles (JPEG blocks, blur, oversharpening, heavy noise, UI/stripes).
3. Scores complexity with ICNet; rejects the junk tier.
4. Deduplicates by perceptual similarity.
5. Writes `tiles/00001_<dataset>.png...` + full provenance.

Output:

```
/path/to/dataset/
├── tiles/               # ready-to-train GT tiles
├── master_lineage.csv   # tile → source image provenance
└── lucid_workspace_<input>/   # scratch (delete or --cleanup)
```

First run logs `Auto-threshold ... (junk floor)` and picks device + batch size automatically.

---

## 3. Recipes

```bash
# Interrupted (power loss / Ctrl+C / disk full)? Continue:
lucid --input /images --output /out --resume

# Small SISR network (limited capacity wants high-complexity exposure):
lucid --input /images --output /out --trim-bottom 50

# Pre-curated input (trust it fully, junk floor only):
lucid --input /images --output /out --threshold 0

# Check disk usage / tile count before a huge run:
lucid --input /images --output /out --dry-run

# GUI:
lucid-gui
```

**Scan fast, write slow:** `--workspace /fast-ssd/scratch --output /usb-drive/dataset` keeps intermediates on the fast drive; `--io-throttle-ms 5` protects USB.

---

## 4. Default quality gates

Calibrated against curated GT ([PSISRD val125](https://huggingface.co/datasets/Phips/PSISRD_val125)): a run keeps **what clean GT never violates** — reject junk, not content.

| Gate | Default | Rejects |
|---|---|---|
| Entropy | ≥ 5.2 bits | flat/sky tiles |
| Laplacian variance | 80–10,000 | blur, oversharpening |
| Gradient energy | ≥ 0.70 | textureless tiles |
| Blockiness (ratio) | ≤ 1.7 | JPEG compression blocks |
| Aliasing (Hann FFT) | ≤ 0.80 | UI screenshots, stripes, ringing |
| Clipped pixels | ≤ 40% | blown/crushed frames |
| ICNet complexity | ≥ 0.30 (junk floor) | tiles below any usable content |
| Dedupe cosine | ≤ 0.96 | near-duplicate content |

**Recommended perceptual tier** (learned, opt-in — `pip install -e ".[iqa]"`):
```
--iqa-gate topiq_nr:0.665 --iqa-gate clipiqa+:0.45 --iqa-gate niqe:6.5 --iqa-gate qrealign-mini:0.78
```
Cheap→expensive cascade; each only sees survivors. Thresholds calibrated on
curated GT (PSISRD val125): topiq p01=0.665, niqe p95=6.65, clipiqa+ keeps 100%,
qrealign p01=0.77. `niqe` catches photographic grain the handcrafted proxies
are blind to; `qrealign-mini` catches oversharpening halos.

Complexity **shaping** (how much of the low-complexity tail to expose) is explicit, not a hidden default:

| Goal | Knob |
|---|---|
| Compact nets learn per-step | `--trim-bottom 50` (keep top ~half by complexity) |
| Balanced | `--trim-bottom 20` |
| Maximum coverage / pre-curated GT | `--trim-bottom 0 --threshold 0` |

Different domain (anime, screenshots, 16-bit scans)? Derive the numbers for it instead of guessing:

```bash
lucid-calibrate-gates /path/to/reference_images
lucid-calibrate       --input /path/to/reference_images   # tile-size score calibration
```

---

## 5. Verify a run

```bash
cat <output>/lucid_workspace_<input>/scores/histogram.txt        # score distribution
cat <output>/lucid_workspace_<input>/scores/scale_pass_rates.txt # multiscale yield per scale
ls <output>/tiles | wc -l                                        # final tile count
column -s, -t < <output>/lucid_workspace_<input>/scores/ingest_scores.csv | less -S
```

`run_config.json` next to `master_lineage.csv` records every gate value + counts — the run is provable.

---

## 6. How it works

**Ingest** — every image is validated (truncated/corrupt files are skipped, `--strict` aborts instead), split into tiles, and filtered by two cheap tiers: signal metrics (table above, CPU, parallel workers) and ICNet complexity (GPU, FP16 on CUDA/XPU). Images >8192 px are downscaled with Pillow Bicubic — no Lanczos ringing in your GT.

**Dedupe** — ResNet-18 fingerprints (512-D); each tile streams against everything accepted so far; above-threshold tiles move to `scored/redundant/`. `pip install lucid-sisr[ann]` switches to HNSW for ~10M-tile scale.

**Finalize** — sequential naming + `master_lineage.csv`; excludes nothing silently: per-image pass rates are in the lineage CSV.

Tile sizes 128/256/512/1024 ship with a per-size validation curve (R² ≥ 0.94) that maps raw ICNet scores to the native 512px distribution; other sizes need `lucid-calibrate --input <representative folder>`.

---

## 7. Full CLI reference

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | *(required)* | Source images (scanned recursively) |
| `--output PATH` | `./lucid_output` | Final tiles destination |
| `--workspace PATH` | *(auto inside output)* | Scratch workspace (put on fast drive) |
| `--tile-size INT` | `512` | 128 / 256 / 512 / 1024 |
| `--threshold FLOAT` | `auto` = junk floor `0.30` | Min ICNet complexity; `0` = trust input fully |
| `--trim-bottom INT` | `0` | Drop lowest-complexity N% at dedupe |
| `--dedupe-threshold FLOAT` | `0.96` | Cosine similarity dedupe cutoff |
| `--multiscale` / `--no-multiscale` | **on** | Extra 0.75×/0.5×/0.25× scales |
| `--clipiqa-threshold FLOAT` | `0.0` (off) | Legacy single perceptual gate; superseded by `--iqa-gate` |
| `--iqa-gate METRIC:THRESHOLD` | *(off)* | pyiqa perceptual gate, repeatable, cascade order — e.g. `--iqa-gate topiq_nr:0.40 --iqa-gate qrealign-mini:0.4`; direction-aware (niqe-style lower-better metrics respected). Requires `pip install -e ".[iqa]"` |
| `--icnet PATH` | bundled | ICNet weights |
| `--no-calibrate` | off | Compare raw ICNet scores |
| `--workers INT` | `cpu_count - 2` | CPU workers for extraction |
| `--batch INT` | `auto` | GPU batch (0 = auto from VRAM) |
| `--max-vram-gb FLOAT` | `0.0` | VRAM cap |
| `--io-throttle-ms FLOAT` | `auto` | Sleep between writes (USB) |
| `--stage STAGE` | `all` | `all` / `ingest` / `dedupe` / `finalize` |
| `--strict` | off | Abort on first corrupt image |
| `--resume` | off | Continue from state DB |
| `--cleanup` | off | Delete scratch workspace on success |
| `--dry-run` | off | Validate + estimate only |
| `-v` | off | Debug logging |

---

## 8. Why filter this way

From the [BHI methodology](https://huggingface.co/blog/Phips/bhi-filtering) ([dataset](https://huggingface.co/datasets/Phips/BHI), [BHI100 validation](https://github.com/Phhofm/bhi100-sisr-iqa-metrics)):

- SISR nets optimize loss functions, not opinion scores — human-pleasing IQA (NIQE, BRISQUE, HyperIQA) doesn't track SR quality.
- Web GT arrives lossy: JPEG blocks, editor oversharpening, compression noise. Humans ignore these; SR learns them.
- **Removing only the worst tier beats keeping only the best**: BHI-filtered DF2K (69% smaller) outperformed the full dataset on PSNR/SSIM/DISTS; extreme reduction (2.3%) hurt.
- Optimal complexity thresholds vary per dataset — hence the data-driven `lucid-calibrate-gates` instead of one magic number.

LUCID = that insight as one fast command: technical cleanliness (artifact gates) + diversity (dedupe) + explicit complexity shaping, fully traceable.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | `--batch 4` or `--tile-size 256`, cap with `--max-vram-gb` |
| Dedupe OOM / slow at scale | `pip install lucid-sisr[ann]` (HNSW) |
| No GPU for my AMD/Intel card | ROCm torch wheel (AMD), `pip install lucid-sisr[xpu]` (Intel ARC), `pip install lucid-sisr[directml]` (Windows iGPU) |
| CPU-only | Works; ~10× slower, batch auto-clamped to 1–4 |
| Resume complains about changed gates | Intentional: never mix two gate configs in one dataset. Fresh run: drop `--resume` / delete workspace |
| `--tile-size` rejected | 128/256/512/1024 only; add sizes via `lucid-calibrate --input <folder>` |
| `lucid-gui` crashes | Needs tkinter: `sudo apt install python3-tk` |
| No images found | Extensions: png/jpg/jpeg/webp/tif/tiff (case-insensitive) |
| Disk full mid-run | Free space, re-run with `--resume` |
| ICNet weights missing | Bundled at `lucid/resources/icnet.pth`; fallback download from [IC9600 releases](https://github.com/umzi2/Dataset_Preprocessing/releases), pass `--icnet` |

---

## 10. Credits

- IC9600 complexity model: [IC9600](https://github.com/tinglyfeng/IC9600) by Tingly Feng.
- Pipeline, filter design, validation sets: [Philip Hofmann](https://github.com/Phhofm) (BHI, BHI100, PSISRD).

MIT License — see [LICENSE](LICENSE).
