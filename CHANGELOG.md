# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.2.0] - 2026-10-21

### Added
- **Per-scale pass-rate stats** (`scores/scale_pass_rates.txt`): seen /
  signal-passed / ICNet-passed per multiscale scale tag, so you can
  measure whether small scales earn their compute. (First real run:
  0.25x produced 0 passing tiles on smooth content.)
- **`run_config.json` manifest** next to `master_lineage.csv`: full gate
  configuration + counts + lucid version + UTC timestamp.

### Changed
- **Multiscale resampling unified to Pillow BICUBIC** — the same kernel
  as the >8192px downscale path (previously a second, more overshooting
  cv2 bicubic).
- Dedupe: ResNet-18 features computed in FP16 on CUDA/XPU and tile
  decoding threaded (8 workers) — throughput; kept-set unchanged.

## [3.1.2] - 2026-10-21

### Fixed
- **EXIF orientation respected**: phone/Flickr photos carry orientation
  tags; tiles were extracted from the sideways-stored pixel data. GT
  orientation is now pixel-correct.
- 16-bit sources (mode `I;16`) now warn — PIL silently collapses them
  during RGB conversion.

### Added
- **--resume now refuses changed filtering gates**: resuming with a
  different `threshold`/`clipiqa`/`multiscale`/etc. than the recorded
  run aborts instead of silently mixing two gate configurations in one
  dataset. Gate config is stored in the state DB.
- GitHub Actions CI: ruff + full test suite on push/PR (CPU torch,
  Python 3.11/3.12).

### Changed
- `tests/test_resume.py` is hermetic (was hardcoding an external input
  folder); suite wall time down from ~70 s to ~44 s.

## [3.3.0] - 2026-10-21

### Changed
- **Auto --threshold is a junk floor (0.30), not a quality bar**: the old
  auto-0.45 was a top-half complexity cut that silently dropped half of
  curated GT (and would nuke low-complexity domains like anime).
  0.30 is calibrated on curated GT (PSISRD val125 p1 = 0.303, keeps
  99.2%); complexity *shaping* stays explicit via `--trim-bottom N`
  (small nets wanting high-complexity exposure: `--trim-bottom 50`).

### Added
- `lucid-calibrate-gates` now also reports the ICNet complexity
  distribution (p1/p5/p50/p99) of the reference folder with a suggested
  `--threshold`, so the floor and shaping can be derived per domain.

## [3.2.1] - 2026-10-21

### Fixed
- **Signal gates recalibrated on real GT data** (first live PSISRD
  val125 run kept only 10/125 curated tiles):
  - `aliasing_max` 0.40 -> **0.80**: real photos have a rich 1/f
    spectrum (measured 0.31-0.74 on curated GT); the 0.40 gate came
    from synthetic fixtures whose spectrum decays too fast. Oversharpening
    capture now rests mainly on `lap_var_max`.
  - `noise_ratio_max` 0.65 -> **1.35**: clean GT measured 0.32-1.00
    (median 0.66), so the old gate rejected half of curated GT even
    before this release. Now a heavy-noise pre-filter (sigma~40+
    separates); moderate noise remains the ICNet/dedupe stages' job.
- Removed the per-image "smaller than 1.1x tile-size" warning spam
  (fired 125x on an all-512px dataset); the 0-tiles-passed warning
  already covers images that produce nothing.

### Measured
- PSISRD val125 hr/ rerun: 125/125 signal-passed, 60/125 finalized
  (48% ICNet retention at the 0.45 auto-threshold).

## [3.1.1] - 2026-10-21

### Fixed
- Dedupe: duplicates processed inside the same batch no longer escape the
  audit; comparisons now stream per tile against everything accepted so far
- `tiles_scored` summary stat was never populated (always 0)

### Changed
- **Blockiness rewritten** as the normalized boundary/interior gradient ratio
  (Wang-style). The old unnormalized sum could not detect JPEG artifacts at
  all (even a Q30 re-encode scored <6 against the 45.0 cutoff). New gate:
  `blockiness_max = 1.7`; clean content measures ~1.0–1.4, JPEG flats ~2.0.
- **Aliasing ratio windowed** with a Hann window before the FFT. The
  unwindowed ring was dominated by tile-boundary leakage and rejected ordinary
  sharp natural content. New gate: `aliasing_max = 0.40` — rejects broadband
  HF clutter (UI/text/screen tiles, ringing), keeps clean textures.
- **CLIP-IQA is now opt-in** (`--clipiqa-threshold`, default 0.0). A
  perceptual metric silently gating on install state made runs
  non-reproducible; the enabled gate set is recorded in
  `workspace/scored/threshold_used.txt`.
- E2E test suite is hermetic (synthetic fixtures, no external input folder).

### Added
- `--trim-bottom N`: percentile prune dropping the lowest-complexity N% of
  tiles at the dedupe stage (BHI finding: removing only the worst tiles beats
  keeping only the best). Trimmed tiles are moved to `scored/redundant/`.

## [3.1.0] - 2026-07-07

### Fixed
- CLIP-IQA per-tile score indexing bug: tiles failing ICNet gate now correctly index into CLIP-IQA scores array
- `--batch` and `--io-throttle-ms` parameters now honored in streaming scoring path
- HNSW dedupe index growth now preserves vectors via `resize_index()` instead of rebuilding
- Pass-count statistics now correctly report tiles passed ICNet scoring
- Tile-name parsing now robust for stems containing `_s` via `source_stem_for_tile()` helper
- Lineage CSV now records true source-image paths instead of workspace paths
- Dedupe stage no longer rescans `redundant/` or counts unreadable tiles as kept
- Worker count default on 2-core machines now uses `max(1, cpu_count - 2)` correctly

### Changed
- Configuration singleton (`LucidConfig`) removed; all parameters now explicit in `run_ingest()` and `run_pipeline()`
- Added `clipiqa` optional dependency group for CLIP-IQA filtering
- Bundled `icnet.pth` now ships in wheels/sdists via MANIFEST.in update

### Removed
- Deprecated `_score_tiles_gpu` function (streaming path only)
- Duplicate `scripts/calibrate_icnet.py` (use packaged copy at `lucid/scripts/`)
- Standalone `score_and_filter.py` (superseded by `--clipiqa-threshold`)

## [3.0.1] - 2026-06-05

### Changed
- `lucid.core.__init__` now lazy-imports `run_dedupe`, `run_ingest`, `run_finalize` (PEP 562). `import lucid.core` no longer requires `torch` to be installed; only attribute access does. This unblocks lightweight test venvs and docs builds.
- GUI ttk style names hoisted to module-level constants (`STYLE_SUB_LABEL`, `STYLE_SECONDARY_BUTTON`).
- `pipeline.py` uses `PNG_GLOB` constant instead of three `"*.png"` literals.

### Fixed
- `lucid/core/filtering.py` now imports `numpy as np` (the two `"np.ndarray"` string annotations are no longer forward-references to a missing name).
- `lucid/gui.py` types `self.process` as `Popen[str] | None` (was implicitly `None`).
- `lucid/core/filtering.py:run_ingest` adds an `assert model_ctx is not None` inside the non-dry-run branch, so mypy can narrow the union.
- `lucid/core/filtering.py:_score_tiles_gpu` (deprecated) extracts the inner PNG-decode loop into `_load_batch_items`; complexity 17 → under 15.
- `lucid/resources/resolve_device` extracts `_auto_pick` and `_honour_explicit` helpers; complexity 19 → under 15.
- `lucid/platform.py:set_process_priority` drops the redundant `PermissionError` from the except tuple (`OSError` already covers it).
- `lucid/core/finalize.py:run_finalize` annotates `records: list[dict[str, object]]` so `stem.split(...)` type-checks; stats writes are wrapped in `str(...)` to keep the row dict homogeneous.

### Added
- `docs/BLOG_STATIC_FIRST_LLM_TRIAGE.md` — essay on the static-first / LLM-triage workflow, walked through LUCID's actual findings.
- `reports/TRIAGE.md` and `reports/DEFERRED.md` — per-finding verdicts with reasoning.
- `reports/{ruff,mypy,semgrep,sonar}-{baseline,after}.{json,txt,sarif}` — raw output for the sweep.
- `sonar-project.properties` — local SonarQube 9.9 LTS Community project config.

## [3.0.0] - 2026-06-02

### Added
- Unified single-command pipeline: `python -m lucid --input <dir> --output <dir>`
- Automatic ICNet model download on first run
- Three-stage pipeline: ingest (signal filter + ICNet scoring) → dedupe (ResNet18) → finalize (sequential naming + lineage CSV)
- SQLite-based state database for crash recovery and resume (`--resume`)
- GPU acceleration with FP16 for ICNet and ResNet18
- CPU fallback with automatic batch-size clamping and warning
- MPS (Apple Silicon) support with automatic FP32 fallback
- Multiprocessing for parallel tile extraction
- Corrupt image detection with eager decode validation
- Large image handling (auto-downscale > 8192px)
- Disk-space estimation and pre-flight guard
- USB/external drive auto-detection with I/O throttling
- `--cleanup` flag to remove workspace after completion
- `--dry-run` mode for validation without writing
- `--stage` flag to run individual pipeline stages
- `--strict` flag to abort on first corrupt/unreadable image (fully wired)
- Tkinter GUI wrapper (`lucid-gui`) with tile-size/stage selectors
- Comprehensive CLI with `--help` documentation
- Unit tests for metrics, state tracking, and validation
- One-click launcher scripts (`run.sh` for Linux/macOS, `run.bat` for Windows)

### Changed (v3.0 release polish)
- **Multi-size tile support with calibration**: 128/256/512/1024 px tile sizes are now all first-class. A per-size linear calibration table (`lucid/resources/icnet_calibration.json`, R² ≥ 0.94) is shipped in the package and auto-applied at non-native sizes.
- `--tile-size` no longer hard-rejects non-512 — it now picks the calibration entry from the table, logs the suggested `--threshold`, and proceeds.
- New `score_calibrated` column in `scored/ingest_scores.csv` next to the raw `complexity_score` for full provenance.
- New `--no-calibrate` flag to opt out of the calibration transform.
- New `lucid-calibrate` console script: regenerate the calibration table from your own representative images.
- Optional extras in `pyproject.toml`: `[ann]` (hnswlib for ~50× faster dedupe, scales to 10M+ tiles), `[xpu]` (Intel GPUs), `[directml]` (Windows), `[rocm]`, `[all]`.
- Dedupe now uses HNSW (hnswlib) when available, with a brute-force cosine-matmul fallback for users without the optional dep.
- Cross-platform GPU detection covers NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS, and Windows DirectML with auto-fallback to CPU.
- ICNet architecture confirmed size-flexible: the assert was relaxed from `== self.size1` to a square-input check.
- Finalize tag now resolves to the source dataset name (`lucid_workspace_<input>/scored/00001_input.png`) instead of the literal string "scored".
- GUI: 128 px added to the tile-size dropdown, live calibration status badge (Native / Calibrated / Approximate / Uncalibrated) next to the dropdown, and a "Calibrate…" button that spawns the `lucid-calibrate` subcommand in the console pane.
- Tests: 38 pass + 3 HNSW skipped (hnswlib build blocked on Linux sandbox without `python3-dev`; prebuilt wheels work on user machines).
- Output workspace path documented correctly as `lucid_workspace_<input_dir_name>/{raw_tiles,scored,scores}/.lucid/lucid_state.db` (README + AGENTS.md aligned to code).

### Changed
- Improved robustness: fixed file-handle leaks in dedupe, collision-safe duplicate naming
- Buffered tile writes now use context manager for crash-safe flushing
- Fixed ICNet upsampling scale (scale=8) to correctly concatenate detail and context branches
- Fixed MPS/CPU dtype handling (FP32 fallback prevents crash on Apple Silicon and CPU-only machines)
- Fixed GUI launch crash and thread-safety for live log streaming
- Wired up `--strict` flag end-to-end for abort-on-corrupt-image behavior
- Added tile-size and stage selectors to GUI
- Added python3-tk auto-install on Linux happy path in run.sh