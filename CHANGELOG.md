# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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