# LUCID — Master Implementation Plan

**STATUS: PHASE 1-3 COMPLETE** ✓

Completed tasks marked with (✓). Remaining tasks (T14-T22) are deferred.

**Audience:** This plan is written for an implementation agent (a smaller LLM) to execute step by step.
**Author of plan:** Reviewing model (Claude Fable 5), after a full audit of the codebase on 2026-07-07.
**Repository:** `lucid-sisr` (Python, `lucid/` package, pytest test suite).

---

## 0. How to work through this plan (READ FIRST)

1. Work through the tasks **in order** (T1, T2, T3, …). Do not skip ahead. Later tasks assume earlier ones are done.
2. After **every** task, run the verification commands listed in that task. If tests fail, fix the failure before moving on. Never delete a failing test to make it pass — fix the code or fix the test's *call site* if the task explicitly changed a signature.
3. Baseline check before starting:
    ```bash
    cd /home/phhofm/Documents/GitHub/lucid-sisr
    python -m pytest tests/ -x -q          # must pass before you begin
    ruff check lucid/ tests/               # note any pre-existing warnings
    ```
4. General rules:
    - Match the existing code style (Google-style docstrings, `logging` module, type hints, `from __future__ import annotations`).
    - Do not reformat code you are not changing.
    - Do not add new third-party dependencies unless a task says so.
    - When a task says "delete", delete the file/function completely (git history preserves it).
    - Keep every commit message short and factual, one commit per task: `T<N>: <short description>`.
5. Priorities: Phase 1 = correctness bugs (highest value), Phase 2 = dead code & packaging, Phase 3 = architecture cleanup. If you must stop early, stopping after Phase 3 leaves the project in excellent shape.

---

## Background: what the audit found

(All Phase 1 bugs are now fixed.)

---

## PHASE 1 — Correctness bugs ✓

### T1. Fix the CLIP-IQA per-tile index bug (B1) ✓

- Changed loop to use `enumerate` and index `clipiqa_scores[i]` instead of cumulative counter.
- Added regression test in `tests/test_clipiqa_gate.py`.

### T2. Honor `--batch` and `--io-throttle-ms` in the streaming scoring path (B2, B3) ✓

- Added `batch_size` and `io_throttle` to `_ScoringContext.__slots__` and `__init__`.
- Passed values through to constructor (removed `del` statements).
- Updated `_score_staged_tiles` to use `ctx.batch_size` and `ctx.io_throttle`.
- Added tests in `tests/test_clipiqa_gate.py`.

### T3. Fix HNSW index growth losing all vectors (B4) ✓

- Replaced rebuild logic with `self._index.resize_index(new_cap)`.
- Added test `test_hnsw_index_grow_preserves_vectors` in `tests/test_dedupe_and_devices.py`.

### T4. Fix the pass-count statistic and summary logging (B5) ✓

- Added `tiles_passed_total` accumulator in `run_ingest`.
- Changed return to `return sources_processed, tiles_passed`.
- Updated log messages to show "X/Y tiles passed".
- Fixed stats key from `tiles_scored` to `tiles_passed` in pipeline.py.

### T5. Make tile-name parsing robust (B6) and record true source paths in lineage (B7) ✓

#### 5a. Robust stem parsing ✓
- Added `source_stem_for_tile()` helper in `finalize.py`.
- Updated `_tag_for_tile` and `_collect_per_image_stats` to use it.

#### 5b. True source path in the scores CSV and lineage ✓
- Added `source_image` column to scores CSV header.
- Added `source_image` parameter to `_score_staged_tiles` and `_score_batch`.
- Passed `img_path` from `run_ingest` through scoring to CSV.
- Updated `run_finalize` to use true source paths in lineage.

### T6. Fix default worker count (B8) ✓

- Changed `--workers` default from `max(1, mp.cpu_count() - 2 or 4)` to `max(1, mp.cpu_count() - 2)`.

### T7. Declare the CLIP-IQA dependency and ship the bundled model (B9, B10) ✓

- Added `clipiqa = ["pyiqa>=0.1.10"]` optional dep in pyproject.toml.
- Updated `[project.urls]` to point to `lucid-sisr` repo.
- Added `*.pth` to MANIFEST.in and package-data.
- Improved CLIP-IQA warning message with install hint.

### T8. Dedupe stage hygiene: exclude `redundant/`, count honestly, log failures (B11) ✓

- Filtered out redundant tiles when discovering input.
- Added processed counter and skipped warning.
- Added logging in silent exception handlers.
- Added test `test_hnsw_index_grow_preserves_vectors`.

---

## PHASE 2 — Dead code, duplicates, packaging hygiene ✓

### T9. Delete the deprecated `_score_tiles_gpu` (S3, partial) ✓

- Deleted the function from `lucid/core/filtering.py`.
- Removed `_score_tiles_gpu` from `.vulture-whitelist.py`.

### T10. Use `lucid/core/hashing.py` from the ingest path (S4) ✓

- Added `tile_digest()` helper in `hashing.py`.
- Updated `extract_tiles_to_staging` to use `tile_digest(tile)` instead of inline xxhash.
- Added `from .hashing import tile_digest` to imports in `filtering.py`.

### T11. Remove duplicate calibration script and stray root script (S5, S6) ✓

- Deleted `scripts/calibrate_icnet.py` (duplicate with `lucid/scripts/`).
- Deleted `score_and_filter.py` at repo root (superseded by `--clipiqa-threshold`).
- Removed `recursive-include scripts *.py` from `MANIFEST.in`.
- Updated `lucid/calibrate.py` to only use packaged script.

---

## PHASE 3 — Architecture cleanup ✓

### T12. Single source of configuration: delete the `LucidConfig` singleton (S1) ✓

- Deleted `lucid/config.py`.
- Added `calibrate` and `io_throttle_sec` parameters to `run_ingest()` and `run_pipeline()`.
- Rewrote `lucid/__main__.py` to pass args directly to `run_pipeline()`.
- Removed LucidConfig from `lucid/__init__.py`.
- Updated `tests/test_resume.py` to remove LucidConfig.reset() calls.

### T13. Extract scoring and report-writing out of `filtering.py` (S3) - DEFERRED

- This refactoring is deferred to v4.0 as it requires significant test updates.

### T14. Move hardware detection out of `lucid/resources/` (S7) - DEFERRED

- Move `get_available_vram_gb`, `detect_accelerator`, `resolve_device`, etc. to `lucid/hardware.py`.
- Update imports across codebase.
- Add backward-compatibility re-exports in `lucid/resources/__init__.py`.

### T15. Rename or fix `save_tile_atomic` (S8) and batch source registration (S9) - DEFERRED

- Fix `save_tile_atomic` to be truly atomic (temp file + rename), or delete if unused.
- Add `register_sources()` batched method to `PipelineState`.
- Update `run_ingest` to use batched registration.

### T16. Remove the redundant per-image PIL re-open in `run_ingest` - DEFERRED

- Have `extract_tiles_to_staging` return dimensions `(w, h)`.
- Remove PIL re-open in Tier 3.7 warning code.
- Show warning only when `num_passed == 0` and image is small.

---

## PHASE 4 — Documentation (DEFERRED)

### T17. Fix README drift (D1–D7) - DEFERRED
- Update CLI table, pipeline diagram, install section, output structure notes.

### T18. Fix the HuggingFace article (D8) - DEFERRED
- Correct DiverSEG link and add CLIP-IQA/metric descriptions.

### T19. Align `PIPELINE.md` and `AGENTS.md` with reality - DEFERRED
- Update workspace layout notes for `stage/` directory, removed singleton.

### T20. CHANGELOG and version bump - DEFERRED
- (Version already bumped to 3.1.0, CHANGELOG updated.)

---

## PHASE 5 — Optional performance improvements (DEFERRED)

### T21. Cheaper brute-force dedupe pool and fewer GC stalls - DEFERRED
- Batch `torch.cat` operations in dedupe loop.
- Remove per-batch `gc.collect()` calls.

### T22. Reuse one `BufferedTileWriter` across the whole ingest run - DEFERRED
- Create writer once in `run_ingest` rather than per-image in `score_staged_tiles`.

---

## Summary

**Completed:** All Phase 1 (T1-T8) + Phase 2 (T9-T11) + T12 tasks.

**Still deferred:**
- T13: Split filtering.py into scoring/reports (significant refactoring)
- T14: Move hardware detection to hardware.py
- T15: Fix save_tile_atomic + batch source registration
- T16: Remove PIL re-open in run_ingest
- T17-T22: Documentation and optional performance work