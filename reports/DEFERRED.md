# LUCID Static Analysis — Deferred Items

These findings are real but not addressed in the conservative pass. Each line says
what it would take, why we're not doing it in this pass, and the suggested approach.

## Cognitive complexity (Sonar S3776) — 7 deferred

These are all large function-level refactors that should be done on dedicated branches
with their own review and test runs. None of them is a runtime bug.

| File:Line | Function | Complexity | Why defer | Suggested approach |
|---|---|---|---|---|
| `lucid/core/filtering.py:35` | `run_ingest` | 64 | 175 lines covering signal filter, staging, scoring, calibration, resume, JPEG warning. | Extract 4 helpers: `_discover_and_register`, `_setup_model_context`, `_stream_score`, `_cleanup_staging`. |
| `lucid/core/filtering.py:268` | `extract_tiles_to_staging` | 24 | Signal-filtering + tile coord math + CSV writing. | Extract `_iter_signal_passing_tiles` and `_write_ingest_row`. |
| `lucid/core/finalize.py:14` | `run_finalize` | 22 | Final CSV + copy/move. | Extract `_build_lineage_rows` (the dict-of-dicts build) and `_commit_outputs` (copy/move + per-stage header). |
| `lucid/pipeline.py:22` | `run_pipeline` | 77 | Orchestrator for all 3 stages. 17 params. | Replace 17 params with a `PipelineConfig` dataclass; extract `_run_ingest_stage` / `_run_dedupe_stage` / `_run_finalize_stage`. |
| `lucid/scripts/calibrate_icnet.py:98` | `compute_observations` | 34 | ICNet calibration loop. | Extract `_score_one_image`, `_aggregate_per_size`. |
| `lucid/core/dedupe.py:172` | `run_dedupe` | 54 | HNSW + brute-force branches + batch scoring. | Extract `_build_index`, `_score_pool`, `_collect_redundant`. |
| `lucid/resources/__init__.py:86` | `resolve_device` | 19 | Device priority chain. | Extract `_try_*` helpers (one per backend). |
| `lucid/core/filtering.py:354` | `_score_tiles_gpu` | 17 | Streaming GPU scoring. | Extract `_load_batch_arrays` from inside the loop. |

> Note: `_score_tiles_gpu` and `resolve_device` are FIXED in the conservative pass
> (only 2-4 over the threshold; one extract each).

## Function with too many parameters (Sonar S107) — 1 deferred

| File:Line | Function | Params | Why defer | Suggested approach |
|---|---|---|---|---|
| `lucid/pipeline.py:23` | `run_pipeline` | 17 | The 17 params are already grouped by stage; collapsing to a dataclass changes every caller. | Introduce `PipelineConfig` dataclass; keep the 17-param signature as a thin wrapper for back-compat. |

## Type checker env noise (mypy) — 38 deferred

Every "Cannot find implementation" or "Library stubs not installed" is because the
linter venv doesn't have the runtime deps (torch, cv2, numpy, xxhash, platformdirs,
psutil, hnswlib). **Not a code defect.**

When users run with their normal venv, mypy should resolve these from the package
stubs. If you want zero noise, install into the linter venv:

```bash
pip install types-Pillow types-PyYAML types-tqdm types-psutil types-hnswlib
# torch/torchvision/cv2/numpy/xxhash/platformdirs ship their own stubs
```

## `# type: ignore` only — 1 deferred

| File:Line | Code | Why defer | Suggested approach |
|---|---|---|---|
| `lucid/calibrate.py:55` | `from calibrate_icnet import main` after `sys.path.insert` | Intentional runtime path injection. The lint comment would have to be `[import-not-found]` and is benign. | Add a `# type: ignore[import-not-found]` comment; or move `calibrate_icnet.py` to a proper subpackage and import normally. |
