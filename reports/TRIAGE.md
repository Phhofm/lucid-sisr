# LUCID Static Analysis Triage — Baseline

**Repo:** `/home/phips/Documents/GitHub/KiloApp2/kiloapp/lucid`  
**Date:** 2026-06-05  
**Tools:** ruff 0.15.15, mypy 2.1.0, semgrep 1.165.0, SonarQube 9.9.8 LTS (community)  
**Counts:** ruff 1, mypy 42 (38 env noise + 4 real), semgrep 0, sonar 14 → **57 total**  
**Raw output:** `reports/{ruff-baseline.json, mypy-baseline.txt, semgrep-baseline.sarif, sonar-baseline.json}`

Verdict legend: **FIX** = apply in this pass, **DEFER** = log to DEFERRED.md, **FP** = false positive (env-only).

## ruff (1)

| # | File:Line | Rule | Verdict | Reason |
|---|---|---|---|---|
| 1 | `lucid/scripts/calibrate_icnet.py:36` | I001 | **FIX** | Trivial import-sort. `ruff --fix`. |

## mypy — env noise (38) — not real, see `reports/mypy-baseline.txt`

`Cannot find implementation or library stub for module named "torch"/"torch.nn"/"torchvision"/"cv2"/"numpy"/"xxhash"/"platformdirs"` × 30  
`Library stubs not installed for "psutil"/"hnswlib"` × 2  
None of these are code defects — runtime deps are missing from the linting venv. Documented in DEFERRED.md.

## mypy — real (4)

| # | File:Line | Code | Verdict | Reason |
|---|---|---|---|---|
| M1 | `lucid/gui.py:456` | [assignment] `self.process` declared `None`, assigned `Popen[str]` | **FIX** | Real type error. Annotate field as `Popen[str] \| None`. |
| M2 | `lucid/core/finalize.py:110-112` | [operator]/[attr-defined]/[call-overload] `rec["original_name"]` typed `object` so `.split()` and `.get(...)` are rejected | **FIX** | Cast to `str` (CSV row is `dict[str, str]`). |
| M3 | `lucid/core/filtering.py:443, 593` | [name-defined] `np` is not defined; the code uses `"np.ndarray"` in two string annotations but `numpy` was never imported | **FIX** | **Real latent bug** — the `# noqa: F821` comment hides it from ruff. Add `import numpy as np`. |
| M4 | `lucid/core/filtering.py:603, 673` | [var-annotated] `record` needs annotation | **FIX** | Trivial: `record: dict[str, str] = ...`. |
| M5 | `lucid/calibrate.py:55` | [import-not-found] `from calibrate_icnet import main` after `sys.path.insert` | **DEFER** | Intentional sys.path injection. Add `# type: ignore[import-not-found]`. |

## semgrep (0)

## SonarQube (14)

| # | File:Line | Rule | Sev | Verdict | Reason |
|---|---|---|---|---|---|
| S1 | `lucid/core/filtering.py:35` run_ingest | S3776 cognitive 64→15 | CRITICAL | **DEFER** | 175 lines, 6 nested concerns. Out of scope for a sweep; would touch all 4 ingest helpers. |
| S2 | `lucid/core/filtering.py:268` extract_tiles_to_staging | S3776 cognitive 24→15 | CRITICAL | **DEFER** | 80 lines, signal filter + tile coord math. Not a one-evening job. |
| S3 | `lucid/core/filtering.py:354` _score_tiles_gpu | S3776 cognitive 17→15 | CRITICAL | **FIX** | Only 2 over; one well-scoped extract fixes it. |
| S4 | `lucid/core/finalize.py:14` run_finalize | S3776 cognitive 22→15 | CRITICAL | **DEFER** | CSV write + copy/move split. Substantive refactor. |
| S5 | `lucid/pipeline.py:22` run_pipeline | S3776 cognitive 77→15 | CRITICAL | **DEFER** | 17 params, 250+ lines, all 3 stages. Needs a config struct. |
| S6 | `lucid/scripts/calibrate_icnet.py:98` compute_observations | S3776 cognitive 34→15 | CRITICAL | **DEFER** | Calibration training loop. Heavy. |
| S7 | `lucid/core/dedupe.py:172` run_dedupe | S3776 cognitive 54→15 | CRITICAL | **DEFER** | HNSW + brute-force branches + scoring. |
| S8 | `lucid/resources/__init__.py:86` resolve_device | S3776 cognitive 19→15 | CRITICAL | **FIX** | Already a chain of `if`s; one extract fixes it. |
| S9 | `lucid/core/filtering.py:123` | S3358 nested conditional in model_ctx block | MAJOR | **FIX** | `batch_size if batch_size else recommend_batch_size(...)` — extract. |
| S10 | `lucid/pipeline.py:23` run_pipeline | S107 17 params > 13 | MAJOR | **DEFER** | Real, but requires a config dataclass. See DEFERRED. |
| S11 | `lucid/pipeline.py:198` | S1192 "*.png" literal duplicated 3× | CRITICAL | **FIX** | Hoist to module constant. |
| S12 | `lucid/scripts/calibrate_icnet.py` | I001 (overlaps with ruff #1) | — | — | Same finding, listed under ruff. |
| S13 | `lucid/gui.py:75` | S1192 "Sub.TLabel" duplicated 3× | CRITICAL | **FIX** | Module constant. |
| S14 | `lucid/gui.py:100` | S1192 "Secondary.TButton" duplicated 6× | CRITICAL | **FIX** | Module constant. |
| S15 | `lucid/platform.py:15` | S5713 redundant `PermissionError` in `(AttributeError, PermissionError, OSError)` | MINOR | **FIX** | `PermissionError` ⊂ `OSError`; drop it. |

## Summary

| Verdict | Count |
|---|---|
| **FIX** | 12 (1 ruff + 4 mypy + 7 sonar) |
| **DEFER** | 9 (1 mypy + 8 sonar) |
| **FP/env** | 38 (mypy env noise) |
| **Unchanged** | 0 semgrep, 0 critical security |
