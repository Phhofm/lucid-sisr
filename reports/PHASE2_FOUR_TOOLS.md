# LUCID — Proof of Concept: 4 Extra Static Analysis Tools

This is the second pass on LUCID after the initial sweep (see
`TRIAGE.md` and `BLOG_STATIC_FIRST_LLM_TRIAGE.md`). It runs the four tools
the first pass didn't include, to validate the "8-tool belt" idea for the
proposed general-purpose pipeline.

## Tool installation (10 min total)

```bash
pipx install pip-audit vulture deptry
curl -sSfL "https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz" \
    | tar -xz -C /home/phips/.local/bin gitleaks
```

## Results on LUCID

| Tool | Findings | Verdict | Notes |
|---|---|---|---|
| **`pip-audit`** | 0 | ✅ clean | No known CVEs in `numpy/opencv-python/pillow/platformdirs/psutil/torch/torchvision/tqdm/xxhash` |
| **`gitleaks`** | 0 | ✅ clean | No committed secrets across 9 commits |
| **`vulture`** | 28 raw, ~15 real, ~13 FP | ⚠ needs config | Huge FP rate on Python protocol methods (`__getattr__`, `forward`, `__call__`). Needs a whitelist. |
| **`deptry`** | 17 raw, 1 real, 16 FP/config | ⚠ needs config | Catches 1 real bug: `matplotlib` imported but undeclared. Most FPs are deptry not knowing `package → module` renames. |

## What's worth fixing from this pass

### `matplotlib` is imported but not declared (real bug)

`lucid/core/filtering.py:768-770` does `import matplotlib` and `import matplotlib.pyplot as plt` inside a `try:` block in `_write_score_artifacts`. If the user doesn't have matplotlib installed, they get a clean warning — but the import is real and the dep should be declared. Added `[plot]` extra in `pyproject.toml`:

```toml
plot = ["matplotlib>=3.7"]
```

### vulture false-positive noise is real

13 of 28 vulture findings are unavoidable FPs in any Python codebase:

- `__getattr__` / `__dir__` (PEP 562 protocol)
- `forward` in every `nn.Module` subclass (5 in `icnet.py` alone)
- Deprecated but-public `_score_tiles_gpu` (kept for back-compat, marked in docstring)
- Cross-file call sites that vulture can't see into `tests/`

A whitelist file (`.vulture-whitelist.py`) is provided that imports the false-positives to register them. The vulture README documents the format; ours is kept minimal.

## What these tools *don't* add to the LUCID pass

**Honest assessment:** on a clean, well-maintained project like LUCID, the four extra tools found:

- 1 real bug (matplotlib)
- 0 security issues
- 0 dead code that's actually dead
- 0 CVEs

The value is mostly **preventive**: when this repo grows or when the user starts a new one, these tools catch the *categories* of issues the original four don't. `pip-audit` catches CVE regressions when deps change. `gitleaks` catches secrets that get committed by accident. `vulture` catches code that becomes unreachable during a refactor. `deptry` catches dep-management rot.

## Configuration files added

| File | Purpose |
|---|---|
| `pyproject.toml` | Added `[tool.deptry.package_module_name_map]` and `[tool.deptry.known_first_party]`. Added `plot = ["matplotlib>=3.7"]` extra. |
| `.vulture-whitelist.py` | Re-imports Python protocol methods, PyTorch `forward`s, and cross-file test references. |
| `reports/{vulture,pip-audit,deptry,gitleaks}-baseline.txt` | Raw output. |

## What this proves for the general-purpose pipeline

- **8 tools is the right number.** More would mean diminishing returns and a higher FP-fixing cost.
- **Per-language tool defaults are necessary.** Python's vulture needs a whitelist from day 1; JS's ESLint needs `eslint:recommended`; Go's `golangci-lint` ships with sane defaults; Java's Checkstyle is a tar pit of opinionated config.
- **Every tool needs a config layer.** The polyglot project will need a "configure these tools" step that:
  1. Auto-detects the language(s) in the repo
  2. Generates a starter config (`.vulture-whitelist.py`, `eslint.config.js`, `checkstyle.xml`, etc.)
  3. Lets the user override the default exclusions/severity
- **Output unification is the hard part.** Each tool emits its own JSON/SARIF/text format. The pipeline needs a normalized finding schema before any of this is usable to an LLM.
