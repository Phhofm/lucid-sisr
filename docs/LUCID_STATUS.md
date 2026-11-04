# LUCID — project status & operational context

**Last updated:** 2026-11-04 (v2 ingest in progress)
**Purpose:** survive conversation loss. Everything needed to resume lives here.

---

## 1. TL;DR

Building a filtered SISR ground-truth dataset family from CC0 sources, using the
LUCID pipeline in this repo.

- **Published (private HF repos):**
  - `Phips/lucid-pd12m-512` — 172,588 training tiles + x2/x4 LR
  - `Phips/lucid-val-512` — 136 whole-scene validation tiles + x2/x4 LR
- **In progress:** `Phips/lucid-pd12m-512-v2` — final finetune tier, merging
  PD12M round 2 (~184k projected) with the UHD-IQA training split (~1.5k),
  deduped against v1.
- **Training recipe:** pxhere set (from scratch) → `lucid-pd12m-512` (finetune)
  → `lucid-pd12m-512-v2` (final finetune). Validate on `lucid-val-512`;
  Urban100 stays the external benchmark.

---

## 2. What is running right now

| job | where | log | state |
|---|---|---|---|
| PD12M v2 ingest | bg process `pd12m_expand.py` | `/media/phhofm/SSD/pd12m_v2/stream.log` | ~parquet 39/125, ~56k tiles |
| v2 finalizer (waits, then gates UHD, merges, uploads) | bg process `v2_finalize.sh` | `/media/phhofm/SSD/pd12m_v2/merge.log` | waiting |

Completion marker: `/media/phhofm/SSD/lucid_v2_final/MERGE_DONE`.

Both are **persistent** background processes (survive session close, not reboot).
If lost, re-run the two commands in §7.

---

## 3. Pipeline changes made (v3.4.0 → v3.5.0)

Committed in this repo:

- **`--iqa-gate METRIC:THRESHOLD`** (repeatable, cascade, direction-aware) —
  any pyiqa metric; cheapest first; failed tiles never reach later gates.
- **Fail-closed gates**: if a gate errors (e.g. CUDA OOM), the tiles are
  rejected, not admitted. (Previously a gate error silently admitted them.)
- **IQA gates run in 8-tile sub-batches** (512px batches OOMed 12 GB GPUs).
- **`clipped_ratio` gate** (pixels on 0/255 rails, ceiling 40%).
- **`noise_ratio` removed** — structurally blind to photographic noise
  (visible noise measured 0.39, its lowest quartile). Noise is now caught by
  `niqe` / `topiq_nr`.
- **`--seed-pool PATH`** — dedupe seeded with a previous dataset's embeddings
  (cross-dataset duplicate exclusion).
- **`--simple-names`** — final tiles `00001.png…`, provenance in lineage CSV.
- Earlier fixes in this session: EXIF orientation, 16-bit warning,
  `--resume` refuses changed gates, dedupe same-batch escape, normalized
  blockiness, Hann-windowed aliasing, disk-guard estimate, hermetic tests, CI.

**Tests:** 84 passed / 4 skipped; `ruff` clean.

---

## 4. Current gate stack (calibrated)

| gate | threshold | notes |
|---|---|---|
| entropy | ≥ 5.2 | flat/empty |
| Laplacian var | 80–10,000 | destroyed blur / extreme sharpen |
| gradient energy | ≥ 0.70 | textureless |
| blockiness | ≤ 1.7 | normalized boundary/interior ratio |
| aliasing | ≤ 0.80 | Hann-windowed ring energy |
| clipped | ≤ 0.40 | blown/crushed frames |
| ICNet complexity | ≥ 0.30 | junk floor (auto default) |
| `topiq_nr` | ≥ 0.665 | val125 p01 |
| `clipiqa+` | ≥ 0.45 | keeps 100% of val125 |
| `niqe` | ≤ 6.5 | val125 p95 = 6.65 |
| `qrealign-mini` | ≥ 0.78 | val125 p01 = 0.77 |
| dedupe cosine | ≤ 0.96 | ResNet-18; seeded across dataset generations |

Key calibration facts:
- PSISRD val125 reference: topiq p01 = 0.665, niqe p95 = 6.65,
  qrealign p01 = 0.77, clipiqa+ ≥ 0.38 keeps 100%.
- **Whole-scene 512 vs native 1:1 crops matter enormously** for the learned
  metrics: on 12 UHD photos, native crops scored topiq 0.31 / niqe 8.8 /
  qrealign 0.28; whole-scene 512 scored 0.66 / 2.96 / 0.86. KonIQ-based
  models expect ~512px imagery; native 4K crops look over-sharp to them.

---

## 5. Datasets

### `Phips/lucid-pd12m-512` (v1 — training)
- 172,588 tiles from a 2% uniform sample of PD12M (250k of 12.4M images)
- Layout: `gt/` (87 shards), `x2/` (87), `x4/` (87), `gt_scores.csv`,
  `contact_gt.png`, `complexity_histogram.png`, `README.md`
- Shard N is aligned across gt/x2/x4 (same tile names) → `zip()` pairs them
- Private repo; flip public when satisfied
- Local copies: `/media/phhofm/SSD/pd12m_v1/` (tiles, ledgers, `v1_pool.npy`)

### `Phips/lucid-val-512` (validation)
- 136 whole-scene 512 tiles from 136 distinct UHD-IQA val+test photos
- GT = short side scaled to 512 + center crop (the PSISRD val convention)
- Source-disjoint from training and from UHD training split
- Diversity: one per photo, DINOv2 farthest-point, ≤5/cluster (60 clusters),
  ≤40/CLIP category, ≤40 green-dominant (final: 17 = 12.5%), cosine ≤0.90
- Layout: `gt/ x2/ x4/ val_scores.csv contact_val.png README.md`
- `val_scores.csv` includes `category`, `green_fraction`, `domain`-less
  (all uhdiqa), signal metrics, four IQA scores

### `Phips/lucid-pd12m-512-v2` (final finetune tier — building)
- **PD12M round 2**: rows 2000–4000 per parquet (250k fresh images,
  source-disjoint from v1) → ~184k projected tiles
- **UHD-IQA training split**: 4,269 4K CC0 photos, whole-scene 512 →
  ~1.5k projected tiles (these files are ~8 MB each, much less compressed
  than the val/test splits)
- Merged, one dedupe pass seeded with **all 172,588 v1 embeddings**
  (`v1_pool.npy`, cosine ≥0.96 excluded), plus internal dedupe
- Output: `gt/ x2/ x4/ scores.csv contact_v2.png README.md`;
  `scores.csv` has a `domain` column (pd12m / uhdiqa)
- Local: `/media/phhofm/SSD/pd12m_v2/`, `/media/phhofm/SSD/uhd_ws_train/`,
  `/media/phhofm/SSD/lucid_v2_final/`

### Measured novelty (v2 vs v1, 833-tile sample)
max cosine similarity to any v1 tile: median 0.873; 28.9% >0.90;
2.1% >0.95; 1.4% >0.96. Interpretation: new photos, same corpus distribution
(CLIP category histograms v1 vs v2 differ by only a few points). At the
seeded dedupe threshold 0.96 only ~1.4% are true near-duplicates.

---

## 6. Decisions & rationale

- **Keep ICNet at the junk floor 0.30**, not 0.45: val125's median complexity
  is 0.443, so 0.45 would reject half of reference-quality content.
- **No CLIP/category quotas on training sets** — quotas bias the training
  distribution. They are only used for small representative sets (val set).
- **Validation set uses whole-scene 512**, not native crops (metric domain
  fit + semantic completeness). Documented in the val card.
- **v2 is deduped against v1 at 0.96** (seed pool), not stricter: 0.94 would
  remove 4.1% and 0.90 29.5% of v2 for little duplicate benefit; source
  disjointness already guarantees new photos.
- **PD12M volume is not the constraint** — 12.4M images available; we consumed
  ~4%. Distribution breadth is the constraint, hence adding UHD-IQA.
- **Open Images excluded**: CC BY 4.0 (attribution), not CC0/PD.
  Other CC0 candidates if more breadth is needed: Wikimedia Commons PD/CC0,
  Smithsonian Open Access (CC0), NASA (PD).
- **PSISRD_val125 not republished**: its LR was produced with DPID
  (non-standard, not artifact-free). The new benchmark is UHD-IQA-based.

---

## 7. How to resume

If the background jobs are gone (reboot / session loss):

```bash
cd /home/phhofm/Documents/GitHub/lucid-sisr

# v2 ingest (resumable via parquet_offsets.json; skips completed offsets)
PD12M_OUT=/media/phhofm/SSD/pd12m_v2 \
PD12M_LEGACY_STATE=/media/phhofm/SSD/pd12m_v1/done_parquets.json \
PD12M_FINALIZE=1 \
PD12M_SEED_POOL=/media/phhofm/SSD/pd12m_v1/v1_pool.npy \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
venv/bin/python /tmp/kilo/pd12m_expand.py

# v2 finalizer: gate UHD training crops → merge → upload
bash /tmp/kilo/v2_finalize.sh
```

The ops scripts live in `/tmp/kilo` (ephemeral). **Copies are committed in
`scripts/ops/`**; if `/tmp` was wiped:

```bash
mkdir -p /tmp/kilo && cp scripts/ops/*.py scripts/ops/*.sh /tmp/kilo/
```

Note the scripts reference `/tmp/kilo/...` paths internally (e.g.
`v2_finalize.sh` calls `/tmp/kilo/merge_v2.py`), hence the copy-back step.

Monitoring:
```bash
tail -f /media/phhofm/SSD/pd12m_v2/stream.log
tail -f /media/phhofm/SSD/pd12m_v2/merge.log
find /media/phhofm/SSD/pd12m_v2/workspace/scored -name '*.png' | wc -l
```

---

## 8. Ops scripts (scripts/ops/)

| script | purpose |
|---|---|
| `pd12m_expand.py` | rolling PD12M driver, offset slices, seeded dedupe/finalize |
| `pd12m_stream.py` | v1 (round 1) driver, kept for reference |
| `v2_finalize.sh` | orchestrator: waits for v2, gates UHD, runs merge |
| `merge_v2.py` | merge v2 + UHD, seeded dedupe, rename, scores, LR, upload |
| `prep_uhd_train.py` | UHD training split → whole-scene 512 crops |
| `release_lib.py` | DINOv2 embeddings, constrained farthest-point, LR, packing, upload |
| `balance_val.py` | content classification (`classify`), green metric, cluster-capped selection |
| `make_val_whole.py` | builds the 136-tile whole-scene val set |
| `make_val_combined.py` | earlier native-crop val set (superseded) |
| `make_bench.py` | earlier UHD native-crop bench (superseded, repo deleted) |
| `make_lr_upload.py` | generate x2/x4 shards and upload |
| `pd12m_pack_upload.py` | pack one set into shards and upload |
| `pd12m_postprocess.py` | v1 tier/split/score-sheet postprocess |
| `pd12m_contact_sheets.py`, `pd12m_write_card.py` | v1 visuals + card |
| `build_v1_pool.py` | v1 embedding pool (`v1_pool.npy`) |
| `restructure_v1.py` | server-side repo restructure (gt/ layout) |
| `classify_content.py`, `compare_v1v2.py` | content audits |
| `pd12m_probe.py` | 100/1k-image yield probe |
| `qa_probe_t1t2.py`, `qa_probe_t3.py` | IQA calibration harnesses |

---

## 9. Environment facts

- Hub: `Phips/lucid-pd12m-512`, `Phips/lucid-val-512`,
  `Phips/lucid-pd12m-512-v2` (private; token cached at
  `~/.cache/huggingface/token`, CLI at `/home/phhofm/.local/bin/hf`)
- Venv: `/home/phhofm/Documents/GitHub/lucid-sisr/venv` (pyiqa from git,
  transformers 5.x, timm 1.0.30, matplotlib, webdataset not required)
- `python3.12-dev` installed (needed for Triton kernels);
  `causal-conv1d` not installed (needs full CUDA toolkit) — harmless warning
- Disk: `/media/phhofm/SSD` ~356 GB free at time of writing; ~90 GB per
  PD12M round, ~13 GB per x2/x4 pair copy
- Source corpora on disk: `/media/phhofm/SSD/uhdiqa/{training,validation,test}`

---

## 10. Open items / next steps

1. Finish v2 ingest → merging → upload (`MERGE_DONE`).
2. Eyeball `contact_v2.png` and `contact_val.png`.
3. Optional: `--seed-threshold` if stricter v1 exclusion is ever wanted
   (not implemented; 0.96 currently used for both internal and seeded dedupe).
4. Optional: more CC0 breadth (Wikimedia PD/CC0 architecture, Smithsonian
   CC0) if Urban100-type content coverage needs widening.
5. Training A/B: v1 vs v2 (and UHD weight) on a short run, checkpoint on
   `lucid-val-512`, report Urban100 — the only real proof of dataset gains.
