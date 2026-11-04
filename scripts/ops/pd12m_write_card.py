"""Generate the HuggingFace dataset card (README.md) from measured numbers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path("/media/phhofm/SSD/pd12m_v1")


def count(csv_name: str) -> dict[str, float]:
    rows = list(csv.DictReader(open(OUT / csv_name)))
    if not rows:
        return {}
    comp = [float(r["complexity"]) for r in rows]
    return {"n": len(rows), "median_complexity": sorted(comp)[len(comp) // 2]}


def main() -> None:
    s = json.loads((OUT / "sets_summary.json").read_text())
    sets = s["sets"]
    full = count("pd12m_train_full_scores.csv")
    small = count("pd12m_train_small_scores.csv")
    mini = count("pd12m_train_mini_scores.csv")
    bench = count("pd12m_bench_scores.csv")
    tv = count("pd12m_trainval_scores.csv")

    n_last = max((sets["pd12m_train_full"] + 1999) // 2000 - 1, 0)

    def gb(n: int) -> str:
        return f"{n * 0.5 / 1024:.0f} GB"

    card = f"""---
license: cdla-permissive-2.0
language:
- en
tags:
- super-resolution
- image-upscaling
- sisr
- training-dataset
- validation-dataset
- pd12m
pipeline_tag: image-to-image
datasets:
- Spawning/PD12M
source_datasets:
- Spawning/PD12M
---

# LUCID PD12M 512 — filtered SISR ground-truth tiles

**What this is.** {sets['pd12m_train_full']:,} clean 512×512 ground-truth tiles for
single-image super-resolution, filtered from a 2% uniform sample of
[PD12M](https://huggingface.co/datasets/Spawning/PD12M) (250k images of 12.4M).

**Why it exists.** Web imagery is full of compression blocks, grain,
oversharpening halos and blown frames. A network trained on that learns to
reproduce it. Every tile here passed six signal gates and four learned
perceptual gates calibrated on curated GT — so what the model imitates is
detail, not damage.

---

## 1. What you get

| Set | Tiles | Use |
|---|---|---|
| `pd12m_train_full/` | {sets['pd12m_train_full']:,} ({gb(sets['pd12m_train_full'])}) | from-scratch / pretraining |
| `pd12m_train_small/` | {sets['pd12m_train_small']:,} ({gb(sets['pd12m_train_small'])}) | finetune, faster |
| `pd12m_train_mini/` | {sets['pd12m_train_mini']:,} ({gb(sets['pd12m_train_mini'])}) | finetune, dense detail |
| `pd12m_trainval/` | {sets['pd12m_trainval']:,} | validation every N iterations |
| `pd12m_bench/` | {sets['pd12m_bench']:,} | final benchmark comparisons |

Train sets get progressively more information-dense:
full median complexity {full.get('median_complexity', 0):.3f} →
small {small.get('median_complexity', 0):.3f} →
mini {mini.get('median_complexity', 0):.3f}.

**Validation sets are source-disjoint** from training and from each other:
no PD12M photo contributes tiles to two sets. Both were picked with
farthest-point (coreset) selection so they cover the visual space instead of
being a random pile.

---

## 2. Quick start

```bash
hf download Phips/lucid-pd12m-512 --repo-type dataset --local-dir .
```

```python
import webdataset as wds

train = (wds.WebDataset("pd12m_train_full/shard-{{00000..00{n_last:03d}}}.tar")
         .decode("pil"))
val = wds.WebDataset("pd12m_trainval/shard-00000.tar").decode("pil")
```

Every set ships a score sheet (`pd12m_<set>_scores.csv`) with the final name,
source id and every filter score — so you can re-tier or audit any subset
without re-running the pipeline.

---

## 3. How filtering works

Cheap gates first; failed tiles never reach the expensive ones:

| Tier | Gate | Rejects |
|---|---|---|
| 1 | entropy ≥ 5.2 | flat / empty |
| 1 | Laplacian var 80–10,000 | destroyed blur, extreme sharpen |
| 1 | gradient energy ≥ 0.70 | textureless |
| 1 | blockiness ≤ 1.7 | JPEG blocks |
| 1 | aliasing ≤ 0.80 | UI stripes, ringing |
| 1 | clipped pixels ≤ 40% | blown / crushed frames |
| 2 | ICNet complexity ≥ 0.30 | junk floor |
| 3 | `topiq_nr` ≥ 0.665 | degradation (jpeg, blur, noise) |
| 4 | `clipiqa+` ≥ 0.45 | semantic implausibility |
| 5 | `niqe` ≤ 6.5 | unnatural statistics / grain |
| 6 | `qrealign-mini` ≥ 0.78 | final holistic verdict |

Thresholds were calibrated against a curated GT reference (PSISRD val125):
each learned floor keeps ~99% of that reference while rejecting the measured
degradation classes. `niqe` exists because a visibly noisy tile scored 0.39
on the old handcrafted noise proxy (its lowest quartile) while measuring 11.9
on NIQE (reference band 3.7–5.0).

Dedup: ResNet-18 embeddings, cosine ≥ 0.96 collapses near-duplicates
(cross-scale multiscale tiles included). {s['leftover_val_source_tiles_discarded']}
tiles of held-out sources were discarded rather than allowed into training.

![complexity distribution](complexity_histogram.png)

![train sample](contact_pd12m_train_full.png)

![bench sample](contact_pd12m_bench.png)

---

## 4. Score sheet

```
tile,source_id,complexity,entropy,lap_var,grad_energy,blockiness,aliasing,clipped,topiq_nr,clipiqa+,niqe,qrealign
000001.png,12ab…,0.612,7.41,890.2,3.10,1.02,0.44,0.011,0.712,0.781,3.94,0.912
```

`complexity` is calibrated ICNet (higher = more detail). `niqe` is
lower-better; everything else higher-better.

---

## 5. Limits (read this)

- **2% corpus sample**: 250k of PD12M's 12.4M images. Statistical coverage,
  not completeness — expansion rounds can be appended later.
- **JPEG sources**: PD12M images are web JPEGs; the gates remove the
  degradation classes we can measure, not every possible artifact.
- **Not proven on Urban100 yet**: gains over the previous dataset generation
  must be established by training A/B, not by gate metrics.
- The corpus is "highly aesthetic" (curated), so expect polish and some
  editing in the sources; naturalness gates are the counterweight.

## 6. Provenance

Built with [LUCID 3.5.0](https://github.com/Phhofm/lucid-sisr) — calibration,
cascade gates, fail-closed semantics, per-tile score ledger.
Counts: `sets_summary.json`; per-parquet ingestion ledgers in the private
workspace of the build machine.

Source imagery: PD12M (CDLA-Permissive-2.0), built by Spawning / Source.Plus.
"""
    (OUT / "README.md").write_text(card)
    print("card written:", OUT / "README.md")


if __name__ == "__main__":
    main()
