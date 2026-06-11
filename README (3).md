# TOM-CSSL: Tomato Contrastive Semi-Supervised Learning

**A rigorous, multi-seed benchmark of self-supervised, semi-supervised, and hybrid learning for tomato leaf disease classification under extreme label scarcity.**

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/backbone-ResNet--18-success.svg" alt="ResNet-18">
  <img src="https://img.shields.io/badge/dataset-PlantVillage%20(Tomato)-green.svg" alt="PlantVillage Tomato">
  <img src="https://img.shields.io/badge/seeds-3%20(42%2C123%2C456)-orange.svg" alt="3 seeds">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License: MIT">
</p>

---

## TL;DR — The headline finding

> **Combining contrastive pretraining (SimCLR) with semi-supervised fine-tuning (MixMatch) — the "TOM-CSSL" hybrid — does *not* outperform MixMatch alone on this task.**
> Across 3 random seeds, **MixMatch is the strongest method at every label budget**, and the hybrid actually **underperforms both MixMatch and pure SimCLR fine-tuning in the extreme 1%-label regime**, where it also overfits the most.

This repository documents that result honestly, with controlled comparisons, multi-seed confidence estimates, a dataset-integrity audit, and fully cached splits so anyone can reproduce it. The contribution here is not a new state-of-the-art number — it is a **clean, reproducible benchmark and a clear negative result about naive hybridization** for low-label plant disease classification.

---

## Table of contents

- [Motivation](#motivation)
- [Methods compared](#methods-compared)
- [Dataset](#dataset)
- [Results](#results)
  - [Main results (multi-seed)](#main-results-multi-seed)
  - [The honest finding](#the-honest-finding)
  - [Overfitting diagnostic](#overfitting-diagnostic)
  - [Hardest classes](#hardest-classes)
- [Why these numbers can be trusted](#why-these-numbers-can-be-trusted)
- [Getting started](#getting-started)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Hyperparameters](#hyperparameters)
- [Limitations and honest notes](#limitations-and-honest-notes)
- [Future work](#future-work)
- [License and acknowledgements](#license-and-acknowledgements)

---

## Motivation

Labeling plant disease images at scale requires agronomy expertise and is expensive. Self-supervised learning (SSL) and semi-supervised learning both promise strong classifiers from few labels by exploiting large pools of *unlabeled* leaf images. A natural and increasingly popular idea is to **stack them**: contrastively pretrain a backbone, then fine-tune it semi-supervised.

**TOM-CSSL** tests exactly that hypothesis for tomato leaf disease. We benchmark four training configurations on identical data splits, at three label budgets (1%, 20%, 40%), and ask a simple question:

> *Does hybridizing contrastive pretraining with semi-supervised fine-tuning beat either ingredient on its own?*

The answer, on this dataset and under fair comparison, is **no** — and the repository is built to make that conclusion verifiable rather than asserted.

---

## Methods compared

All four configurations share the same **ResNet-18** backbone, the same input resolution (224×224), the same 30-epoch fine-tuning budget, and the same evaluation protocol (best-validation-accuracy checkpoint). Only the training recipe differs.

| ID | Method | Init | Uses unlabeled data? | One-line description |
|----|--------|------|----------------------|----------------------|
| **B1** | Supervised baseline | ImageNet | No | Fine-tune ResNet-18 on labeled images only. |
| **B3** | Self-supervised | ImageNet → SimCLR | Yes (pretraining) | SimCLR contrastive pretraining on all images, then full fine-tune on labels. |
| **B4** | Semi-supervised (MixMatch) | ImageNet | Yes (training) | MixMatch: pseudo-labels unlabeled images via sharpened, averaged augmentations + MixUp. |
| **B5** | **Hybrid — TOM-CSSL (proposed)** | ImageNet → SimCLR | Yes (both) | SimCLR pretraining **followed by** MixMatch fine-tuning. |

**SimCLR pretraining stage.** ResNet-18 encoder (ImageNet-initialized) with a 2-layer projection head (`512 → 256 → 128`), trained with the NT-Xent contrastive loss (temperature `0.5`) for 10 epochs. The pretraining is run **once** and cached, so every downstream method that needs it starts from an identical checkpoint.

**MixMatch.** `K = 2` augmentations per unlabeled image, sharpening temperature `0.5`, MixUp `α = 0.75`, and unlabeled loss weight `λ_u = 1.0`.

---

## Dataset

**PlantVillage — tomato subset**, 10 classes, audited at **16,011 images** after an integrity pass.

| Property | Value |
|----------|-------|
| Classes | 10 (9 diseases + healthy) |
| Total images (audited) | 16,011 |
| Class imbalance (max : min) | **8.6 : 1** |
| Largest class | `Tomato_YellowLeaf_Curl_Virus` (3,208) |
| Smallest class | `Tomato_mosaic_virus` (373) |
| Corrupt files found | 0 |
| Cross-class duplicate images | 0 |

The ten classes are: `Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`, `Septoria_leaf_spot`, `Spider_mites (Two-spotted)`, `Target_Spot`, `YellowLeaf_Curl_Virus`, `Tomato_mosaic_virus`, and `healthy`.

**Splits.** A stratified, per-class split carves each class into *labeled / unlabeled / validation* pools, with a hard floor of **≥10 images per class** in both the labeled and validation sets. Validation is fixed at 10%; the labeled budget is the experimental variable (1% / 20% / 40%); everything else becomes the unlabeled pool. Every split is **serialized to JSON** (`split_<pct>pct_seed<seed>.json`) so results are byte-for-byte reproducible.

> **Set the dataset path** via `TOMATO_PATH` in *Section 2 — Hyperparameters* of each notebook before running.

---

## Results

> **Reading guide.** 1% and 20% budgets are reported as **mean ± std over 3 seeds**. The 40% budget is reported from the **single-seed** run; its multi-seed sweep was not completed. **Bold** marks the best method in each row.

### Main results (multi-seed)

**Macro-F1** (primary metric — robust to the 8.6:1 class imbalance):

| Label budget | B1 Supervised | B3 SimCLR FT | B4 **MixMatch** | B5 Hybrid (TOM-CSSL) |
|:---|:---:|:---:|:---:|:---:|
| **1%**  | 0.6755 ± 0.018 | 0.8156 ± 0.013 | **0.8226 ± 0.011** | 0.7707 ± 0.010 |
| **20%** | 0.9821 ± 0.006 | 0.9895 ± 0.002 | **0.9925 ± 0.001** | 0.9918 ± 0.002 |
| **40%** † | 0.9837 | 0.9933 | 0.9951 | **0.9955** |

**Accuracy:**

| Label budget | B1 Supervised | B3 SimCLR FT | B4 **MixMatch** | B5 Hybrid (TOM-CSSL) |
|:---|:---:|:---:|:---:|:---:|
| **1%**  | 0.7278 ± 0.008 | 0.8589 ± 0.010 | **0.8629 ± 0.010** | 0.8324 ± 0.009 |
| **20%** | 0.9835 ± 0.005 | 0.9910 ± 0.002 | **0.9931 ± 0.001** | 0.9927 ± 0.002 |
| **40%** † | 0.9856 | 0.9937 | 0.9956 | 0.9956 |

<sub>† single-seed (seed 42); not averaged.</sub>

### The honest finding

Three things stand out, and the repository is structured to let you confirm each:

1. **MixMatch (B4) wins or ties everywhere.** It is the best method at 1% and 20% by macro-F1, and is statistically indistinguishable from the hybrid at 40% (where all methods saturate above 0.99).
2. **The hybrid (TOM-CSSL) does not earn its extra complexity.** Adding SimCLR pretraining *on top of* MixMatch never improves over MixMatch alone. At 1% labels it is **5.2 macro-F1 points worse** than MixMatch (0.771 vs 0.823) — and worse than even pure SimCLR fine-tuning (0.816).
3. **The gains at low labels come from using unlabeled data at all, not from hybridizing.** Going from B1 (supervised) to *any* unlabeled-data method buys ~+13 macro-F1 points at 1% labels; the choice *among* those methods is a much smaller, second-order effect.

This is the central, reproducible result of the project.

### Overfitting diagnostic

A train-vs-validation gap analysis at 1% labels explains *why* the hybrid lags. The hybrid not only overfits — its train/val gap **widens fastest over training** (the worst gap trend of any method), consistent with SimCLR pretraining and MixMatch fine-tuning pulling the representation in ways that don't compose cleanly when labels are scarce.

| Method | Final train acc | Final val acc | Gap | Verdict |
|:---|:---:|:---:|:---:|:---|
| B1 Supervised | 77.6% | 60.1% | 17.5 | Overfitting (large gap) |
| B3 SimCLR FT | 100.0% | 85.9% | 14.1 | Mild gap, acceptable |
| **B4 MixMatch** | 100.0% | 86.3% | **13.7** | Mild gap, acceptable |
| B5 Hybrid (TOM-CSSL) | 98.8% | 83.2% | 15.5 | Overfitting (large gap) |

### Hardest classes

Even the best method (MixMatch @ 1% labels) struggles on specific diseases. Per-class macro-F1 (mean over 3 seeds):

- **Hardest:** `Early_blight` — **0.475 ± 0.023** (frequently confused with `Late_blight`/`Septoria`).
- **Easiest:** `healthy` (0.968) and `YellowLeaf_Curl_Virus` (0.957).

`Early_blight` is the clearest target for future improvement and a more useful lever than method hybridization.

---

## Why these numbers can be trusted

This project was explicitly built to avoid the most common pitfalls that make "our hybrid wins" claims unreliable:

- **Controlled comparisons.** All methods use the *same* fixed split per seed, the *same* ImageNet-normalized inputs, the *same* full fine-tuning, and the *same* best-epoch checkpoint selection. No method gets a different random split or a longer training budget by accident.
- **Multi-seed verification.** The 1% and 20% conclusions are averaged over seeds **42, 123, 456**, with per-method standard deviations reported.
- **Cached, serialized splits.** Every train/unlabeled/val split is written to JSON, so the exact data partition is recoverable.
- **Dataset integrity audit.** A one-time hash-based pass checks for corrupt images, intra-class duplicates, and — critically — **cross-class duplicate leakage** (found: zero).
- **Frozen SimCLR checkpoint.** Contrastive pretraining is run once (`SIMCLR_SEED = 42`) and reused, so SSL randomness is not a hidden confounder across methods.

> Earlier exploratory runs that appeared to show "hybrid wins" were traced back to *uncontrolled* comparisons (different splits, partial fine-tuning, missing normalization). Those artifacts are exactly what the controlled pipeline here was designed to eliminate.

---

## Getting started

### Requirements

- Python 3.10+
- PyTorch 2.x + TorchVision
- `numpy`, `tqdm`, `pillow`, `matplotlib`, `scikit-learn`

```bash
git clone https://github.com/<your-username>/tom-cssl.git
cd tom-cssl
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision numpy tqdm pillow matplotlib scikit-learn
```

> Notebooks were developed on Apple Silicon (MPS) but auto-detect `cuda` / `mps` / `cpu`. `NUM_WORKERS = 0` is set for cross-platform safety; raise it on Linux/CUDA for speed.

### Dataset setup

1. Download the **PlantVillage tomato subset** and arrange it as one folder per class (`ImageFolder` layout):

   ```
   DATASET/
   ├── Tomato_Bacterial_spot/
   ├── Tomato_Early_blight/
   ├── ...
   └── Tomato_healthy/
   ```

2. Point `TOMATO_PATH` (in *Section 2*) at that folder.

---

## Reproducing the experiments

Open **`New_Hybrid_Training_v4_MultiSeed_Verification_.ipynb`** (the canonical pipeline) and run top to bottom:

1. **Sections 0–7** — environment, paths, hyperparameters, dataset integrity check, transforms, the stratified split, and metric utilities.
2. **Section 9** — SimCLR pretraining (run **once**; the checkpoint is cached and reused by B3 and B5).
3. **Section 10** — set the label budget (`LABEL_PCT`) and build the labeled / unlabeled / validation loaders.
4. **Sections 11–14** — train **B1 → B3 → B4 → B5**, each across seeds `[42, 123, 456]`.
5. **Section 15** — overfitting (train-vs-val gap) diagnostic across seeds.
6. **Sections 16–17** — per-budget summary tables with confidence intervals and per-class F1 heatmaps.

Repeat steps 3–6 for each budget (`1pct`, `20pct`, `40pct`).

For a faster single-seed pass (useful for sanity checks), use **v2** (clean single-seed) or **v3** (single-seed + the integrity and overfitting diagnostics).

---

## Hyperparameters

| Group | Setting | Value |
|---|---|---|
| **General** | Image size | 224 × 224 |
| | Batch size | 128 |
| | Fine-tuning epochs | 30 |
| | Backbone | ResNet-18 (ImageNet-initialized) |
| | Checkpoint selection | Best validation accuracy |
| | Seeds (method runs) | 42, 123, 456 |
| **SimCLR** | Epochs | 10 |
| | Learning rate | 1e-4 |
| | NT-Xent temperature | 0.5 |
| | Projection head | 512 → 256 → 128 |
| | Pretraining seed | 42 (fixed) |
| **MixMatch** | Augmentations `K` | 2 |
| | Sharpening temperature | 0.5 |
| | MixUp `α` | 0.75 |
| | Unlabeled weight `λ_u` | 1.0 |
| **Per-method LR** | B1 / B3 / B4 / B5 | 1e-3 / 1e-4 / 1e-4 / 1e-4 |
| **Splits** | Validation fraction | 10% |
| | Min per class (labeled & val) | 10 |

---

## Limitations and honest notes

- **Short SSL schedule.** SimCLR was pretrained for only 10 epochs (loss 4.07 → 3.72), warm-started from ImageNet. A longer contrastive schedule and/or larger batch might change the hybrid's standing — this is the most important untested variable.
- **40% is single-seed.** Treat the 40% row as indicative, not confidence-bounded.
- **Saturation at higher budgets.** At 20%+ labels all methods exceed 0.98 macro-F1, so differences there are small and partly within noise; the interesting regime is **1% labels**.
- **One dataset, one backbone.** Findings are specific to PlantVillage tomato + ResNet-18 and should not be over-generalized to other crops, domains, or architectures without re-running the controlled pipeline.
- **Negative result, stated as such.** The hybrid is reported as *not* beating MixMatch — deliberately, because the controlled evidence says so.

---

## Future work

- Longer / larger-batch SimCLR pretraining before declaring the hybrid ineffective.
- Targeted handling of `Early_blight` (hardest class), e.g. class-balanced sampling or focal loss.
- FixMatch-style consistency regularization as an alternative semi-supervised backbone.
- Multi-seed completion of the 40% budget for full confidence intervals.
- Extending the controlled protocol to other PlantVillage crops.

---

## License and acknowledgements

Released under the **MIT License** (see `LICENSE`).

- **Dataset:** PlantVillage (tomato subset).
- **Methods:** SimCLR (Chen et al., 2020) and MixMatch (Berthelot et al., 2019).
- **Frameworks:** PyTorch and TorchVision.

> *Maintained as a reproducible research artifact. Issues and pull requests that probe, extend, or attempt to overturn the central finding are especially welcome — that is exactly what a benchmark is for.*
