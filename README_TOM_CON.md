# 🍅 TOM-Con: Hybrid Self-Supervised + Semi-Supervised Learning for Tomato Disease Classification

## Overview

TOM-Con is a research project that combines **Self-Supervised Learning (SSL)** and **Semi-Supervised Learning (Semi-SL)** to perform high-accuracy tomato leaf disease classification while minimizing the amount of labeled data required.

The framework leverages:

- **SimCLR** for contrastive self-supervised representation learning
- **MixMatch** for semi-supervised training
- **ResNet-18** backbone
- Multi-seed verification experiments for robust evaluation

---

## Motivation

Labeling agricultural datasets is expensive and time-consuming, while unlabeled plant images are abundant.

This project investigates whether combining:

- Self-Supervised Learning (SimCLR)
- Semi-Supervised Learning (MixMatch)

can improve performance under limited-label settings and outperform traditional supervised learning.

---

## Proposed TOM-Con Pipeline

```text
Unlabeled Images
        │
        ▼
     SimCLR
(Contrastive Pretraining)
        │
        ▼
 Learned Feature Encoder
        │
        ▼
    MixMatch
(Semi-Supervised Training)
        │
        ▼
 Disease Classification
```

---

## Experimental Setup

### Backbone
- ResNet-18

### Compared Methods

| Method | Description |
|----------|-------------|
| B1 | Supervised Learning |
| B3 | SimCLR + Fine-Tuning |
| B4 | MixMatch |
| B5 | TOM-Con (SimCLR + MixMatch) |

### Label Availability Scenarios

- 1% labeled data
- 20% labeled data
- 40% labeled data

---

# Results

## Classification Accuracy (%)

| Label Percentage | Supervised | SimCLR | MixMatch | TOM-Con |
|------------------|------------|---------|-----------|---------|
| 1% | 75.02 | **85.41** | 85.28 | 82.97 |
| 20% | 97.24 | 99.19 | 99.31 | **99.50** |
| 40% | 98.56 | 99.37 | **99.56** | **99.56** |

---

## Key Findings

- SimCLR significantly improves performance in extremely low-label scenarios.
- MixMatch effectively leverages unlabeled samples.
- TOM-Con achieves the highest performance at 20% label availability.
- Near-perfect classification performance (**99.56% accuracy**) is achieved using only a fraction of labeled data.
- Hybrid SSL + Semi-SL training provides robust representations for plant disease recognition.

---

## Visual Results

### SimCLR Pretraining Loss

```text
RESULTS/simclr/simclr_loss_curve.png
```

### Training Curves

```text
RESULTS/20pct/B5_hybrid/training_curves.png
RESULTS/40pct/B5_hybrid/training_curves.png
```

### Confusion Matrices

```text
RESULTS/20pct/B5_hybrid/confusion_matrix.png
RESULTS/40pct/B5_hybrid/confusion_matrix.png
```

After uploading the repository, replace the code blocks above with actual markdown image links:

```md
![Training Curve](RESULTS/20pct/B5_hybrid/training_curves.png)
```

---

## Repository Structure

```text
.
├── New_Hybrid_Training_v2(SingleSeed Experiment).ipynb
├── New_Hybrid_Training_v4(MultiSeed Verification).ipynb
│
├── RESULTS/
│   ├── 1pct/
│   ├── 20pct/
│   ├── 40pct/
│   └── simclr/
│
└── README.md
```

---

## Tech Stack

- Python
- PyTorch
- ResNet-18
- SimCLR
- MixMatch
- NumPy
- Scikit-Learn
- Matplotlib

---

## Future Work

- Vision Transformers (ViT)
- DINO / BYOL Pretraining
- Active Learning
- Edge Deployment for Smart Agriculture
- Cross-Dataset Generalization

---

## Author

**Shreyansh Sandilya**

B.Tech CSE (AI & ML)

Research Interests:
- Computer Vision
- Self-Supervised Learning
- Semi-Supervised Learning
- Deep Learning for Agriculture
- Foundation Models

---

### ⭐ If you find this project useful, consider starring the repository.
