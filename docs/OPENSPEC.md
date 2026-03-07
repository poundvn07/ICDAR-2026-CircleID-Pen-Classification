# OpenSpec: ICDAR 2026 CircleID — Pen Classification Pipeline

**Author:** AI Research Team  
**Date:** 2026-03-06  
**Version:** 1.0  
**Status:** Draft — Pending Review

---

## Abstract

This document specifies the end-to-end pipeline for the ICDAR 2026 CircleID Pen Classification task. The objective is to classify which of **8 pen types** was used to draw a hand-drawn circle, given a 400 dpi scanned grayscale crop. The dataset comprises **40,250 images** collected from **51 writers**. The primary evaluation metric is **Top-1 Accuracy** on the full test set; **Macro F1-Score** is reported as a secondary metric. A critical constraint is that test images may contain writers absent from the training set, requiring the model to generalize pen-specific features independently of writer identity.

---

## 1. Problem Formulation

### 1.1 Task Definition

Given an input image $x_i \in \mathbb{R}^{H \times W \times C}$ of a hand-drawn circle, predict the pen class $\hat{y}_i \in \{0, 1, \ldots, 7\}$.

$$\hat{y}_i = \arg\max_{k} \; f_\theta(x_i)_k$$

where $f_\theta: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^{8}$ is the classification model parameterized by $\theta$.

### 1.2 Key Challenges

| Challenge | Description |
|---|---|
| **Writer–Pen Confound** | Writer-specific drawing habits (pressure, speed, curvature) are entangled with pen-specific traits (ink deposition, stroke width, texture). The model must learn pen-invariant features. |
| **Unseen Writers at Test Time** | Test set may contain writers absent from training. Naive splits leak writer identity, inflating validation accuracy. |
| **Subtle Inter-Class Differences** | Some pen types produce visually similar strokes. Discriminative features reside at the texture/granularity level, not at the shape level. |
| **Class Imbalance Risk** | Although the dataset is controlled, per-fold distributions may vary. Macro F1 monitors minority-class performance. |

---

## 2. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                        │
├──────────────┬───────────────┬──────────────┬────────────────────┤
│  Stage 1     │  Stage 2      │  Stage 3     │  Stage 4           │
│  Data        │  Augmentation │  Feature     │  Classification    │
│  Loading     │  (Ink-Safe)   │  Extraction  │  & Optimization    │
│              │               │  Backbone    │                    │
│  GroupKFold  │  Albumentations│  ConvNeXt/  │  Cross-Entropy     │
│  by writer   │  pipeline     │  EfficientNet│  + AdamW + Cosine  │
└──────┬───────┴───────┬───────┴──────┬───────┴────────┬───────────┘
       │               │              │                │
       ▼               ▼              ▼                ▼
   CircleDataset → Transforms → Backbone → Classifier Head
       │                                               │
       │              ┌────────────────────────────┐   │
       └──────────────│  Evaluation: Acc + Macro F1 │◄──┘
                      └────────────────────────────┘
```

---

## 3. Stage 1 — Data Loading & Splitting

### 3.1 Data Inventory

| Property | Value |
|---|---|
| Total images | 40,250 |
| Writers | 51 |
| Pen types | 8 |
| Scan resolution | 400 dpi |
| Image format | Tightly-cropped circle regions |

### 3.2 GroupKFold Cross-Validation

To prevent **data leakage**, validation splits are grouped by `writer_id`. This ensures no writer appears in both the training and validation sets within any fold.

$$\text{GroupKFold}(K=5): \quad \forall k, \; \mathcal{W}_{\text{train}}^{(k)} \cap \mathcal{W}_{\text{val}}^{(k)} = \emptyset$$

where $\mathcal{W}^{(k)}$ denotes the set of writer IDs in fold $k$.

**Implementation:**

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=writer_ids)):
    assert set(writer_ids[train_idx]).isdisjoint(set(writer_ids[val_idx]))
```

### 3.3 Dataset Class Contract

```
Input:  image_path (str), label (int), writer_id (int)
Output: {
    "image":     torch.Tensor of shape (C, H, W),   # float32, [0, 1]
    "label":     torch.Tensor of shape (),           # int64
    "writer_id": int
}
```

### 3.4 DataLoader Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 32–64 | Tuned per GPU memory |
| `num_workers` | 4–8 | Balance I/O and CPU |
| `pin_memory` | `True` | Accelerate CUDA transfer |
| `drop_last` | `True` (train) / `False` (val) | Stable batch norm stats |
| `shuffle` | `True` (train) / `False` (val) | Standard practice |

---

## 4. Stage 2 — Data Augmentation (Ink-Safe)

### 4.1 Design Principle

Augmentations must **preserve physical ink characteristics** — stroke texture, ink deposition density, and granularity. Geometric transforms that alter circle shape are acceptable only if they simulate realistic scanning artifacts.

### 4.2 Training Augmentation Pipeline

| Transform | Parameters | Rationale |
|---|---|---|
| `resize_with_pad` | `target_size=224` | Preserve aspect ratio; use `INTER_AREA` (downscale) or `INTER_CUBIC` (upscale) |
| `A.HorizontalFlip` | `p=0.5` | Circles are rotationally symmetric |
| `A.VerticalFlip` | `p=0.5` | Same reasoning |
| `A.Rotate` | `limit=180, border_mode=BORDER_CONSTANT` | Full rotation invariance |
| `A.ColorJitter` | `brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02` | Light variation; preserves ink intensity |
| `A.GaussNoise` | `var_limit=(5, 25)` | Simulates scanner noise |
| `A.CoarseDropout` | `max_holes=4, max_height=16, max_width=16` | Regularization; simulates occlusion |
| `A.Normalize` | `mean, std` from dataset | Channel-wise normalization |
| `ToTensorV2` | — | Convert to `(C, H, W)` torch.Tensor |

### 4.3 Validation / Test Pipeline

Only deterministic transforms: `resize_with_pad` → `A.Normalize` → `ToTensorV2`.

---

## 5. Stage 3 — Feature Extraction Backbone

### 5.1 Candidate Architectures

| Architecture | Params (M) | ImageNet Top-1 | Rationale |
|---|---|---|---|
| **ConvNeXt-Tiny** | 28.6 | 82.1% | Strong inductive bias for texture; modern ConvNet design |
| **ConvNeXt-Small** | 50.2 | 83.1% | Higher capacity if data suffices |
| **EfficientNet-B3** | 12.0 | 81.6% | Efficient; good for constrained compute |
| **SwinV2-Tiny** | 28.3 | 81.8% | Attention-based; captures global context |

### 5.2 Transfer Learning Strategy

1. **Initialize** backbone with ImageNet-1K pretrained weights.
2. **Freeze** backbone for first $E_{\text{warmup}}$ epochs (classifier head only).
3. **Unfreeze** and fine-tune entire network with discriminative learning rates:

$$\text{lr}_{\text{backbone}} = \frac{\text{lr}_{\text{head}}}{\gamma}, \quad \gamma \in [5, 10]$$

### 5.3 Feature Dimensions

```
Input:   (B, 3, 224, 224)
Backbone output: (B, D)         # D = 768 for ConvNeXt-Tiny
Classifier:      (B, D) → (B, 8)
```

---

## 6. Stage 4 — Classification & Optimization

### 6.1 Classification Head

```python
nn.Sequential(
    nn.LayerNorm(D),
    nn.Dropout(p=0.3),
    nn.Linear(D, NUM_CLASSES),  # NUM_CLASSES = 8
)
```

### 6.2 Loss Function

**Cross-Entropy Loss** with optional **label smoothing** $\epsilon = 0.1$:

$$\mathcal{L} = -\sum_{k=1}^{K} q_k \log p_k, \quad q_k = (1 - \epsilon) \cdot \mathbb{1}[k = y] + \frac{\epsilon}{K}$$

### 6.3 Optimizer & Scheduler

| Component | Choice | Parameters |
|---|---|---|
| Optimizer | AdamW | `lr=1e-4`, `weight_decay=0.05` |
| Scheduler | CosineAnnealingWarmRestarts | `T_0=10`, `T_mult=2`, `eta_min=1e-6` |
| Warmup | Linear warmup | 5 epochs, `start_factor=0.1` |
| Gradient clipping | `max_norm=1.0` | Prevent gradient explosion |

### 6.4 Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 50–100 |
| Early stopping patience | 10 epochs (monitor: val accuracy) |
| Mixed precision | `torch.amp` (fp16) |
| EMA | Exponential Moving Average, decay = 0.9999 |

---

## 7. Evaluation Protocol

### 7.1 Metrics

| Metric | Formula | Usage |
|---|---|---|
| **Top-1 Accuracy** | $\frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$ | Primary — competition leaderboard |
| **Macro F1-Score** | $\frac{1}{K}\sum_{k=1}^{K} F_{1,k}$ | Secondary — class imbalance monitor |

### 7.2 Submission Format

```csv
image_id,pen_id
test_00001,3
test_00002,7
...
```

Generated by `scripts/make_submission.py` → saved to `submissions/`.

### 7.3 Leaderboard Split

| Split | Percentage | Visibility |
|---|---|---|
| Public | 30% | During competition |
| Private | 70% | Final ranking |

---

## 8. Computational Complexity Analysis

### 8.1 Data Loading

- Image loading: $\mathcal{O}(N)$ where $N$ = dataset size
- GroupKFold split computation: $\mathcal{O}(N)$ — single pass groupby
- Augmentation: $\mathcal{O}(H \times W)$ per image (vectorized NumPy/OpenCV ops)

### 8.2 Training

- Forward pass (ConvNeXt-Tiny): $\mathcal{O}(B \times D^2 \times \frac{HW}{P^2})$ per layer
- Memory: ~4–6 GB GPU for batch_size=32 at 224×224 with mixed precision
- Sort-based operations (e.g., top-k predictions): $\mathcal{O}(B \times K \log K)$

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Writer leakage in splits | Overfitted validation scores | GroupKFold by `writer_id` (AP-02) |
| Augmentation destroys ink features | Model cannot distinguish pens | Ink-safe augmentation policy (AP-07, AP-09) |
| Overfitting on small writer pool | Poor generalization to unseen writers | Dropout, EMA, label smoothing, weight decay |
| Class imbalance per fold | Biased accuracy | Monitor Macro F1; optional class-weighted loss |
| Interpolation artifacts | Lost texture details | Strict interpolation policy (AP-09) |

---

## References

1. ICDAR 2026 Competition on Writer and Pen Identification from Hand-Drawn Circles. https://circleid.github.io
2. Kaggle — ICDAR 2026 CircleID: Pen Classification. https://www.kaggle.com/competitions/icdar-2026-circleid-pen-classification
3. Liu, Z. et al. "A ConvNet for the 2020s." CVPR 2022.
4. Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019.

---

*End of OpenSpec v1.0*
