# CONVENTIONS.md — ICDAR 2026 CircleID: Pen Classification

> **Canonical reference** for the entire codebase. All contributors and all commits must comply with this document.

---

## 1. Tech Stack

| Layer | Technology | Version Constraint |
|---|---|---|
| Language | Python | ≥ 3.10 |
| Deep Learning | PyTorch | ≥ 2.1 (CUDA 12.x) |
| Image Processing | OpenCV (`cv2`) | ≥ 4.8 |
| Numerical Computing | NumPy | ≥ 1.24 |
| Data Augmentation | Albumentations | ≥ 1.3 |
| Experiment Tracking | Weights & Biases (`wandb`) | latest |
| Configuration | Hydra / OmegaConf | ≥ 1.3 |
| Testing | pytest + pytest-cov | ≥ 7.0 |
| Linting / Formatting | Ruff | latest |
| Type Checking | mypy (strict mode) | latest |

---

## 2. Project Structure

```
SIDA/
├── CONVENTIONS.md                    # ← this file
├── ARCHITECTURE_DECISION_RECORD.md   # ADR log (Step 3)
├── configs/
│   ├── default.yaml                  # Hydra root config
│   ├── model/
│   ├── data/
│   └── train/
├── data/
│   ├── raw/                          # Original data, DO NOT modify
│   ├── processed/                    # Preprocessed images
│   └── splits/                       # GroupKFold split indices (JSON/CSV)
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # torch.utils.data.Dataset
│   │   ├── datamodule.py             # DataLoader factory + GroupKFold
│   │   ├── transforms.py            # Albumentations pipelines
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py               # Feature extraction (e.g. ConvNeXt)
│   │   ├── classifier.py             # Classification head
│   │   └── factory.py                # Model registry / builder
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop
│   │   ├── losses.py                 # Custom loss functions
│   │   ├── metrics.py                # Top-1 Accuracy + Macro F1-Score
│   │   └── optimizer.py              # Optimizer & scheduler factory
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                   # Reproducibility utilities
│       ├── logger.py                 # Logging setup
│       └── visualization.py          # Debug / analysis plots
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_transforms.py
│   ├── test_model.py
│   └── test_training.py
├── notebooks/                        # EDA only, NO production logic
├── scripts/
│   ├── train.py                      # Entrypoint: training
│   ├── evaluate.py                   # Entrypoint: evaluation
│   ├── predict.py                    # Entrypoint: inference
│   └── make_submission.py            # Export CSV (image_id, pen_id)
├── submissions/                      # Kaggle submission .csv files
├── outputs/                          # Hydra auto-generated run logs
├── checkpoints/                      # Model weights (.pt / .ckpt)
├── requirements.txt
└── pyproject.toml
```

### Directory Rules

- `data/raw/` is **read-only**. Never overwrite original data.
- `notebooks/` is for EDA / visualization only. **Do not import logic** from notebooks into `src/`.
- Every module under `src/` must have an `__init__.py` with explicit exports.

---

## 3. Coding Standards

### 3.1 Naming Conventions

```python
# Classes: PascalCase
class CircleDataset(torch.utils.data.Dataset): ...

# Functions / methods: snake_case
def compute_ink_features(image: np.ndarray) -> np.ndarray: ...

# Constants: UPPER_SNAKE_CASE
NUM_CLASSES = 8  # ICDAR 2026 CircleID: 8 pen types
DEFAULT_IMAGE_SIZE = 224

# Private: leading underscore
def _validate_split_indices(indices: list[int]) -> bool: ...
```

### 3.2 Type Hints — Mandatory

```python
# ✅ Correct
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x)

# ❌ Forbidden
def forward(self, x):
    return self.backbone(x)
```

### 3.3 Docstrings — Google Style

```python
def load_image(path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and resize an image from disk.

    Args:
        path: Absolute path to the image file.
        size: Target (height, width) in pixels.

    Returns:
        Image as float32 ndarray with shape (H, W, C), range [0, 1].

    Raises:
        FileNotFoundError: If path does not exist.
    """
```

### 3.4 Tensor Shape Comments — Mandatory

```python
# Every tensor transformation MUST have a shape comment
x = self.conv1(x)          # (B, 64, H/2, W/2)
x = self.pool(x)           # (B, 64, H/4, W/4)
x = x.flatten(start_dim=1) # (B, 64 * H/4 * W/4)
```

---

## 4. Anti-Patterns — STRICTLY PROHIBITED

### 🚫 AP-01: For-loops over pixels / samples when vectorization is possible

```python
# ❌ FORBIDDEN
for i in range(batch_size):
    output[i] = model(input[i])

# ✅ REQUIRED — Vectorized batch operation
output = model(input)  # (B, C, H, W) → (B, num_classes)
```

### 🚫 AP-02: Data Leakage via GroupKFold

```python
# ❌ FORBIDDEN — Same writer_id appears in both train and val
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(images):
    ...

# ✅ REQUIRED — Group by writer_id
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(images, labels, groups=writer_ids):
    ...
```

### 🚫 AP-03: Hard-coded paths / magic numbers

```python
# ❌ FORBIDDEN
img = cv2.imread("/home/user/data/img001.png")
lr = 0.001

# ✅ REQUIRED — Use config (Hydra / OmegaConf)
img = cv2.imread(str(cfg.data.raw_dir / filename))
lr = cfg.train.learning_rate
```

### 🚫 AP-04: `import *` or circular imports

```python
# ❌ FORBIDDEN
from src.models import *

# ✅ REQUIRED
from src.models.backbone import ConvNeXtBackbone
```

### 🚫 AP-05: Uncontrolled global random state mutation

```python
# ❌ FORBIDDEN
import random
random.seed(42)
np.random.seed(42)

# ✅ REQUIRED — Centralized seeding
from src.utils.seed import seed_everything
seed_everything(cfg.seed)  # handles random, np, torch, cuda
```

### 🚫 AP-06: Missing `.detach().cpu()` chain

```python
# ❌ FORBIDDEN — RuntimeError on CUDA tensors
loss_value = loss.numpy()

# ✅ REQUIRED
loss_value = loss.detach().cpu().item()
```

### 🚫 AP-07: Augmentation that destroys physical ink characteristics

```python
# ❌ FORBIDDEN — Heavy color jitter destroys ink intensity information
A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

# ✅ REQUIRED — Light augmentation, preserving ink properties
A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
```

### 🚫 AP-08: Missing experiment logging

```python
# ❌ FORBIDDEN — Training without tracking
for epoch in range(epochs):
    train(model)

# ✅ REQUIRED — Every run must be logged to W&B
wandb.init(project="circleid", config=dict(cfg))
for epoch in range(epochs):
    metrics = train(model)
    wandb.log(metrics, step=epoch)
```

### 🚫 AP-09: Image interpolation that destroys ink granularity

```python
# ❌ FORBIDDEN — INTER_LINEAR blurs stroke details when downscaling
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

# ❌ FORBIDDEN — Direct resize distorts aspect ratio
img = cv2.resize(img, (224, 224))

# ✅ REQUIRED — Preserve aspect ratio with padding + correct interpolation
def resize_with_pad(
    img: np.ndarray,
    target_size: int = 224,
    pad_color: int = 255,
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    y_off, x_off = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
```

> **Interpolation Rules:**
> | Case | Interpolation |
> |---|---|
> | Downscale | `cv2.INTER_AREA` |
> | Upscale | `cv2.INTER_CUBIC` or `cv2.INTER_LANCZOS4` |

---

## 4.1 Evaluation Metrics

| Metric | Role | Notes |
|---|---|---|
| **Top-1 Accuracy** | **Primary** — official competition metric | Used for submission and leaderboard ranking |
| **Macro F1-Score** | **Secondary** — class imbalance monitor | Early detection when model neglects minority classes |

> `src/training/metrics.py` **must** implement both metrics above. Every training run must log both Accuracy and Macro F1 to W&B.

---

## 5. Git Workflow

| Rule | Detail |
|---|---|
| Branch naming | `feature/<module>`, `fix/<issue>`, `exp/<experiment-name>` |
| Commit message | `[module] verb: description` — e.g. `[data] add: GroupKFold split logic` |
| Pre-commit hooks | `ruff check .` + `mypy src/` must pass |
| Large files | Use Git LFS for `*.pt`, `*.ckpt`, `*.pth`. **DO NOT** commit model weights directly |

---

## 6. Reproducibility Checklist

- [ ] `seed_everything()` is called before all random operations
- [ ] `torch.backends.cudnn.deterministic = True`
- [ ] `torch.backends.cudnn.benchmark = False` (when absolute reproducibility is required)
- [ ] Full config is logged with experiment (W&B / Hydra output)
- [ ] Dataset split indices are saved to `data/splits/` as JSON
- [ ] `requirements.txt` / `pyproject.toml` pin exact versions

---

*Last updated: 2026-03-06*
