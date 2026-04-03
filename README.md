# ICDAR 2026 CircleID - Pen Classification

Professional implementation for the "ICDAR 2026 - CircleID: Pen Classification" computer vision task. This project uses a deep learning approach to classify pen types based on hand-drawn circle images.

## Features
- **Backbone**: ConvNeXt-Tiny (via `timm`)
- **Training**: Automatic Mixed Precision (AMP) on MPS (Mac) / CUDA (Nvidia)
- **Validation**: Writer-disjoint GroupKFold cross-validation
- **Augmentations**: Ink-safe geometric and pixel-level transforms using `albumentations`
- **Architecture**: Modular and TDD-verified (85%+ test coverage)

## Project Structure
- `src/`: Core logic (Data loading, Model architecture, Training loop)
- `scripts/`: Execution scripts (`train.py`, `make_submission.py`)
- `tests/`: TDD test suite
- `weights/`: Trained model weights
- `submissions/`: Generated Kaggle submission files

## Quick Start

### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision timm albumentations pandas scikit-learn opencv-python tqdm
```

### 2. Training
Place the dataset in `icdar-2026-circleid-pen-classification/` and run:
```bash
python scripts/train.py --epochs 10 --batch_size 32
```

### 3. Inference
Generate Kaggle submission:
```bash
python scripts/make_submission.py
```

## Results
- **Best Validation Macro F1 ~ 0.91917
