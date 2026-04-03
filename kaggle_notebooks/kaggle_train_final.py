"""
ICDAR 2026 CircleID — Pen Classification — V2 (Multi-task Baseline + Scale-Up)
===============================================================================
Self-contained Kaggle Notebook script.

Architecture:
  1. Standard nn.Linear head (no ArcFace) + Multi-task Writer auxiliary
  2. CNN + Transformer ensemble: ConvNeXt-Tiny (224) + SwinV2-Tiny (256)
  3. 3-fold × 2 models = 6 ensemble members
  4. Label Smoothing (0.1) + Cosine-decaying Writer alpha
  5. Pseudo-Labeling disabled (establish baseline first)

Speed optimizations retained:
  - Single-GPU (no DataParallel)
  - Offline image resizing assumed (no A.Resize in transforms)
  - NUM_WORKERS=2, cudnn.benchmark=True, AMP
  - TTA on final inference only

Usage on Kaggle:
    1. Add "icdar-2026-circleid-pen-classification" as input dataset
    2. Add "sida-resized" dataset (pre-resized images)
    3. Enable GPU accelerator (P100 / T4)
    4. Run this script
"""

import os
import sys
import math
import zipfile
import contextlib
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================
class CFG:
    # Data — auto-detect Kaggle input path
    DATA_DIR = None
    IMG_DIR = None
    OUTPUT_DIR = "/kaggle/working"

    # Models for Ensemble — CNN (ConvNeXt) + Transformer (SwinV2) for diversity
    MODELS = ["convnext_tiny", "swinv2_tiny_window8_256"]
    IMAGE_SIZES = {"convnext_tiny": 224, "swinv2_tiny_window8_256": 256}
    NUM_CLASSES = 8

    # Training
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 5e-4
    WEIGHT_DECAY = 0.05
    BACKBONE_LR_FACTOR = 0.1
    LABEL_SMOOTHING = 0.1
    NUM_WORKERS = 2              # stable on Kaggle single GPU

    # TTA
    USE_TTA = True

    # Multi-task
    MULTI_TASK = True
    WRITER_LOSS_ALPHA = 0.3

    # Pseudo-Labeling
    USE_PSEUDO_LABELING = False   # disabled — caused timeout, only 12 pseudo-labels
    PSEUDO_CONFIDENCE_THRESHOLD = 0.95
    PSEUDO_FINETUNE_EPOCHS = 2
    PSEUDO_FINETUNE_LR = 1e-4

    # Cross-Validation
    N_FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # 3 folds × 2 models = 6 ensemble members

    # AMP
    USE_AMP = True
    SEED = 42


def _find_data_dir():
    """Auto-detect the dataset directory under /kaggle/input/."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for csv_path in kaggle_input.rglob("train.csv"):
            return str(csv_path.parent)
        for csv_path in kaggle_input.rglob("additional_train.csv"):
            return str(csv_path.parent)
    local = Path("icdar-2026-circleid-pen-classification")
    if local.exists():
        return str(local)
    if kaggle_input.exists():
        print("Dataset not found automatically. Here is what is in /kaggle/input/:")
        for root, _, files in os.walk("/kaggle/input"):
            level = root.replace("/kaggle/input", "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}    {f}")
    raise FileNotFoundError(
        "Cannot find train.csv or additional_train.csv. "
        "Please check the logs above to see your exact Kaggle directory structure."
    )


CFG.DATA_DIR = _find_data_dir()

# --- Load BOTH train.csv and additional_train.csv for maximum data ---
_train_path = Path(f"{CFG.DATA_DIR}/train.csv")
_additional_path = Path(f"{CFG.DATA_DIR}/additional_train.csv")

_dfs = []
if _train_path.exists():
    _dfs.append(pd.read_csv(_train_path))
    print(f"  Loaded train.csv: {len(_dfs[-1])} rows")
if _additional_path.exists():
    _dfs.append(pd.read_csv(_additional_path))
    print(f"  Loaded additional_train.csv: {len(_dfs[-1])} rows")

if not _dfs:
    raise FileNotFoundError("Cannot find train.csv or additional_train.csv!")

_combined_df = pd.concat(_dfs, ignore_index=True)

# Deduplicate by image_id (in case both CSVs contain the same images)
_combined_df = _combined_df.drop_duplicates(subset=["image_id"], keep="first")

# Handle writer_id = -1 (unknown writer in additional_train.csv)
# Assign unique pseudo-writer IDs so GroupKFold distributes them across folds
_unknown_mask = _combined_df["writer_id"].astype(str).isin(["-1", "-1.0"])
_n_unknown = _unknown_mask.sum()
if _n_unknown > 0:
    _combined_df.loc[_unknown_mask, "writer_id"] = [
        f"pseudo_{i}" for i in range(_n_unknown)
    ]
    print(f"  Assigned {_n_unknown} pseudo-writer IDs for unknown writers")

# Save combined CSV to a writable location
_combined_csv_path = "/kaggle/working/combined_train.csv" if Path("/kaggle/working").exists() else "/tmp/combined_train.csv"
_combined_df.to_csv(_combined_csv_path, index=False)
CFG.TRAIN_CSV = _combined_csv_path
print(f"  Combined training data: {len(_combined_df)} rows → {CFG.TRAIN_CSV}")

CFG.TEST_CSV = None
kaggle_input = Path("/kaggle/input")
if kaggle_input.exists():
    for csv_path in kaggle_input.rglob("test.csv"):
        CFG.TEST_CSV = str(csv_path)
        break
if CFG.TEST_CSV is None and Path(f"{CFG.DATA_DIR}/test.csv").exists():
    CFG.TEST_CSV = f"{CFG.DATA_DIR}/test.csv"


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # allow non-deterministic for speed
    torch.backends.cudnn.benchmark = True        # auto-tune convolution algorithms


# ============================================================================
# DATA UTILITIES & VALIDATION
# ============================================================================
def _auto_extract_zips():
    kaggle_input = Path("/kaggle/input")
    extract_dir = Path("/kaggle/working/extracted")
    if not kaggle_input.exists():
        return
    for zf in kaggle_input.rglob("*.zip"):
        target = extract_dir / zf.stem
        if target.exists():
            print(f"Already extracted: {zf.name} → {target}")
            continue
        print(f"📦 Extracting {zf} → {target} ...")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(target)
        print(f"   Done! ({len(list(target.rglob('*')))} files extracted)")


def find_image_base_dir(sample_rel_path: str, data_dir: str) -> Path:
    sample_name = Path(sample_rel_path).name
    base = Path(data_dir)
    if (base / sample_rel_path).exists():
        print(f"✅ Images found relative to CSV dir: {base}")
        return base
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print(f"Image '{sample_rel_path}' not found in {base}. Searching /kaggle/input...")
        for p in kaggle_input.rglob(sample_name):
            parts_num = len(Path(sample_rel_path).parts)
            found_base = p.parents[parts_num - 1]
            print(f"✅ Found image at {p}! Image base dir: {found_base}")
            return found_base
    _auto_extract_zips()
    extract_dir = Path("/kaggle/working/extracted")
    if extract_dir.exists():
        print("Searching extracted zip contents...")
        for p in extract_dir.rglob(sample_name):
            parts_num = len(Path(sample_rel_path).parts)
            found_base = p.parents[parts_num - 1]
            print(f"✅ Found image in extracted zip at {p}! Image base dir: {found_base}")
            return found_base
    print("\n" + "=" * 60)
    print("❌ CRITICAL: No image files found!")
    print("=" * 60)
    all_search = [kaggle_input, extract_dir]
    for search_root in all_search:
        if search_root.exists():
            png_files = list(search_root.rglob("*.png"))
            jpg_files = list(search_root.rglob("*.jpg"))
            print(f"{search_root}: {len(png_files)} PNG, {len(jpg_files)} JPG")
            if png_files:
                print(f"  First 5: {[str(p) for p in png_files[:5]]}")
    print("\nDirectory structure of /kaggle/input:")
    if kaggle_input.exists():
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root.replace("/kaggle/input", "").count(os.sep)
            if depth <= 4:
                indent = "  " * depth
                print(f"{indent}{os.path.basename(root)}/  ({len(files)} files)")
    print("\n💡 FIX: Add the dataset containing the 'images/' folder as notebook input.")
    print("   CSV references:", sample_rel_path)
    print("=" * 60 + "\n")
    return base


def validate_dataset(annotations: list, label: str = "dataset") -> int:
    missing_count = 0
    total = len(annotations)
    print(f"\nValidating {label} ({total} images)...")
    for ann in annotations:
        if not Path(ann["image_path"]).exists():
            missing_count += 1
    if missing_count == 0:
        print(f"✅ All {total} images found!")
    elif missing_count == total:
        print(f"❌ ALL {total} images are missing! Your image dataset is not loaded.")
        print("   Please add the image dataset as an input to your Kaggle notebook.")
        print("   Stopping to avoid training on blank placeholders.\n")
        sys.exit(1)
    else:
        pct = missing_count / total * 100
        print(f"⚠️ {missing_count}/{total} ({pct:.1f}%) images missing. Placeholders will be used.")
    return missing_count


def resize_with_pad(img: np.ndarray, target_size: int = 224, pad_color: int = 255) -> np.ndarray:
    """Aspect-ratio-preserving resize with white padding."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if img.ndim == 3:
        canvas = np.full((target_size, target_size, img.shape[2]), pad_color, dtype=np.uint8)
    else:
        canvas = np.full((target_size, target_size), pad_color, dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def get_train_transform(image_size: int = 224) -> A.Compose:
    """Training augmentations with ink-aware domain transforms.
    No A.Resize — assumes images are pre-resized offline."""
    return A.Compose([
        # --- Geometric ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05),
                 rotate=(-15, 15), border_mode=0, fill=255, p=0.5),
        # --- Ink-aware domain augmentations ---
        A.CLAHE(clip_limit=(1, 4), p=0.3),                                      # local ink contrast
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),               # ink edge emphasis
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.3),  # paper variation
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),                              # ink bleed / scan blur
        # --- Noise / dropout ---
        A.GaussNoise(std_range=(0.02, 0.05), mean_range=(0.0, 0.0), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(int(image_size * 0.04), int(image_size * 0.08)),
                        hole_width_range=(int(image_size * 0.04), int(image_size * 0.08)),
                        fill=255, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    """Validation/inference transform. No A.Resize — assumes images are pre-resized offline."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================================
# DATASET
# ============================================================================
class CircleDataset(Dataset):
    def __init__(self, annotations, transform=None, image_size=224, writer_id_map=None):
        self.annotations = annotations
        self.image_size = image_size
        self.transform = transform or get_val_transform(image_size)
        if writer_id_map is not None:
            self.writer_id_map = writer_id_map
        else:
            unique_writers = sorted(set(str(r["writer_id"]) for r in annotations))
            self.writer_id_map = {wid: idx for idx, wid in enumerate(unique_writers)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        record = self.annotations[idx]
        img_path = record["image_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (self.image_size, self.image_size):
                # Fallback: if not pre-resized offline, do it on the fly
                img = resize_with_pad(img, target_size=self.image_size)
        transformed = self.transform(image=img)
        image_tensor = transformed["image"]
        label = torch.tensor(record["pen_id"], dtype=torch.int64)
        writer_label = torch.tensor(
            self.writer_id_map.get(str(record["writer_id"]), 0), dtype=torch.int64
        )
        return {
            "image": image_tensor,
            "label": label,
            "writer_id": record["writer_id"],
            "writer_label": writer_label,
            "image_id": record.get("image_id", ""),
        }


# ============================================================================
# MODEL  (Standard Linear Heads — no ArcFace)
# ============================================================================
class PenClassifierHead(nn.Module):
    """2-layer MLP head with GELU bottleneck for non-linear ink feature combination."""

    def __init__(self, in_features: int, num_classes: int = 8, p_dropout: float = 0.3):
        super().__init__()
        hidden = in_features // 2
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(p=p_dropout * 0.5),
            nn.Linear(hidden, num_classes)
        )
        # Xavier init on both linear layers
        nn.init.xavier_uniform_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)
        nn.init.xavier_uniform_(self.head[5].weight)
        nn.init.zeros_(self.head[5].bias)

    def forward(self, x):
        return self.head(x)


class PenClassificationModel(nn.Module):
    def __init__(self, backbone, head, writer_head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head                # PenClassifierHead (standard linear)
        self.writer_head = writer_head  # PenClassifierHead (standard linear)

    def forward(self, x):
        features = self.backbone(x)
        pen_logits = self.head(features)
        if self.writer_head is not None and self.training:
            writer_logits = self.writer_head(features)
            return pen_logits, writer_logits
        return pen_logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(model_name="convnext_tiny", pretrained=True, num_classes=8,
                 p_dropout=0.3, num_writers=0, **kwargs):
    """Create model with standard linear head."""
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    feature_dim = backbone.num_features

    head = PenClassifierHead(feature_dim, num_classes, p_dropout)
    writer_head = PenClassifierHead(feature_dim, num_writers, p_dropout) if num_writers > 0 else None
    return PenClassificationModel(backbone, head, writer_head)


# ============================================================================
# LOSSES
# ============================================================================
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.3, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha  # mutable — updated per-epoch by Trainer
        self.pen_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.writer_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, pen_logits, pen_targets, writer_logits=None, writer_targets=None):
        pen_loss = self.pen_criterion(pen_logits, pen_targets)
        if writer_logits is not None and writer_targets is not None and self.alpha > 0:
            writer_loss = self.writer_criterion(writer_logits, writer_targets)
            return pen_loss + self.alpha * writer_loss
        return pen_loss

# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(preds, targets):
    preds_np = preds.detach().to(torch.float32).cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    pred_labels = np.argmax(preds_np, axis=1) if preds_np.ndim == 2 else preds_np
    
    acc = accuracy_score(targets_np, pred_labels)
    macro_f1 = f1_score(targets_np, pred_labels, average="macro", zero_division=0)
    
    cm = confusion_matrix(targets_np, pred_labels, labels=np.arange(CFG.NUM_CLASSES))
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
    
    for i, class_acc in enumerate(per_class_acc):
        metrics[f"acc_pen_{i}"] = class_acc
        
    return metrics
# ============================================================================
# OPTIMIZER (Discriminative LR)
# ============================================================================
def create_optimizer(model, lr=1e-3, weight_decay=0.05, backbone_lr_factor=0.1):
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_factor, "name": "backbone"},
        {"params": head_params, "lr": lr, "name": "heads"},
    ], weight_decay=weight_decay)


def create_scheduler(optimizer, max_lr, epochs, steps_per_epoch,
                     pct_start=0.1, backbone_lr_factor=0.1):
    num_groups = len(optimizer.param_groups)
    max_lrs = [max_lr * backbone_lr_factor, max_lr] if num_groups == 2 else max_lr
    return OneCycleLR(optimizer, max_lr=max_lrs, epochs=epochs,
                      steps_per_epoch=steps_per_epoch, pct_start=pct_start,
                      anneal_strategy="cos")


# ============================================================================
# TRAINER  (Standard — no ArcFace labels injection)
# ============================================================================
class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None,
                 use_amp=True, save_path=None, multi_task=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.save_path = save_path
        self.multi_task = multi_task
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None
        self.history = defaultdict(list)

    def _amp_context(self):
        if self.use_amp and self.device.type in ('cuda', 'mps'):
            dtype = torch.bfloat16 if self.device.type == 'mps' else torch.float16
            return torch.autocast(device_type=self.device.type, dtype=dtype)
        return contextlib.nullcontext()

    def _train_epoch(self, dl):
        self.model.train()
        epoch_loss, all_preds, all_targets = 0.0, [], []

        for batch in tqdm(dl, desc="  Train", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with self._amp_context():
                output = self.model(images)

                # Unpack tuple if multi-task model returns (pen, writer)
                if isinstance(output, tuple):
                    pen_logits, writer_logits = output
                else:
                    pen_logits = output
                    writer_logits = None

                if self.multi_task and writer_logits is not None:
                    writer_labels = batch["writer_label"].to(self.device)
                    loss = self.criterion(pen_logits, labels, writer_logits, writer_labels)
                else:
                    loss = self.criterion(pen_logits, labels)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            epoch_loss += loss.item()
            all_preds.append(pen_logits.detach())
            all_targets.append(labels.detach())

        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics["loss"] = epoch_loss / len(dl)
        return metrics

    @torch.no_grad()
    def _validate_epoch(self, dl):
        self.model.eval()
        epoch_loss, all_preds, all_targets = 0.0, [], []

        for batch in tqdm(dl, desc="  Val  ", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with self._amp_context():
                pen_logits = self.model(images)
                if self.multi_task:
                    loss = self.criterion.pen_criterion(pen_logits, labels)
                else:
                    loss = self.criterion(pen_logits, labels)

            epoch_loss += loss.item()
            all_preds.append(pen_logits.detach())
            all_targets.append(labels.detach())

        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        metrics["loss"] = epoch_loss / len(dl)
        return metrics

    def fit(self, train_dl, val_dl, epochs, initial_alpha=0.3):
        """Train loop. Returns (history_dict, best_f1)."""
        best_f1 = 0.0
        for epoch in range(1, epochs + 1):
            # --- Dynamic alpha cosine decay ---
            if self.multi_task and hasattr(self.criterion, 'alpha'):
                progress = (epoch - 1) / max(epochs - 1, 1)
                self.criterion.alpha = initial_alpha * 0.5 * (1 + math.cos(math.pi * progress))
                print(f"  ⚖️  Writer alpha: {self.criterion.alpha:.4f}")

            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'=' * 50}")

            train_m = self._train_epoch(train_dl)
            print(f"  Train | Loss: {train_m['loss']:.4f} | Acc: {train_m['accuracy']:.4f} | F1: {train_m['macro_f1']:.4f}")

            val_m = self._validate_epoch(val_dl)
            val_class_acc = ", ".join([f"pen_{i}: {val_m.get(f'acc_pen_{i}', 0.0):.2f}" for i in range(CFG.NUM_CLASSES)])
            
            print(f"  Val   | Loss: {val_m['loss']:.4f} | Acc: {val_m['accuracy']:.4f} | F1: {val_m['macro_f1']:.4f}")
            print(f"          Per-class: {val_class_acc}")
            
            lrs = " | ".join(f"{pg.get('name', 'g')}: {pg['lr']:.2e}" for pg in self.optimizer.param_groups)
            print(f"  LR: {lrs}")

            for k, v in train_m.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_m.items():
                self.history[f"val_{k}"].append(v)

            if val_m["macro_f1"] > best_f1:
                print(f"  🌟 New best F1: {val_m['macro_f1']:.4f} (prev: {best_f1:.4f})")
                best_f1 = val_m["macro_f1"]
                if self.save_path:
                    # Save model-only weights (for inference)
                    torch.save(self.model.state_dict(), self.save_path)
                    # Save full checkpoint (for resuming training)
                    full_ckpt = {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "epoch": epoch,
                        "best_f1": best_f1,
                    }
                    torch.save(full_ckpt, self.save_path.replace(".pth", "_full.pth"))

        return dict(self.history), best_f1


# ============================================================================
# DATA SPLITTING
# ============================================================================
def create_group_kfold_splits(annotations, n_splits=5):
    labels = np.array([r["pen_id"] for r in annotations])
    groups = np.array([r["writer_id"] for r in annotations])
    gkf = GroupKFold(n_splits=n_splits)
    return [(train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in gkf.split(np.zeros(len(annotations)), labels, groups)]

# --- Helper: remap annotations to use a specific image base directory ---
def remap_annotations(anns, base_dir):
    """Return new annotation list with image_path pointing to base_dir."""
    return [{**a, "image_path": str(base_dir / a["rel_path"])} for a in anns]


# ============================================================================
# ENSEMBLE INFERENCE (with optional TTA + weighted averaging)
# ============================================================================
def run_ensemble_inference(all_fold_weights, test_annotations, writer_id_map,
                           num_writers, device, use_tta=True, resolve_img_base_fn=None):
    """Run weighted-ensemble inference.

    Args:
        use_tta: If True, apply TTA. 
        resolve_img_base_fn: Function that takes target_size and returns a Path.
    """
    tta_label = "TTA" if use_tta else "no-TTA"
    print(f"  Inference mode: {tta_label}")
    all_probs = []
    all_weights = []

    for model_name, wpath, fold_f1 in all_fold_weights:
        print(f"Loading {wpath} ({model_name}, val-F1={fold_f1:.4f})...")
        img_size = CFG.IMAGE_SIZES[model_name]
        
        # Use resized base if resolver provided, else keep original
        base_dir = resolve_img_base_fn(img_size) if resolve_img_base_fn else None
        test_ann_mapped = remap_annotations(test_annotations, base_dir) if base_dir else test_annotations
        
        test_ds = CircleDataset(test_ann_mapped, get_val_transform(img_size),
                                img_size, writer_id_map)
        test_dl = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                             num_workers=CFG.NUM_WORKERS, pin_memory=True)

        model = create_model(model_name, pretrained=False,
                             num_classes=CFG.NUM_CLASSES,
                             num_writers=num_writers if CFG.MULTI_TASK else 0)
        model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        model.to(device).eval()

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"  Inference {Path(wpath).stem}", leave=False):
                images = batch["image"].to(device)

                # --- TTA: original + flips + rotations + scale (only if use_tta) ---
                tta_views = [images]
                if use_tta:
                    tta_views.append(torch.flip(images, [3]))                     # hflip
                    tta_views.append(torch.flip(images, [2]))                     # vflip
                    tta_views.append(torch.rot90(images, k=1, dims=[2, 3]))       # rot90
                    tta_views.append(torch.rot90(images, k=3, dims=[2, 3]))       # rot270
                    # --- Scale TTA: capture ink texture at multiple scales ---
                    h, w = images.shape[-2:]
                    # 1.1× zoom-in → center-crop back
                    up = F.interpolate(images, scale_factor=1.1, mode='bilinear', align_corners=False)
                    dh, dw = (up.shape[-2] - h) // 2, (up.shape[-1] - w) // 2
                    tta_views.append(up[:, :, dh:dh+h, dw:dw+w])
                    # 0.9× zoom-out → pad back with zeros
                    down = F.interpolate(images, scale_factor=0.9, mode='bilinear', align_corners=False)
                    pad_h, pad_w = (h - down.shape[-2]) // 2, (w - down.shape[-1]) // 2
                    pad_h2, pad_w2 = h - down.shape[-2] - pad_h, w - down.shape[-1] - pad_w
                    tta_views.append(F.pad(down, [pad_w, pad_w2, pad_h, pad_h2], value=0))

                batch_probs = []
                for view in tta_views:
                    if CFG.USE_AMP and device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits = model(view)
                    else:
                        logits = model(view)
                    batch_probs.append(torch.softmax(logits.float(), dim=1))

                avg_tta = torch.stack(batch_probs).mean(dim=0)
                fold_probs.append(avg_tta.cpu().numpy())

        all_probs.append(np.concatenate(fold_probs))
        all_weights.append(fold_f1)

    weights = np.array(all_weights)
    weights = weights / weights.sum()
    print(f"Ensemble weights (normalised F1): {dict(zip([n for n, _, _ in all_fold_weights], weights))}")
    avg_probs = np.average(all_probs, axis=0, weights=weights)

    image_ids = [ann["image_id"] for ann in test_annotations]
    return image_ids, avg_probs


# ============================================================================
# PSEUDO-LABELING UTILITIES
# ============================================================================
def select_pseudo_labels(test_annotations, avg_probs, threshold=0.95):
    """Select test samples with high-confidence predictions as pseudo-labels."""
    max_probs = avg_probs.max(axis=1)
    pred_classes = avg_probs.argmax(axis=1)
    mask = max_probs >= threshold

    pseudo_annotations = []
    for i in np.where(mask)[0]:
        ann = test_annotations[i].copy()
        ann["pen_id"] = int(pred_classes[i])
        ann["writer_id"] = "pseudo"
        pseudo_annotations.append(ann)

    num_selected = len(pseudo_annotations)
    total = len(test_annotations)
    print(f"\n📋 Pseudo-labeling: {num_selected}/{total} test images "
          f"({num_selected / total * 100:.1f}%) above confidence {threshold}")

    if num_selected > 0:
        pseudo_pens = [a["pen_id"] for a in pseudo_annotations]
        unique, counts = np.unique(pseudo_pens, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"   Class {cls}: {cnt} pseudo-labels")

    return pseudo_annotations, num_selected


def finetune_with_pseudo(model, model_name, annotations, pseudo_annotations,
                         val_ann, writer_id_map, num_writers, device,
                         save_path, finetune_epochs, finetune_lr, resolve_img_base_fn=None):
    """Fine-tune a trained model with original + pseudo-labeled data."""
    combined = annotations + pseudo_annotations
    print(f"\n🔄 Fine-tuning {model_name} with {len(annotations)} real + "
          f"{len(pseudo_annotations)} pseudo = {len(combined)} total samples")

    img_size = CFG.IMAGE_SIZES[model_name]

    # Use resized base if resolver provided
    base_dir = resolve_img_base_fn(img_size) if resolve_img_base_fn else None
    train_ds = CircleDataset(remap_annotations(combined, base_dir) if base_dir else combined,
                             get_train_transform(img_size), img_size, writer_id_map)
    val_ds = CircleDataset(remap_annotations(val_ann, base_dir) if base_dir else val_ann,
                           get_val_transform(img_size), img_size, writer_id_map)

    train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Use smaller LR for fine-tuning
    optimizer = AdamW(model.parameters(), lr=finetune_lr, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs * len(train_dl), eta_min=1e-7)

    criterion = MultiTaskLoss(alpha=0.0, label_smoothing=CFG.LABEL_SMOOTHING)

    trainer = Trainer(model, optimizer, criterion, device, scheduler,
                      CFG.USE_AMP, save_path, multi_task=False)
    _, best_f1 = trainer.fit(train_dl, val_dl, finetune_epochs, initial_alpha=0.0)
    return best_f1


# ============================================================================
# MAIN
# ============================================================================
def main():
    seed_everything(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Device: {device}")
    print(f"Multi-task: {CFG.MULTI_TASK}, Writer alpha: {CFG.WRITER_LOSS_ALPHA}")
    print(f"Label smoothing: {CFG.LABEL_SMOOTHING}")
    print(f"Image sizes: {CFG.IMAGE_SIZES}")
    print(f"Batch size: {CFG.BATCH_SIZE}")
    print(f"Epochs: {CFG.EPOCHS}, Folds: {CFG.TRAIN_FOLDS}")

    # --- Load Data ---
    train_df = pd.read_csv(CFG.TRAIN_CSV)

    test_df = None
    if CFG.TEST_CSV and Path(CFG.TEST_CSV).exists():
        test_df = pd.read_csv(CFG.TEST_CSV)
        print(f"Found test data: {len(test_df)} rows")
    else:
        print("WARNING: test.csv not found! Skipping inference phase.")

    # Build global pen_id map
    unique_pens = sorted(train_df["pen_id"].unique())
    pen_id_map = {int(p): i for i, p in enumerate(unique_pens)}
    inv_pen_id_map = {i: int(p) for p, i in pen_id_map.items()}
    CFG.NUM_CLASSES = len(pen_id_map)
    print(f"Pen IDs in CSV: {unique_pens} → remapped to 0..{CFG.NUM_CLASSES - 1}")

    # Build global writer_id map
    unique_writers = sorted(train_df["writer_id"].unique())
    writer_id_map = {str(w): i for i, w in enumerate(unique_writers)}
    num_writers = len(writer_id_map)
    print(f"Pen classes: {CFG.NUM_CLASSES}, Writers: {num_writers}")

    # --- Robust Image Path Resolution ---
    if CFG.IMG_DIR is not None and Path(CFG.IMG_DIR).exists():
        img_base_dir = Path(CFG.IMG_DIR)
        print(f"Using explicitly configured IMG_DIR: {img_base_dir}")
    else:
        first_img_rel = str(train_df.iloc[0]["image_path"])
        img_base_dir = find_image_base_dir(first_img_rel, CFG.DATA_DIR)
        CFG.IMG_DIR = str(img_base_dir)

    # --- Helper: resolve per-model resized directory ---
    def _resolve_img_base(target_size: int) -> Path:
        """Search for pre-resized images in working dir first, then input dir."""
        sample_rel = str(train_df.iloc[0]["image_path"])

        # 1. Check if created in current working directory
        work_dir = Path(f"/kaggle/working/resized_{target_size}")
        if (work_dir / sample_rel).exists():
            print(f"  ✨ Using pre-resized {target_size}px images from {work_dir}")
            return work_dir

        # 2. Check if uploaded directly as the 'SIDA-resized' dataset
        for possible_base in [
            Path("/kaggle/input/sida-resized"),
            Path("/kaggle/input/datasets/pound07/sida-resized")
        ]:
            uploaded_dir = possible_base / f"resized_{target_size}"
            if uploaded_dir.is_dir() and (uploaded_dir / sample_rel).exists():
                print(f"  ✨ Using pre-resized {target_size}px images from dataset: {uploaded_dir}")
                return uploaded_dir

        # If we get here, it wasn't found
        print(f"  ⚠️ Pre-resized {target_size}px not found in expected paths.")
        print(f"     -> Falling back to on-the-fly resizing from: {img_base_dir}")

        return img_base_dir

    # Build annotations (store relative path + absolute path)
    annotations = []
    for _, row in train_df.iterrows():
        annotations.append({
            "image_id": str(row["image_id"]),
            "rel_path": str(row["image_path"]),
            "image_path": str(img_base_dir / str(row["image_path"])),
            "writer_id": str(row["writer_id"]),
            "pen_id": pen_id_map[int(row["pen_id"])],
        })

    test_annotations = []
    if test_df is not None:
        for _, row in test_df.iterrows():
            test_annotations.append({
                "image_id": str(row["image_id"]),
                "rel_path": str(row["image_path"]),
                "image_path": str(img_base_dir / str(row["image_path"])),
                "writer_id": "dummy",
                "pen_id": 0,
            })

    validate_dataset(annotations, "train")

    # --- KFold Splits ---
    splits = create_group_kfold_splits(annotations, n_splits=CFG.N_FOLDS)

    # ==================================================================
    # STAGE 1: Train each model × fold + Extract OOF & Test probs
    # ==================================================================
    all_fold_weights = []

    # --- Storage for stacking features ---
    # OOF: dict mapping image_id → {model_name_p0: prob, model_name_p1: prob, ...}
    oof_records = {}  # image_id → dict of prob columns
    oof_labels = {}   # image_id → true label (remapped)
    # Test: list of (model_name, fold, probs_array) for averaging per model
    test_probs_per_model = defaultdict(list)  # model_name → list of prob arrays

    for model_name in CFG.MODELS:
        img_size = CFG.IMAGE_SIZES[model_name]
        for fold in CFG.TRAIN_FOLDS:
            print(f"\n{'#' * 60}")
            print(f"# STAGE 1 | MODEL {model_name} | FOLD {fold}")
            print(f"{'#' * 60}")

            train_idx, val_idx = splits[fold]
            train_ann = [annotations[i] for i in train_idx]
            val_ann = [annotations[i] for i in val_idx]

            save_path = f"{CFG.OUTPUT_DIR}/{model_name}_fold{fold}.pth"

            # ── Check for existing checkpoint (resume support) ──────
            existing_ckpt = None
            if not Path(save_path).exists():
                # Search /kaggle/input/ for checkpoint from a previous run
                ckpt_name = f"{model_name}_fold{fold}.pth"
                kaggle_input = Path("/kaggle/input")
                if kaggle_input.exists():
                    found = list(kaggle_input.rglob(ckpt_name))
                    if found:
                        existing_ckpt = found[0]
                        print(f"  ✅ Found checkpoint: {existing_ckpt}")
                        # Copy to working dir so downstream code works
                        import shutil
                        shutil.copy2(str(existing_ckpt), save_path)
                        print(f"     Copied to {save_path}")
            else:
                existing_ckpt = Path(save_path)
                print(f"  ✅ Checkpoint already exists: {save_path}")

            mt_writers = num_writers if CFG.MULTI_TASK else 0

            if existing_ckpt is not None:
                # ── Skip training, use existing checkpoint ──────────
                print(f"  ⏭️  Skipping training — loading existing checkpoint")
                # Try to recover best_f1 from full checkpoint
                full_ckpt_name = f"{model_name}_fold{fold}_full.pth"
                fold_best_f1 = 0.0
                full_ckpt_paths = list(Path("/kaggle/input").rglob(full_ckpt_name)) if Path("/kaggle/input").exists() else []
                if full_ckpt_paths:
                    try:
                        ckpt_data = torch.load(str(full_ckpt_paths[0]), map_location="cpu", weights_only=False)
                        fold_best_f1 = ckpt_data.get("best_f1", 0.0)
                        print(f"     Recovered best_f1={fold_best_f1:.4f} from full checkpoint")
                    except Exception:
                        pass
                if fold_best_f1 == 0.0:
                    # Quick validation pass to get actual F1
                    print(f"     Running quick validation to determine F1...")
                    _model = create_model(model_name, pretrained=False,
                                          num_classes=CFG.NUM_CLASSES, num_writers=mt_writers)
                    _model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
                    _model.to(device).eval()
                    val_ds_tmp = CircleDataset(remap_annotations(val_ann, _resolve_img_base(img_size)),
                                              get_val_transform(img_size), img_size, writer_id_map)
                    val_dl_tmp = DataLoader(val_ds_tmp, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
                    all_preds_tmp, all_labels_tmp = [], []
                    with torch.no_grad():
                        for batch in val_dl_tmp:
                            images = batch["image"].to(device)
                            if CFG.USE_AMP and device.type == 'cuda':
                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                    logits = _model(images)
                            else:
                                logits = _model(images)
                            all_preds_tmp.append(logits.argmax(dim=1).cpu())
                            all_labels_tmp.append(batch["label"])
                    fold_best_f1 = f1_score(
                        torch.cat(all_labels_tmp).numpy(),
                        torch.cat(all_preds_tmp).numpy(),
                        average="macro", zero_division=0)
                    print(f"     Validation F1={fold_best_f1:.4f}")
                    del _model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                all_fold_weights.append((model_name, save_path, fold_best_f1))
                print(f"  Model {model_name} Fold {fold} RESUMED. F1={fold_best_f1:.4f}")
            else:
                # ── Normal training ─────────────────────────────────
                train_ds = CircleDataset(remap_annotations(train_ann, _resolve_img_base(img_size)),
                                         get_train_transform(img_size), img_size, writer_id_map)
                val_ds = CircleDataset(remap_annotations(val_ann, _resolve_img_base(img_size)),
                                      get_val_transform(img_size), img_size, writer_id_map)

                train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                      num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
                val_dl = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                                    num_workers=CFG.NUM_WORKERS, pin_memory=True)

                model = create_model(model_name, pretrained=True,
                                     num_classes=CFG.NUM_CLASSES, num_writers=mt_writers)

                if CFG.MULTI_TASK:
                    criterion = MultiTaskLoss(alpha=CFG.WRITER_LOSS_ALPHA,
                                              label_smoothing=CFG.LABEL_SMOOTHING)
                else:
                    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)

                optimizer = create_optimizer(model, CFG.LR, CFG.WEIGHT_DECAY, CFG.BACKBONE_LR_FACTOR)
                scheduler = create_scheduler(optimizer, CFG.LR, CFG.EPOCHS, len(train_dl),
                                             backbone_lr_factor=CFG.BACKBONE_LR_FACTOR)

                trainer = Trainer(model, optimizer, criterion, device, scheduler,
                                  CFG.USE_AMP, save_path, CFG.MULTI_TASK)
                _, fold_best_f1 = trainer.fit(train_dl, val_dl, CFG.EPOCHS,
                                              initial_alpha=CFG.WRITER_LOSS_ALPHA)
                all_fold_weights.append((model_name, save_path, fold_best_f1))
                print(f"Model {model_name} Fold {fold} complete. Best F1={fold_best_f1:.4f}")

            # ==========================================================
            # Extract OOF probabilities (validation set for this fold)
            # ==========================================================
            print(f"  Extracting OOF probs for {model_name} fold {fold}...")
            # Reload best checkpoint
            best_model = create_model(model_name, pretrained=False,
                                      num_classes=CFG.NUM_CLASSES, num_writers=mt_writers)
            best_model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
            best_model.to(device).eval()

            # Build val dataloader (no shuffle, deterministic order)
            val_ds_oof = CircleDataset(remap_annotations(val_ann, _resolve_img_base(img_size)),
                                       get_val_transform(img_size), img_size, writer_id_map)
            val_dl_oof = DataLoader(val_ds_oof, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                                    num_workers=CFG.NUM_WORKERS, pin_memory=True)

            oof_probs_list = []
            with torch.no_grad():
                for batch in tqdm(val_dl_oof, desc=f"  OOF {model_name} f{fold}", leave=False):
                    images = batch["image"].to(device)
                    tta_views = [images]
                    if CFG.USE_TTA:
                        tta_views.append(torch.flip(images, [3]))
                        tta_views.append(torch.flip(images, [2]))
                        tta_views.append(torch.rot90(images, k=1, dims=[2, 3]))
                        tta_views.append(torch.rot90(images, k=3, dims=[2, 3]))
                        h, w = images.shape[-2:]
                        up = F.interpolate(images, scale_factor=1.1, mode='bilinear', align_corners=False)
                        dh, dw = (up.shape[-2] - h) // 2, (up.shape[-1] - w) // 2
                        tta_views.append(up[:, :, dh:dh+h, dw:dw+w])
                        down = F.interpolate(images, scale_factor=0.9, mode='bilinear', align_corners=False)
                        pad_h, pad_w = (h - down.shape[-2]) // 2, (w - down.shape[-1]) // 2
                        pad_h2, pad_w2 = h - down.shape[-2] - pad_h, w - down.shape[-1] - pad_w
                        tta_views.append(F.pad(down, [pad_w, pad_w2, pad_h, pad_h2], value=0))

                    batch_probs = []
                    for view in tta_views:
                        if CFG.USE_AMP and device.type == 'cuda':
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                logits = best_model(view)
                        else:
                            logits = best_model(view)
                        batch_probs.append(torch.softmax(logits.float(), dim=1))
                    avg_tta = torch.stack(batch_probs).mean(dim=0)
                    oof_probs_list.append(avg_tta.cpu().numpy())

            oof_probs = np.concatenate(oof_probs_list)  # (N_val, 8)

            # Store OOF probs keyed by image_id
            for i, ann in enumerate(val_ann):
                iid = ann["image_id"]
                if iid not in oof_records:
                    oof_records[iid] = {}
                    oof_labels[iid] = ann["pen_id"]
                for c in range(CFG.NUM_CLASSES):
                    oof_records[iid][f"{model_name}_p{c}"] = oof_probs[i, c]

            print(f"  ✅ OOF probs: {oof_probs.shape[0]} samples for {model_name}")

            # ==========================================================
            # Extract Test probabilities (per fold, to be averaged later)
            # ==========================================================
            if test_df is not None:
                print(f"  📊 Extracting test probs for {model_name} fold {fold}...")
                test_ds_fold = CircleDataset(
                    remap_annotations(test_annotations, _resolve_img_base(img_size)),
                    get_val_transform(img_size), img_size, writer_id_map)
                test_dl_fold = DataLoader(test_ds_fold, batch_size=CFG.BATCH_SIZE * 2,
                                          shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

                test_probs_list = []
                with torch.no_grad():
                    for batch in tqdm(test_dl_fold, desc=f"  Test {model_name} f{fold}", leave=False):
                        images = batch["image"].to(device)
                        tta_views = [images]
                        if CFG.USE_TTA:
                            tta_views.append(torch.flip(images, [3]))
                            tta_views.append(torch.flip(images, [2]))
                            tta_views.append(torch.rot90(images, k=1, dims=[2, 3]))
                            tta_views.append(torch.rot90(images, k=3, dims=[2, 3]))
                            h, w = images.shape[-2:]
                            up = F.interpolate(images, scale_factor=1.1, mode='bilinear', align_corners=False)
                            dh, dw = (up.shape[-2] - h) // 2, (up.shape[-1] - w) // 2
                            tta_views.append(up[:, :, dh:dh+h, dw:dw+w])
                            down = F.interpolate(images, scale_factor=0.9, mode='bilinear', align_corners=False)
                            pad_h, pad_w = (h - down.shape[-2]) // 2, (w - down.shape[-1]) // 2
                            pad_h2, pad_w2 = h - down.shape[-2] - pad_h, w - down.shape[-1] - pad_w
                            tta_views.append(F.pad(down, [pad_w, pad_w2, pad_h, pad_h2], value=0))

                        batch_probs = []
                        for view in tta_views:
                            if CFG.USE_AMP and device.type == 'cuda':
                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                    logits = best_model(view)
                            else:
                                logits = best_model(view)
                            batch_probs.append(torch.softmax(logits.float(), dim=1))
                        avg_tta = torch.stack(batch_probs).mean(dim=0)
                        test_probs_list.append(avg_tta.cpu().numpy())

                fold_test_probs = np.concatenate(test_probs_list)  # (N_test, 8)
                test_probs_per_model[model_name].append(fold_test_probs)
                print(f"  ✅ Test probs: {fold_test_probs.shape[0]} samples for {model_name}")

            # Free GPU memory
            del best_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==================================================================
    # AGGREGATE & EXPORT stacking features
    # ==================================================================
    print(f"\n{'#' * 60}")
    print("# EXPORTING STACKING FEATURES (OOF + Test)")
    print(f"{'#' * 60}")

    # --- OOF train features ---
    oof_rows = []
    for iid, probs_dict in oof_records.items():
        row = {"image_id": iid, "label": oof_labels[iid]}
        row.update(probs_dict)
        oof_rows.append(row)
    oof_df = pd.DataFrame(oof_rows)

    # Ensure consistent column order: image_id, label, model1_p0..p7, model2_p0..p7
    prob_cols = []
    for mn in CFG.MODELS:
        for c in range(CFG.NUM_CLASSES):
            prob_cols.append(f"{mn}_p{c}")
    oof_df = oof_df[["image_id", "label"] + prob_cols]

    oof_path = f"{CFG.OUTPUT_DIR}/oof_train_features.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\n✅ OOF train features saved: {oof_path}")
    print(f"   Shape: {oof_df.shape} ({len(oof_df)} samples × {len(prob_cols)} prob columns)")
    print(oof_df.head())

    # --- Test features ---
    if test_df is not None and len(test_probs_per_model) > 0:
        test_rows = []
        test_image_ids = [ann["image_id"] for ann in test_annotations]

        for i, iid in enumerate(test_image_ids):
            row = {"image_id": iid}
            for mn in CFG.MODELS:
                # Average probabilities across folds for this model
                model_fold_probs = test_probs_per_model[mn]  # list of (N_test, 8)
                avg_probs = np.mean([fp[i] for fp in model_fold_probs], axis=0)
                for c in range(CFG.NUM_CLASSES):
                    row[f"{mn}_p{c}"] = avg_probs[c]
            test_rows.append(row)

        test_feat_df = pd.DataFrame(test_rows)
        test_feat_df = test_feat_df[["image_id"] + prob_cols]

        test_feat_path = f"{CFG.OUTPUT_DIR}/test_features.csv"
        test_feat_df.to_csv(test_feat_path, index=False)
        print(f"\n✅ Test features saved: {test_feat_path}")
        print(f"   Shape: {test_feat_df.shape} ({len(test_feat_df)} samples × {len(prob_cols)} prob columns)")
        print(test_feat_df.head())

        # ==============================================================
        # FINAL SUBMISSION (weighted ensemble — same as before)
        # ==============================================================
        print(f"\n{'#' * 60}")
        print("# FINAL SUBMISSION")
        print(f"{'#' * 60}")

        # Use the existing ensemble inference for submission
        image_ids, avg_probs = run_ensemble_inference(
            all_fold_weights, test_annotations, writer_id_map,
            num_writers, device, use_tta=CFG.USE_TTA, resolve_img_base_fn=_resolve_img_base)

        final_preds = np.argmax(avg_probs, axis=1)
        final_pen_ids = [inv_pen_id_map[p] for p in final_preds]

        submission = pd.DataFrame({"image_id": image_ids, "pen_id": final_pen_ids})
        sub_path = f"{CFG.OUTPUT_DIR}/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"\n✅ Final submission saved to {sub_path}")
        print(submission.head(10))
        print(f"Total predictions: {len(submission)}")
    else:
        print("\nNo test dataset found. Training complete, weights and OOF features saved.")


if __name__ == "__main__":
    main()
