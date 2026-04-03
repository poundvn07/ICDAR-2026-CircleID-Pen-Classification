"""
ICDAR 2026 CircleID — Pen Classification — Kaggle Training Notebook
====================================================================
Self-contained script for Kaggle GPU training.
All modules are inlined — no external `src/` imports needed.

Features:
- ConvNeXt-Tiny backbone via timm
- Multi-task Learning (auxiliary writer head)
- Discriminative Learning Rates (backbone LR << head LR)
- Automatic Mixed Precision (AMP)
- Best model auto-saving on validation Macro F1

Usage on Kaggle:
    1. Add "icdar-2026-circleid-pen-classification" as input dataset
    2. Enable GPU accelerator
    3. Run this script
"""

import os
import sys
import math
import zipfile
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================
class CFG:
    # Data — auto-detect Kaggle input path
    DATA_DIR = None  # Set automatically below
    IMG_DIR = None   # Auto-detected if None, or set explicitly (e.g. "/kaggle/input/datasets/...")
    OUTPUT_DIR = "/kaggle/working"
    
    # Models for Ensemble (each at its optimal pretrained resolution)
    MODELS = ["convnext_tiny", "swinv2_tiny_window8_256"]
    IMAGE_SIZES = {"convnext_tiny": 384, "swinv2_tiny_window8_256": 256}
    NUM_CLASSES = 8
    
    # Training
    EPOCHS = 15
    BATCH_SIZE = 32  # reduced for 384×384 VRAM headroom
    LR = 1e-3
    WEIGHT_DECAY = 0.05
    BACKBONE_LR_FACTOR = 0.1  # Discriminative LR
    LABEL_SMOOTHING = 0.1
    NUM_WORKERS = 2
    
    # TTA
    USE_TTA = True  # Test-Time Augmentation (hflip + vflip)
    
    # Multi-task
    MULTI_TASK = True
    WRITER_LOSS_ALPHA = 0.3
    
    # Cross-Validation
    N_FOLDS = 5
    TRAIN_FOLDS = [0, 1]  # 2 folds × 2 architectures = 4 ensemble members
    
    # AMP
    USE_AMP = True
    SEED = 42


def _find_data_dir():
    """Auto-detect the dataset directory under /kaggle/input/."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Recursively search for train.csv OR additional_train.csv
        for csv_path in kaggle_input.rglob("train.csv"):
            return str(csv_path.parent)
        for csv_path in kaggle_input.rglob("additional_train.csv"):
            return str(csv_path.parent)
            
    # Local fallback
    local = Path("icdar-2026-circleid-pen-classification")
    if local.exists():
        return str(local)
        
    # Provide helpful debug info if nothing is found
    if kaggle_input.exists():
        print("Dataset not found automatically. Here is what is in /kaggle/input/:")
        for root, _, files in os.walk("/kaggle/input"):
            level = root.replace("/kaggle/input", "").count(os.sep)
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}    {f}")
                
    raise FileNotFoundError("Cannot find train.csv or additional_train.csv. Please check the logs above to see your exact Kaggle directory structure.")


CFG.DATA_DIR = _find_data_dir()

# Automatically use additional_train.csv if it exists, otherwise train.csv
if Path(f"{CFG.DATA_DIR}/additional_train.csv").exists():
    CFG.TRAIN_CSV = f"{CFG.DATA_DIR}/additional_train.csv"
else:
    CFG.TRAIN_CSV = f"{CFG.DATA_DIR}/train.csv"

# Find test.csv independently (in case it's in a different dataset folder)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# DATA UTILITIES & VALIDATION
# ============================================================================
def _auto_extract_zips():
    """Find and extract any .zip files under /kaggle/input to /kaggle/working/."""
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
    """Find the actual base directory where images live on Kaggle.
    
    Searches: CSV dir → /kaggle/input/ → auto-extracted zips → diagnostics.
    """
    sample_name = Path(sample_rel_path).name  # e.g. '00001.png'
    base = Path(data_dir)
    
    # 1. Check if the image exists relative to the CSV directory
    if (base / sample_rel_path).exists():
        print(f"✅ Images found relative to CSV dir: {base}")
        return base
    
    # 2. Search all of /kaggle/input recursively (covers auto-extracted zips)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print(f"Image '{sample_rel_path}' not found in {base}. Searching /kaggle/input...")
        for p in kaggle_input.rglob(sample_name):
            parts_num = len(Path(sample_rel_path).parts)
            found_base = p.parents[parts_num - 1]
            print(f"✅ Found image at {p}! Image base dir: {found_base}")
            return found_base
    
    # 3. Maybe the images are inside a .zip — extract and retry
    _auto_extract_zips()
    extract_dir = Path("/kaggle/working/extracted")
    if extract_dir.exists():
        print("Searching extracted zip contents...")
        for p in extract_dir.rglob(sample_name):
            parts_num = len(Path(sample_rel_path).parts)
            found_base = p.parents[parts_num - 1]
            print(f"✅ Found image in extracted zip at {p}! Image base dir: {found_base}")
            return found_base
    
    # 4. Nothing found — print detailed diagnostics
    print("\n" + "="*60)
    print("❌ CRITICAL: No image files found!")
    print("="*60)
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
    print("="*60 + "\n")
    return base


def validate_dataset(annotations: list, label: str = "dataset") -> int:
    """Scan annotations for missing images before training begins."""
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
        import sys
        sys.exit(1)
    else:
        pct = missing_count / total * 100
        print(f"⚠️ {missing_count}/{total} ({pct:.1f}%) images missing. Placeholders will be used.")
    return missing_count


def resize_with_pad(img: np.ndarray, target_size: int = 224, pad_color: int = 255) -> np.ndarray:
    """Aspect-ratio-preserving resize with padding (AP-09)."""
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
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05),
                 rotate=(-15, 15), border_mode=0, fill=255, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.05), mean_range=(0.0, 0.0), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(int(image_size * 0.04), int(image_size * 0.08)),
                        hole_width_range=(int(image_size * 0.04), int(image_size * 0.08)),
                        fill=255, p=0.2),
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
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
        
        # Robust loading with placeholder fallback
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Return a blank white placeholder image instead of crashing
            img_h, img_w = self.image_size, self.image_size
            img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
# MODEL
# ============================================================================
class PenClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 8, p_dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features, num_classes)
        )
        nn.init.xavier_uniform_(self.head[2].weight)
        if self.head[2].bias is not None:
            nn.init.zeros_(self.head[2].bias)

    def forward(self, x):
        return self.head(x)


class PenClassificationModel(nn.Module):
    def __init__(self, backbone, head, writer_head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.writer_head = writer_head

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
                 p_dropout=0.3, num_writers=0):
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
    return {
        "accuracy": accuracy_score(targets_np, pred_labels),
        "macro_f1": f1_score(targets_np, pred_labels, average="macro", zero_division=0),
    }


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
# TRAINER
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
        import contextlib
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
                if self.multi_task and isinstance(output, tuple):
                    pen_logits, writer_logits = output
                    writer_labels = batch["writer_label"].to(self.device)
                    loss = self.criterion(pen_logits, labels, writer_logits, writer_labels)
                else:
                    pen_logits = output
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
            # --- Dynamic alpha cosine decay (0.3 → 0.0 over epochs) ---
            if self.multi_task and hasattr(self.criterion, 'alpha'):
                progress = (epoch - 1) / max(epochs - 1, 1)  # 0→1
                self.criterion.alpha = initial_alpha * 0.5 * (1 + math.cos(math.pi * progress))
                print(f"  ⚖️  Writer alpha: {self.criterion.alpha:.4f}")

            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*50}")

            train_m = self._train_epoch(train_dl)
            print(f"  Train | Loss: {train_m['loss']:.4f} | Acc: {train_m['accuracy']:.4f} | F1: {train_m['macro_f1']:.4f}")

            val_m = self._validate_epoch(val_dl)
            print(f"  Val   | Loss: {val_m['loss']:.4f} | Acc: {val_m['accuracy']:.4f} | F1: {val_m['macro_f1']:.4f}")

            lrs = " | ".join(f"{pg.get('name','g')}: {pg['lr']:.2e}" for pg in self.optimizer.param_groups)
            print(f"  LR: {lrs}")

            for k, v in train_m.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_m.items():
                self.history[f"val_{k}"].append(v)

            if val_m["macro_f1"] > best_f1:
                print(f"  🌟 New best F1: {val_m['macro_f1']:.4f} (prev: {best_f1:.4f})")
                best_f1 = val_m["macro_f1"]
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)

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


# ============================================================================
# INFERENCE
# ============================================================================
def run_inference(model, test_dl, device, use_amp=True):
    model.eval()
    preds, ids = [], []
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Predicting"):
            images = batch["image"].to(device)
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            ids.extend(batch["image_id"])
    return ids, np.concatenate(preds)


# ============================================================================
# MAIN
# ============================================================================
def main():
    seed_everything(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Data ---
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    
    test_df = None
    if CFG.TEST_CSV and Path(CFG.TEST_CSV).exists():
        test_df = pd.read_csv(CFG.TEST_CSV)
        print(f"Found test data: {len(test_df)} rows")
    else:
        print("WARNING: test.csv not found! Skipping inference phase.")

    # Build global pen_id map (remap to contiguous 0..N-1)
    unique_pens = sorted(train_df["pen_id"].unique())
    pen_id_map = {int(p): i for i, p in enumerate(unique_pens)}
    inv_pen_id_map = {i: int(p) for p, i in pen_id_map.items()}  # for submission
    CFG.NUM_CLASSES = len(pen_id_map)
    print(f"Pen IDs in CSV: {unique_pens} → remapped to 0..{CFG.NUM_CLASSES-1}")

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

    # Build annotations (with remapped pen_id)
    annotations = []
    for _, row in train_df.iterrows():
        annotations.append({
            "image_id": str(row["image_id"]),
            "image_path": str(img_base_dir / str(row["image_path"])),
            "writer_id": str(row["writer_id"]),
            "pen_id": pen_id_map[int(row["pen_id"])],  # remapped!
        })

    test_annotations = []
    if test_df is not None:
        for _, row in test_df.iterrows():
            test_annotations.append({
                "image_id": str(row["image_id"]),
                "image_path": str(img_base_dir / str(row["image_path"])),
                "writer_id": "dummy",
                "pen_id": 0,  # dummy label
            })

    # Validate dataset before training (exits if ALL images missing)
    validate_dataset(annotations, "train")

    # --- KFold Splits ---
    splits = create_group_kfold_splits(annotations, n_splits=CFG.N_FOLDS)

    # --- Train each fold ---
    all_fold_weights = []
    
    for model_name in CFG.MODELS:
        img_size = CFG.IMAGE_SIZES[model_name]
        for fold in CFG.TRAIN_FOLDS:
            print(f"\n{'#'*60}")
            print(f"# MODEL {model_name} | FOLD {fold}")
            print(f"{'#'*60}")
    
            train_idx, val_idx = splits[fold]
            train_ann = [annotations[i] for i in train_idx]
            val_ann = [annotations[i] for i in val_idx]
    
            train_ds = CircleDataset(train_ann, get_train_transform(img_size),
                                     img_size, writer_id_map)
            val_ds = CircleDataset(val_ann, get_val_transform(img_size),
                                   img_size, writer_id_map)
    
            train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                  num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
            val_dl = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                                num_workers=CFG.NUM_WORKERS, pin_memory=True)
    
            # Model
            mt_writers = num_writers if CFG.MULTI_TASK else 0
            model = create_model(model_name, pretrained=True,
                                 num_classes=CFG.NUM_CLASSES, num_writers=mt_writers)
    
            # Loss
            if CFG.MULTI_TASK:
                criterion = MultiTaskLoss(alpha=CFG.WRITER_LOSS_ALPHA,
                                          label_smoothing=CFG.LABEL_SMOOTHING)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    
            # Optimizer
            optimizer = create_optimizer(model, CFG.LR, CFG.WEIGHT_DECAY, CFG.BACKBONE_LR_FACTOR)
            scheduler = create_scheduler(optimizer, CFG.LR, CFG.EPOCHS, len(train_dl),
                                         backbone_lr_factor=CFG.BACKBONE_LR_FACTOR)
    
            # Train
            save_path = f"{CFG.OUTPUT_DIR}/{model_name}_fold{fold}.pth"
            trainer = Trainer(model, optimizer, criterion, device, scheduler,
                              CFG.USE_AMP, save_path, CFG.MULTI_TASK)
            _, fold_best_f1 = trainer.fit(train_dl, val_dl, CFG.EPOCHS,
                                          initial_alpha=CFG.WRITER_LOSS_ALPHA)
            all_fold_weights.append((model_name, save_path, fold_best_f1))
            print(f"Model {model_name} Fold {fold} complete.  Best F1={fold_best_f1:.4f}  Weights: {save_path}")

    # --- Inference & Submission ---
    if test_df is not None:
        print(f"\n{'#'*60}")
        print("# GENERATING SUBMISSION")
        print(f"{'#'*60}")
    
        # Ensemble: weighted-average softmax probabilities across all trained models
        all_probs = []
        all_weights = []
        for model_name, wpath, fold_f1 in all_fold_weights:
            print(f"Loading {wpath} ({model_name}, val-F1={fold_f1:.4f})...")
            
            img_size = CFG.IMAGE_SIZES[model_name]
            test_ds = CircleDataset(test_annotations, get_val_transform(img_size),
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

                    # --- TTA: original + hflip + vflip ---
                    tta_views = [images]
                    if CFG.USE_TTA:
                        tta_views.append(torch.flip(images, [3]))  # horizontal flip
                        tta_views.append(torch.flip(images, [2]))  # vertical flip

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
    
        # Weighted-average probabilities (weight = validation best F1)
        weights = np.array(all_weights)
        weights = weights / weights.sum()  # normalise to sum=1
        print(f"Ensemble weights (normalised F1): {dict(zip([n for n,_,_ in all_fold_weights], weights))}")
        avg_probs = np.average(all_probs, axis=0, weights=weights)
        final_preds = np.argmax(avg_probs, axis=1)
        
        # Remap predictions back to original pen_id values
        final_pen_ids = [inv_pen_id_map[p] for p in final_preds]
    
        # Collect image_ids
        image_ids = [ann["image_id"] for ann in test_annotations]
    
        submission = pd.DataFrame({"image_id": image_ids, "pen_id": final_pen_ids})
        sub_path = f"{CFG.OUTPUT_DIR}/submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"\nSubmission saved to {sub_path}")
        print(submission.head(10))
        print(f"Total predictions: {len(submission)}")
    else:
        print("\nNo test dataset found. Training complete, weights saved.")



if __name__ == "__main__":
    main()
