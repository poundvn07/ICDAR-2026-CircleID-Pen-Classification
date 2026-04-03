"""Main Training script for ICDAR 2026 CircleID Pen Classification.

Supports:
- Multi-task Learning (auxiliary writer head)
- Discriminative Learning Rates (backbone LR < head LR)
- Multiple backbone architectures (ConvNeXt, SwinV2, EfficientNet)
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports from 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import torch
import pandas as pd

from src.data.datamodule import CircleDataModule
from src.models.factory import create_model
from src.training.losses import get_loss_fn, MultiTaskLoss
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pen Classification Model")
    parser.add_argument("--csv_file", type=str, default="icdar-2026-circleid-pen-classification/train.csv")
    parser.add_argument("--image_dir", type=str, default="icdar-2026-circleid-pen-classification")
    parser.add_argument("--fold", type=int, default=0, help="Hold out fold for validation (0-4)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="convnext_tiny")
    parser.add_argument("--output_dir", type=str, default="./weights")
    # Advanced techniques
    parser.add_argument("--multi_task", action="store_true", help="Enable multi-task writer head")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1,
                        help="Backbone LR = lr * this factor (discriminative LR)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Select Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend.")
    else:
        device = torch.device("cpu")
        print("Using CPU backend.")

    # 3. DataModule
    print("Loading DataModule...")
    df = pd.read_csv(args.csv_file)
    
    # --- Handle unknown writers (-1) by assigning pseudo-writer IDs ---
    unknown_mask = df["writer_id"] == -1
    if unknown_mask.any():
        df.loc[unknown_mask, "writer_id"] = [
            f"pseudo_{i}" for i in range(unknown_mask.sum())
        ]
        print(f"Assigned {unknown_mask.sum()} pseudo-writer IDs")

    # Build writer_id -> int mapping from FULL dataset (before split)
    unique_writers = sorted(df["writer_id"].unique())
    writer_id_map = {str(wid): idx for idx, wid in enumerate(unique_writers)}
    num_writers = len(writer_id_map)
    print(f"Writers: {num_writers} unique")
    
    # Pre-process dataframe into list of dicts with absolute paths
    annotations = []
    for _, row in df.iterrows():
        annotations.append({
            "image_id": str(row["image_id"]),
            "image_path": str(Path(args.image_dir) / row["image_path"]),
            "writer_id": str(row["writer_id"]),
            "pen_id": int(row["pen_id"])
        })
    
    data_module = CircleDataModule(
        annotations=annotations,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # 4. Model Architecture
    mt_writers = num_writers if args.multi_task else 0
    print(f"Creating Model ({args.model_name}, multi_task={args.multi_task})...")
    model = create_model(
        model_name=args.model_name, 
        pretrained=True,
        num_writers=mt_writers
    )
    
    # 5. Loss Function
    if args.multi_task:
        criterion = MultiTaskLoss(alpha=0.3, label_smoothing=0.1)
        print("Using MultiTaskLoss (pen + writer)")
    else:
        criterion = get_loss_fn(label_smoothing=0.1)
    
    # 6. Optimizer (Discriminative LR)
    print(f"Discriminative LR: backbone={args.lr * args.backbone_lr_factor:.1e}, "
          f"heads={args.lr:.1e}")
    optimizer = create_optimizer(
        model, 
        lr=args.lr, 
        weight_decay=0.05,
        backbone_lr_factor=args.backbone_lr_factor
    )
    
    scheduler_cfg = create_scheduler(
        optimizer=optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dl),
        backbone_lr_factor=args.backbone_lr_factor
    )

    # 7. Trainer
    save_path = str(out_dir / f"{args.model_name}_fold{args.fold}.pth")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler_cfg,
        use_amp=True,
        save_path=save_path
    )

    print("Starting training...")
    history = trainer.fit(train_dl, val_dl, epochs=args.epochs)
    print(f"\nTraining complete. Best model saved to {save_path}")


if __name__ == "__main__":
    main()
