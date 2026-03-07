"""src.data — Data loading, transforms, and dataset utilities.

Public API
----------
- CircleDataset: PyTorch Dataset for pen-classification circle images.
- CircleDataModule: DataLoader factory with GroupKFold splitting.
- create_group_kfold_splits: Writer-disjoint cross-validation splits.
- get_train_transform: Training augmentation pipeline.
- get_val_transform: Deterministic validation transform pipeline.
- resize_with_pad: Aspect-ratio-preserving resize with padding (AP-09).
- load_image: Load image from disk with resize+pad.
"""
from src.data.dataset import CircleDataset
from src.data.datamodule import CircleDataModule, create_group_kfold_splits
from src.data.transforms import get_train_transform, get_val_transform
from src.data.utils import resize_with_pad, load_image

__all__ = [
    "CircleDataset",
    "CircleDataModule",
    "create_group_kfold_splits",
    "get_train_transform",
    "get_val_transform",
    "resize_with_pad",
    "load_image",
]
