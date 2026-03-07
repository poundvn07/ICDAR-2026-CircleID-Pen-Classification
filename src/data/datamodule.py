"""CircleDataModule — DataLoader factory with GroupKFold splitting.

Implements writer-disjoint cross-validation (AP-02) and configurable
DataLoader creation for training and validation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.data.dataset import CircleDataset
from src.data.transforms import get_train_transform, get_val_transform


# ---------------------------------------------------------------------------
# GroupKFold Split Factory
# ---------------------------------------------------------------------------
def create_group_kfold_splits(
    annotations: list[dict[str, Any]],
    n_splits: int = 5,
) -> list[tuple[list[int], list[int]]]:
    """Create writer-disjoint GroupKFold splits.

    AP-02 compliance: No writer_id appears in both train and val for any fold.

    Args:
        annotations: List of annotation dicts, each must have "writer_id" and
            "pen_id" keys.
        n_splits: Number of cross-validation folds.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
        Each element is a list[int] of indices into the annotations list.
    """
    labels = np.array([r["pen_id"] for r in annotations])
    groups = np.array([r["writer_id"] for r in annotations])

    gkf = GroupKFold(n_splits=n_splits)
    splits: list[tuple[list[int], list[int]]] = []

    for train_idx, val_idx in gkf.split(X=np.zeros(len(annotations)), y=labels, groups=groups):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class CircleDataModule:
    """DataLoader factory with integrated GroupKFold splitting.

    Creates train/val datasets for a specific fold, applying appropriate
    augmentations to each split.

    Args:
        annotations: Full list of annotation records.
        fold: Zero-based fold index to use for train/val split.
        n_splits: Total number of GroupKFold splits.
        batch_size: Batch size for DataLoaders.
        image_size: Target image spatial dimension.
        num_workers: Number of DataLoader worker processes.
    """

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        fold: int = 0,
        n_splits: int = 5,
        batch_size: int = 32,
        image_size: int = 224,
        num_workers: int = 4,
    ) -> None:
        self._annotations = annotations
        self._fold = fold
        self._n_splits = n_splits
        self._batch_size = batch_size
        self._image_size = image_size
        self._num_workers = num_workers

        # Compute splits and select fold
        splits = create_group_kfold_splits(annotations, n_splits=n_splits)
        self._train_indices, self._val_indices = splits[fold]

        # Build fold-specific annotation lists
        self._train_annotations = [annotations[i] for i in self._train_indices]
        self._val_annotations = [annotations[i] for i in self._val_indices]

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader (shuffle=True).

        Returns:
            DataLoader yielding batches with keys: image (B, C, H, W),
            label (B,), writer_id, image_id.
        """
        train_ds = CircleDataset(
            annotations=self._train_annotations,
            transform=get_train_transform(image_size=self._image_size),
            image_size=self._image_size,
        )
        return DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader (shuffle=False).

        Returns:
            DataLoader yielding batches with keys: image (B, C, H, W),
            label (B,), writer_id, image_id.
        """
        val_ds = CircleDataset(
            annotations=self._val_annotations,
            transform=get_val_transform(image_size=self._image_size),
            image_size=self._image_size,
        )
        return DataLoader(
            val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=False,
        )
