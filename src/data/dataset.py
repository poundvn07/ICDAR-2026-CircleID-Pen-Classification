"""CircleDataset — PyTorch Dataset for ICDAR 2026 CircleID pen classification.

Loads hand-drawn circle images, applies aspect-ratio-preserving resize (AP-09),
optional Albumentations transform, and returns tensors matching the contract:
  - image:    (C, H, W) float32
  - label:    scalar int64
  - writer_id: int
  - image_id:  str
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A

from src.data.utils import resize_with_pad


# ---------------------------------------------------------------------------
# Constants (from CONVENTIONS.md §3.1)
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 8
DEFAULT_IMAGE_SIZE: int = 224

# Required keys in each annotation record
_REQUIRED_KEYS = {"image_path", "pen_id", "writer_id"}


class CircleDataset(Dataset):
    """Dataset for pen-classification circle images.

    Args:
        annotations: List of dicts, each with keys:
            - image_id (str): unique sample identifier
            - image_path (str): absolute path to image file
            - pen_id (int): label in [0, NUM_CLASSES)
            - writer_id (int): writer group id (for GroupKFold)
        transform: Optional Albumentations Compose pipeline.
            If None, a default deterministic pipeline (resize + normalize +
            ToTensorV2) is applied.
        image_size: Target spatial dimension (default 224).

    Raises:
        ValueError: If annotations is empty.
        KeyError: If any record is missing required keys (image_path, pen_id,
            writer_id).
    """

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        transform: Optional[A.Compose] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if not annotations:
            raise ValueError("annotations must be a non-empty list of records.")

        # Validate required keys on first record (fail-fast)
        missing = _REQUIRED_KEYS - set(annotations[0].keys())
        if missing:
            raise KeyError(
                f"Annotation records missing required keys: {missing}. "
                f"Required: {_REQUIRED_KEYS}"
            )

        self._annotations = annotations
        self._image_size = image_size

        if transform is not None:
            self._transform = transform
        else:
            # Default: deterministic resize + normalize + to tensor
            from src.data.transforms import get_val_transform
            self._transform = get_val_transform(image_size=image_size)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample dict.

        Returns:
            dict with keys:
                image    — torch.Tensor, shape (C, H, W), dtype float32
                label    — torch.Tensor, scalar, dtype int64
                writer_id — int
                image_id  — str
        """
        record = self._annotations[idx]

        # --- Load image (BGR → RGB) ---
        img = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)  # (H, W, 3) BGR
        if img is None:
            raise FileNotFoundError(
                f"Cannot read image at index {idx}: {record['image_path']}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H, W, 3) RGB

        # --- Aspect-ratio-preserving resize + padding (AP-09) ---
        img = resize_with_pad(img, target_size=self._image_size)  # (S, S, 3) uint8

        # --- Apply augmentation / normalization pipeline ---
        transformed = self._transform(image=img)
        image_tensor: torch.Tensor = transformed["image"]  # (C, H, W) float32

        # --- Label ---
        label = torch.tensor(record["pen_id"], dtype=torch.int64)  # scalar

        return {
            "image": image_tensor,
            "label": label,
            "writer_id": record["writer_id"],
            "image_id": record.get("image_id", ""),
        }
