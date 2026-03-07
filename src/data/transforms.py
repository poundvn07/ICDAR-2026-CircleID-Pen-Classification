"""Albumentations transform pipelines for training and validation.

Implements ink-safe augmentation policy (AP-07) and proper resize strategy (AP-09).
Training pipeline includes light geometric and color augmentations that preserve
ink texture features. Validation pipeline is fully deterministic.

Compatible with Albumentations >= 2.0 API.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size: int = 224) -> A.Compose:
    """Build the training augmentation pipeline.

    Augmentations comply with AP-07 (ink-safe jitter bounds)
    and ADR-002 (restricted augmentation policy).

    Args:
        image_size: Target spatial dimension (height = width).

    Returns:
        Albumentations Compose pipeline that accepts image as np.ndarray
        and returns {"image": torch.Tensor} with shape (C, H, W).
    """
    return A.Compose([
        # --- Geometric (do not alter pixel intensities) ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            border_mode=0,  # cv2.BORDER_CONSTANT
            fill=255,       # white padding
            p=0.5,
        ),

        # --- Ink-safe color augmentation (AP-07: bounded jitter) ---
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05,
            hue=0.02,
            p=0.3,
        ),

        # --- Light noise (bounded per ADR-002) ---
        A.GaussNoise(
            std_range=(0.02, 0.05),
            mean_range=(0.0, 0.0),
            p=0.2,
        ),

        # --- Coarse dropout for regularization ---
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(int(image_size * 0.04), int(image_size * 0.08)),
            hole_width_range=(int(image_size * 0.04), int(image_size * 0.08)),
            fill=255,  # white fill to match background
            p=0.2,
        ),

        # --- Resize to target (already padded, but ensure exact size) ---
        A.Resize(height=image_size, width=image_size),

        # --- Normalize to [0, 1] then to ImageNet stats ---
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),

        # --- Convert HWC float32 ndarray → CHW float32 Tensor ---
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    """Build the validation/test transform pipeline.

    Fully deterministic — no stochastic augmentations. Only resize,
    normalize, and tensor conversion.

    Args:
        image_size: Target spatial dimension (height = width).

    Returns:
        Albumentations Compose pipeline that accepts image as np.ndarray
        and returns {"image": torch.Tensor} with shape (C, H, W).
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
