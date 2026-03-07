"""Utility functions for data preprocessing.

Implements aspect-ratio-preserving resize with padding (AP-09 compliance).
"""
import cv2
import numpy as np


def resize_with_pad(
    img: np.ndarray,
    target_size: int = 224,
    pad_color: int = 255,
) -> np.ndarray:
    """Resize an image while preserving aspect ratio, padding to square.

    Uses INTER_AREA for downscaling (anti-aliased) and INTER_CUBIC for
    upscaling to preserve ink granularity details (AP-09).

    Args:
        img: Input image as uint8 ndarray with shape (H, W, C).
        target_size: Target square dimension in pixels.
        pad_color: Padding fill value (default 255 = white background).

    Returns:
        Resized and padded image as uint8 ndarray with shape
        (target_size, target_size, C).

    Raises:
        ValueError: If img is None or has invalid shape.
    """
    if img is None or img.ndim < 2:
        raise ValueError("Input image must be a valid ndarray with >= 2 dimensions.")

    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    # AP-09: Use INTER_AREA for downscale, INTER_CUBIC for upscale
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)  # (new_h, new_w, C)

    # Create padded canvas
    if img.ndim == 3:
        canvas = np.full(
            (target_size, target_size, img.shape[2]), pad_color, dtype=np.uint8
        )
    else:
        canvas = np.full((target_size, target_size), pad_color, dtype=np.uint8)

    # Center the resized image on the canvas
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    return canvas


def load_image(path: str, target_size: int = 224) -> np.ndarray:
    """Load an image from disk, convert to RGB, and resize with padding.

    Args:
        path: Absolute path to the image file.
        target_size: Target square dimension in pixels.

    Returns:
        Image as uint8 ndarray with shape (target_size, target_size, 3).

    Raises:
        FileNotFoundError: If path does not exist or image cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # (H, W, 3) in BGR
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # (H, W, 3) in RGB
    img = resize_with_pad(img, target_size=target_size)  # (target_size, target_size, 3)
    return img
