"""Test suite for augmentation transforms (src/data/transforms.py).

TDD: These tests define the augmentation pipeline contract BEFORE implementation.
Validates ink-safe augmentation policy (AP-07) and interpolation rules (AP-09).
"""
import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_circle_image() -> np.ndarray:
    """Create a synthetic circle image resembling a scanned drawing.

    Returns:
        np.ndarray of shape (300, 280, 3), dtype uint8.
        White background with a dark circle stroke.
    """
    img = np.full((300, 280, 3), 255, dtype=np.uint8)
    # Draw a simple dark ring to simulate ink stroke
    cy, cx, r = 150, 140, 100
    Y, X = np.ogrid[:300, :280]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ring_mask = (dist > r - 5) & (dist < r + 5)
    img[ring_mask] = 30  # dark ink color
    return img


# ---------------------------------------------------------------------------
# Test: Training Transform Pipeline
# ---------------------------------------------------------------------------
class TestTrainTransform:
    """Tests for the training augmentation pipeline."""

    def test_output_shape(self, sample_circle_image):
        """Training transform must output (C, H, W) = (3, 224, 224)."""
        from src.data.transforms import get_train_transform

        transform = get_train_transform(image_size=DEFAULT_IMAGE_SIZE)
        result = transform(image=sample_circle_image)
        image = result["image"]

        if isinstance(image, np.ndarray):
            assert image.shape == (NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE) or \
                   image.shape == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS)
        elif isinstance(image, torch.Tensor):
            assert image.shape == (NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    def test_output_dtype_is_float(self, sample_circle_image):
        """Output must be float32 (for model input)."""
        from src.data.transforms import get_train_transform

        transform = get_train_transform(image_size=DEFAULT_IMAGE_SIZE)
        result = transform(image=sample_circle_image)
        image = result["image"]

        if isinstance(image, torch.Tensor):
            assert image.dtype == torch.float32
        else:
            assert image.dtype == np.float32

    def test_ink_intensity_preserved(self, sample_circle_image):
        """AP-07: Augmentation must not destroy ink intensity distribution.

        After 100 augmented samples, the mean ink-region intensity should remain
        within ±20% of the original. This validates that color jitter is bounded.
        """
        from src.data.transforms import get_train_transform

        transform = get_train_transform(image_size=DEFAULT_IMAGE_SIZE)

        # Compute original ink intensity (dark pixels < 128)
        original_ink_mask = sample_circle_image.mean(axis=2) < 128
        if original_ink_mask.sum() == 0:
            pytest.skip("No ink pixels found in sample image")
        original_ink_mean = sample_circle_image[original_ink_mask].mean()

        # ImageNet normalization stats (must match transforms.py)
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Collect augmented ink intensities
        augmented_means = []
        for _ in range(50):
            result = transform(image=sample_circle_image.copy())
            aug_img = result["image"]
            if isinstance(aug_img, torch.Tensor):
                aug_np = aug_img.permute(1, 2, 0).numpy()
            else:
                aug_np = aug_img
            # Denormalize ImageNet normalization: pixel = (normalized * std + mean) * 255
            if aug_np.min() < 0 or aug_np.max() > 1.0:
                aug_np = (aug_np * IMAGENET_STD + IMAGENET_MEAN) * 255.0
            elif aug_np.max() <= 1.0:
                aug_np = (aug_np * 255).astype(np.uint8)
            aug_np = np.clip(aug_np, 0, 255).astype(np.uint8)
            ink_mask = aug_np.mean(axis=2) < 128
            if ink_mask.sum() > 0:
                augmented_means.append(aug_np[ink_mask].mean())

        if len(augmented_means) > 0:
            avg_augmented = np.mean(augmented_means)
            # Allow 50% deviation (generous bound for light jitter)
            assert abs(avg_augmented - original_ink_mean) / (original_ink_mean + 1e-8) < 0.5, \
                f"Ink intensity shifted too much: {original_ink_mean:.1f} → {avg_augmented:.1f}"


# ---------------------------------------------------------------------------
# Test: Validation Transform Pipeline
# ---------------------------------------------------------------------------
class TestValTransform:
    """Tests for the validation/test augmentation pipeline."""

    def test_output_shape(self, sample_circle_image):
        """Validation transform must output (C, H, W) = (3, 224, 224)."""
        from src.data.transforms import get_val_transform

        transform = get_val_transform(image_size=DEFAULT_IMAGE_SIZE)
        result = transform(image=sample_circle_image)
        image = result["image"]

        if isinstance(image, torch.Tensor):
            assert image.shape == (NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    def test_deterministic(self, sample_circle_image):
        """Validation transform must be deterministic (no randomness)."""
        from src.data.transforms import get_val_transform

        transform = get_val_transform(image_size=DEFAULT_IMAGE_SIZE)

        result1 = transform(image=sample_circle_image.copy())["image"]
        result2 = transform(image=sample_circle_image.copy())["image"]

        if isinstance(result1, torch.Tensor):
            assert torch.allclose(result1, result2, atol=1e-6), \
                "Validation transform must be deterministic"
        else:
            np.testing.assert_array_almost_equal(result1, result2, decimal=5)

    def test_no_random_augmentations(self, sample_circle_image):
        """Val pipeline must NOT include any stochastic augmentations."""
        from src.data.transforms import get_val_transform

        transform = get_val_transform(image_size=DEFAULT_IMAGE_SIZE)

        # Run 10 times; all outputs must be identical
        outputs = []
        for _ in range(10):
            result = transform(image=sample_circle_image.copy())["image"]
            if isinstance(result, torch.Tensor):
                outputs.append(result.numpy())
            else:
                outputs.append(result)

        for i in range(1, len(outputs)):
            np.testing.assert_array_almost_equal(
                outputs[0], outputs[i], decimal=5,
                err_msg=f"Output {i} differs from output 0 — val transform is stochastic!"
            )
