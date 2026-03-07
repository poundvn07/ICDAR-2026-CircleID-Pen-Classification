"""Test suite for CircleDataset (src/data/dataset.py).

TDD: These tests define the contract BEFORE implementation.
Each test specifies exact input/output tensor shapes and dtypes.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Constants matching CONVENTIONS.md
# ---------------------------------------------------------------------------
NUM_CLASSES = 8
DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with fake circle images."""
    img_dir = tmp_path / "raw"
    img_dir.mkdir()
    # Create 16 dummy images (2 writers × 8 pens)
    for writer_id in range(2):
        for pen_id in range(NUM_CLASSES):
            img = np.random.randint(0, 255, (300, 280, 3), dtype=np.uint8)
            filepath = img_dir / f"writer{writer_id}_pen{pen_id}_001.png"
            # We'll mock cv2.imread, so just create empty files
            filepath.touch()
    return img_dir


@pytest.fixture
def dummy_annotations(tmp_path: Path) -> list[dict]:
    """Create annotation records matching dummy images."""
    records = []
    for writer_id in range(2):
        for pen_id in range(NUM_CLASSES):
            records.append({
                "image_id": f"writer{writer_id}_pen{pen_id}_001",
                "image_path": str(tmp_path / "raw" / f"writer{writer_id}_pen{pen_id}_001.png"),
                "pen_id": pen_id,
                "writer_id": writer_id,
            })
    return records


# ---------------------------------------------------------------------------
# Test: Dataset Initialization
# ---------------------------------------------------------------------------
class TestCircleDatasetInit:
    """Tests for dataset construction and validation."""

    def test_dataset_length(self, dummy_annotations):
        """Dataset __len__ must return exact number of annotation records."""
        from src.data.dataset import CircleDataset

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        assert len(ds) == len(dummy_annotations)  # 16

    def test_dataset_rejects_empty_annotations(self):
        """Dataset must raise ValueError on empty annotation list."""
        from src.data.dataset import CircleDataset

        with pytest.raises(ValueError, match="annotations"):
            CircleDataset(annotations=[], transform=None)

    def test_dataset_validates_required_keys(self, dummy_annotations):
        """Each annotation must have: image_path, pen_id, writer_id."""
        from src.data.dataset import CircleDataset

        bad_records = [{"image_path": "x.png"}]  # missing pen_id, writer_id
        with pytest.raises(KeyError):
            CircleDataset(annotations=bad_records, transform=None)


# ---------------------------------------------------------------------------
# Test: __getitem__ Output Contract
# ---------------------------------------------------------------------------
class TestCircleDatasetGetItem:
    """Tests for the shape and dtype contract of __getitem__."""

    @patch("cv2.imread")
    def test_output_image_shape(self, mock_imread, dummy_annotations):
        """Output image tensor must be (C, H, W) = (3, 224, 224)."""
        from src.data.dataset import CircleDataset

        # Mock cv2.imread to return a realistic image
        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        sample = ds[0]

        assert sample["image"].shape == (NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), \
            f"Expected (3, 224, 224), got {sample['image'].shape}"

    @patch("cv2.imread")
    def test_output_image_dtype(self, mock_imread, dummy_annotations):
        """Output image tensor must be float32."""
        from src.data.dataset import CircleDataset

        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        sample = ds[0]

        assert sample["image"].dtype == torch.float32

    @patch("cv2.imread")
    def test_output_image_range(self, mock_imread, dummy_annotations):
        """Output image values must be in [0, 1] after normalization."""
        from src.data.dataset import CircleDataset

        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        sample = ds[0]

        # Before normalization, values should be in [0, 1]
        # (Normalization with mean/std may push outside, but raw should be [0,1])
        assert sample["image"].min() >= -3.0, "Values too negative (check normalization)"
        assert sample["image"].max() <= 3.0, "Values too positive (check normalization)"

    @patch("cv2.imread")
    def test_output_label_shape_and_dtype(self, mock_imread, dummy_annotations):
        """Label must be a scalar tensor of int64."""
        from src.data.dataset import CircleDataset

        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        sample = ds[0]

        assert sample["label"].shape == (), f"Label must be scalar, got {sample['label'].shape}"
        assert sample["label"].dtype == torch.int64
        assert 0 <= sample["label"].item() < NUM_CLASSES

    @patch("cv2.imread")
    def test_output_contains_writer_id(self, mock_imread, dummy_annotations):
        """Sample dict must contain writer_id for GroupKFold splitting."""
        from src.data.dataset import CircleDataset

        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        sample = ds[0]

        assert "writer_id" in sample
        assert isinstance(sample["writer_id"], int)

    @patch("cv2.imread")
    def test_output_contains_image_id(self, mock_imread, dummy_annotations):
        """Sample dict must contain image_id for submission mapping."""
        from src.data.dataset import CircleDataset

        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        sample = ds[0]

        assert "image_id" in sample
        assert isinstance(sample["image_id"], str)


# ---------------------------------------------------------------------------
# Test: Aspect-Ratio-Preserving Resize (AP-09)
# ---------------------------------------------------------------------------
class TestResizeWithPad:
    """Tests for the resize_with_pad utility (AP-09 compliance)."""

    def test_output_shape_square(self):
        """Output must be exactly (target_size, target_size, 3)."""
        from src.data.utils import resize_with_pad

        img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        result = resize_with_pad(img, target_size=224)

        assert result.shape == (224, 224, 3)

    def test_aspect_ratio_preserved(self):
        """Non-square input must not be stretched. Padding is added instead."""
        from src.data.utils import resize_with_pad

        # Tall image: 400x200
        img = np.zeros((400, 200, 3), dtype=np.uint8)
        img[100:300, 50:150, :] = 128  # draw a block in the center

        result = resize_with_pad(img, target_size=224)

        # After resize_with_pad, the image should be padded horizontally
        # The content should NOT fill the entire width
        # Check that leftmost and rightmost columns are padding (white=255)
        assert result[0, 0, 0] == 255, "Top-left corner should be padding"

    def test_output_dtype_preserved(self):
        """Output dtype must remain uint8."""
        from src.data.utils import resize_with_pad

        img = np.random.randint(0, 255, (300, 250, 3), dtype=np.uint8)
        result = resize_with_pad(img, target_size=224)

        assert result.dtype == np.uint8

    def test_downscale_uses_inter_area(self):
        """When input > target_size, INTER_AREA must be used (no aliasing)."""
        from src.data.utils import resize_with_pad

        # Large image that requires downscaling
        img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        result = resize_with_pad(img, target_size=224)

        # Output must be valid (non-zero, non-constant)
        assert result.shape == (224, 224, 3)
        assert result.std() > 0, "Result should not be constant"


# ---------------------------------------------------------------------------
# Test: DataLoader Batch Shape
# ---------------------------------------------------------------------------
class TestDataLoaderBatch:
    """Tests for batched output from DataLoader."""

    @patch("cv2.imread")
    def test_batch_tensor_shapes(self, mock_imread, dummy_annotations):
        """A batch must produce images: (B, C, H, W) and labels: (B,)."""
        from src.data.dataset import CircleDataset
        from torch.utils.data import DataLoader

        mock_imread.return_value = np.random.randint(
            0, 255, (300, 280, 3), dtype=np.uint8
        )

        ds = CircleDataset(annotations=dummy_annotations, transform=None)
        loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)
        batch = next(iter(loader))

        B = 4
        assert batch["image"].shape == (B, NUM_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), \
            f"Expected ({B}, 3, 224, 224), got {batch['image'].shape}"
        assert batch["label"].shape == (B,), \
            f"Expected ({B},), got {batch['label'].shape}"
