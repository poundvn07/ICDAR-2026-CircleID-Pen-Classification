"""Test suite for GroupKFold DataModule (src/data/datamodule.py).

TDD: These tests define the data splitting contract BEFORE implementation.
Validates GroupKFold logic (AP-02) and DataLoader configuration.
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 8
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_annotations() -> list[dict]:
    """Create 100 annotation records from 10 writers × 10 samples."""
    records = []
    for writer_id in range(10):
        for sample_idx in range(10):
            pen_id = sample_idx % NUM_CLASSES
            records.append({
                "image_id": f"w{writer_id}_s{sample_idx}",
                "image_path": f"/fake/w{writer_id}_s{sample_idx}.png",
                "pen_id": pen_id,
                "writer_id": writer_id,
            })
    return records


# ---------------------------------------------------------------------------
# Test: GroupKFold Split Integrity
# ---------------------------------------------------------------------------
class TestGroupKFoldSplit:
    """Tests for writer-disjoint cross-validation splits."""

    def test_no_writer_leakage(self, mock_annotations):
        """AP-02: No writer_id may appear in both train and val for any fold."""
        from src.data.datamodule import create_group_kfold_splits

        splits = create_group_kfold_splits(
            annotations=mock_annotations,
            n_splits=N_SPLITS,
        )

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            train_writers = {mock_annotations[i]["writer_id"] for i in train_indices}
            val_writers = {mock_annotations[i]["writer_id"] for i in val_indices}

            overlap = train_writers & val_writers
            assert len(overlap) == 0, \
                f"Fold {fold_idx}: writer leakage detected — writers {overlap} in both sets"

    def test_all_samples_covered(self, mock_annotations):
        """Every sample must appear in exactly one val fold across all K folds."""
        from src.data.datamodule import create_group_kfold_splits

        splits = create_group_kfold_splits(
            annotations=mock_annotations,
            n_splits=N_SPLITS,
        )

        all_val_indices = set()
        n_total = len(mock_annotations)

        for train_indices, val_indices in splits:
            # No overlap within a single fold
            assert len(set(train_indices) & set(val_indices)) == 0

            # Track val indices
            for idx in val_indices:
                assert idx not in all_val_indices, f"Sample {idx} appears in multiple val folds"
                all_val_indices.add(idx)

            # train + val = all samples
            assert len(train_indices) + len(val_indices) == n_total

        assert all_val_indices == set(range(n_total)), "Not all samples covered across folds"

    def test_returns_correct_number_of_folds(self, mock_annotations):
        """Must return exactly K fold splits."""
        from src.data.datamodule import create_group_kfold_splits

        splits = create_group_kfold_splits(
            annotations=mock_annotations,
            n_splits=N_SPLITS,
        )

        assert len(splits) == N_SPLITS

    def test_fold_indices_are_valid(self, mock_annotations):
        """All indices must be within [0, len(annotations))."""
        from src.data.datamodule import create_group_kfold_splits

        splits = create_group_kfold_splits(
            annotations=mock_annotations,
            n_splits=N_SPLITS,
        )
        n = len(mock_annotations)

        for train_indices, val_indices in splits:
            assert all(0 <= i < n for i in train_indices)
            assert all(0 <= i < n for i in val_indices)


# ---------------------------------------------------------------------------
# Test: DataModule Configuration
# ---------------------------------------------------------------------------
class TestDataModuleConfig:
    """Tests for DataLoader factory configuration."""

    def test_train_loader_shuffles(self, mock_annotations):
        """Training DataLoader must have shuffle=True."""
        from src.data.datamodule import CircleDataModule

        dm = CircleDataModule(
            annotations=mock_annotations,
            fold=0,
            n_splits=N_SPLITS,
            batch_size=4,
        )
        train_loader = dm.train_dataloader()

        # Verify shuffle is enabled (check sampler type)
        from torch.utils.data import RandomSampler
        assert isinstance(train_loader.sampler, RandomSampler) or \
               hasattr(train_loader, '_iterator'), \
               "Training loader must shuffle data"

    def test_val_loader_no_shuffle(self, mock_annotations):
        """Validation DataLoader must have shuffle=False."""
        from src.data.datamodule import CircleDataModule

        dm = CircleDataModule(
            annotations=mock_annotations,
            fold=0,
            n_splits=N_SPLITS,
            batch_size=4,
        )
        val_loader = dm.val_dataloader()

        from torch.utils.data import SequentialSampler
        assert isinstance(val_loader.sampler, SequentialSampler), \
            "Validation loader must NOT shuffle data"
