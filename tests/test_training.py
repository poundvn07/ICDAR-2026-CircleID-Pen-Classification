"""Test suite for Training Components (src/training/).

Validates metrics (Top-1 Acc, Macro F1), custom losses, and optimizer/scheduler setup.
"""
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# ---------------------------------------------------------------------------
# Test: Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    """Tests for evaluation metrics."""

    def test_metrics_calculation(self):
        """Test accuracy and Macro F1 score calculation."""
        from src.training.metrics import compute_metrics

        # Predict classes 0, 1, 2, 3
        # True classes    0, 2, 2, 3
        # Match mask: [True, False, True, True] -> Acc: 0.75
        preds = torch.tensor([[5.0, 1.0, 0.0, 0.0],
                              [1.0, 5.0, 0.0, 0.0],
                              [0.0, 1.0, 5.0, 0.0],
                              [0.0, 0.0, 1.0, 5.0]])
        targets = torch.tensor([0, 2, 2, 3])

        metrics = compute_metrics(preds, targets)
        
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert pytest.approx(metrics["accuracy"], 0.01) == 0.75


# ---------------------------------------------------------------------------
# Test: Losses
# ---------------------------------------------------------------------------
class TestLosses:
    """Tests for loss functions."""

    def test_loss_with_label_smoothing(self):
        """Test that the loss correctly applies label smoothing."""
        from src.training.losses import get_loss_fn

        loss_fn = get_loss_fn(label_smoothing=0.1)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.label_smoothing == 0.1

        preds = torch.randn(4, 8)
        targets = torch.tensor([0, 1, 2, 3])
        
        loss = loss_fn(preds, targets)
        assert loss.ndim == 0  # Scalar output
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# Test: Optimizer & Scheduler
# ---------------------------------------------------------------------------
class TestOptimizerScheduler:
    """Tests for optimizer and LR scheduler factories."""

    @pytest.fixture
    def dummy_model(self):
        """Dummy model for optimizer injection."""
        return nn.Linear(10, 8)

    def test_create_optimizer(self, dummy_model):
        """Test AdamW optimizer creation."""
        from src.training.optimizer import create_optimizer

        lr = 1e-3
        weight_decay = 0.05
        optimizer = create_optimizer(dummy_model, lr=lr, weight_decay=weight_decay)

        assert isinstance(optimizer, AdamW)
        assert optimizer.param_groups[0]["lr"] == lr
        assert optimizer.param_groups[0]["weight_decay"] == weight_decay

    def test_create_scheduler(self, dummy_model):
        """Test OneCycleLR scheduler creation."""
        from src.training.optimizer import create_optimizer, create_scheduler

        optimizer = create_optimizer(dummy_model, lr=1e-3)
        scheduler_dict = create_scheduler(
            optimizer, 
            max_lr=1e-3, 
            epochs=10, 
            steps_per_epoch=100
        )

        assert "scheduler" in scheduler_dict
        assert "interval" in scheduler_dict
        assert scheduler_dict["interval"] == "step"
        assert isinstance(scheduler_dict["scheduler"], OneCycleLR)
