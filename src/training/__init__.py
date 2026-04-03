"""src.training — Training Pipeline Components

Public API
----------
- compute_metrics: Top-1 Acc and Macro F1
- get_loss_fn: Label smooth cross entropy
- MultiTaskLoss: Combined pen + writer loss
- create_optimizer: AdamW factory with discriminative LR
- create_scheduler: OneCycleLR factory with per-group max_lr
- Trainer: Training/Eval Loop with multi-task support
"""

from src.training.metrics import compute_metrics
from src.training.losses import get_loss_fn, MultiTaskLoss
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.trainer import Trainer


__all__ = [
    "compute_metrics",
    "get_loss_fn",
    "MultiTaskLoss",
    "create_optimizer",
    "create_scheduler",
    "Trainer"
]
