"""src.training — Training Pipeline Components

Public API
----------
- compute_metrics: Top-1 Acc and Macro F1
- get_loss_fn: Label smooth cross entropy
- create_optimizer: AdamW factory
- create_scheduler: OneCycleLR factory
- Trainer: Training/Eval Loop
"""

from src.training.metrics import compute_metrics
from src.training.losses import get_loss_fn
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.trainer import Trainer


__all__ = [
    "compute_metrics",
    "get_loss_fn",
    "create_optimizer",
    "create_scheduler",
    "Trainer"
]
