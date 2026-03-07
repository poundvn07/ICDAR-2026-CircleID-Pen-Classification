"""Optimizer and Learning Rate Scheduler module.

Configures AdamW and OneCycleLR per modern deep learning best practices.
"""
from typing import Dict, Any
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import OneCycleLR


def create_optimizer(
    model: nn.Module, 
    lr: float = 1e-3, 
    weight_decay: float = 0.05
) -> Optimizer:
    """Create the AdamW optimizer.
    
    AdamW decouples weight decay from the optimization steps, which is crucial
    for training deep vision models (ConvNeXt, ViT) and preventing overfitting.
    
    Args:
        model: The neural network module.
        lr: Base learning rate.
        weight_decay: L2 penalty term decoupled from gradient updates.
        
    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters normally needing no weight decay (bias, norm Layers)
    # is a common optimization, but standard AdamW handles our setup well enough as baseline.
    return AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )


def create_scheduler(
    optimizer: Optimizer, 
    max_lr: float, 
    epochs: int, 
    steps_per_epoch: int,
    pct_start: float = 0.1
) -> Dict[str, Any]:
    """Create a learning rate scheduler (OneCycleLR).
    
    OneCycleLR warms up the learning rate from a small value to `max_lr` over 
    `pct_start` of the training cycle, then aggressively anneals it down. This
    leads to super-convergence and acts as strong regularization.
    
    Args:
        optimizer: The PyTorch Optimizer.
        max_lr: The peak learning rate to achieve.
        epochs: Total number of epochs for the cycle.
        steps_per_epoch: Number of batches per epoch.
        pct_start: The percentage of the cycle spent increasing the LR.
        
    Returns:
        Dictionary configuration suitable for standard training loops
        and PyTorch Lightning (if used later).
    """
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy="cos"
    )
    
    return {
        "scheduler": scheduler,
        "interval": "step"  # Scheduler needs step() called after every batch
    }
