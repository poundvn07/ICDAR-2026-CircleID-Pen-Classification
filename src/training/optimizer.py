"""Optimizer and Learning Rate Scheduler module.

Supports Discriminative Learning Rates: backbone (pretrained) gets a lower LR
than randomly-initialized heads to preserve learned features.
"""
from typing import Dict, Any, List
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import OneCycleLR


def create_optimizer(
    model: nn.Module, 
    lr: float = 1e-3, 
    weight_decay: float = 0.05,
    backbone_lr_factor: float = 0.1,
) -> Optimizer:
    """Create AdamW optimizer with discriminative learning rates.
    
    The backbone (pretrained on ImageNet) receives a lower learning rate
    to avoid destroying useful pretrained features, while the head(s)
    (randomly initialized) get the full learning rate.
    
    Args:
        model: The neural network module (must have `.backbone` and `.head`).
        lr: Base learning rate for the head(s).
        weight_decay: L2 penalty term.
        backbone_lr_factor: Multiplier for backbone LR (e.g., 0.1 means
            backbone_lr = lr * 0.1). Set to 1.0 to disable discriminative LR.
        
    Returns:
        Configured AdamW optimizer with per-group learning rates.
    """
    backbone_lr = lr * backbone_lr_factor
    
    # Collect backbone parameters
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = set(id(p) for p in backbone_params)
    
    # Collect all non-backbone parameters (head, writer_head, etc.)
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    
    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": lr, "name": "heads"},
    ]
    
    return AdamW(param_groups, weight_decay=weight_decay)


def create_scheduler(
    optimizer: Optimizer, 
    max_lr: float, 
    epochs: int, 
    steps_per_epoch: int,
    pct_start: float = 0.1,
    backbone_lr_factor: float = 0.1,
) -> Dict[str, Any]:
    """Create a learning rate scheduler (OneCycleLR).
    
    Supports per-group max_lr for discriminative learning rates.
    
    Args:
        optimizer: The PyTorch Optimizer (may have multiple param groups).
        max_lr: The peak learning rate for the head(s).
        epochs: Total number of epochs for the cycle.
        steps_per_epoch: Number of batches per epoch.
        pct_start: The percentage of the cycle spent increasing the LR.
        backbone_lr_factor: Factor for backbone's max_lr.
        
    Returns:
        Dictionary configuration for the training loop.
    """
    # Build per-group max_lr list matching optimizer param_groups
    num_groups = len(optimizer.param_groups)
    if num_groups == 2:
        # [backbone, heads]
        max_lrs: List[float] | float = [max_lr * backbone_lr_factor, max_lr]
    else:
        max_lrs = max_lr
    
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lrs,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        anneal_strategy="cos"
    )
    
    return {
        "scheduler": scheduler,
        "interval": "step"
    }
