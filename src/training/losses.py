"""Loss functions module for training.

Provides access to robust validation losses like CrossEntropy with Label Smoothing.
"""
import torch.nn as nn


def get_loss_fn(label_smoothing: float = 0.1) -> nn.Module:
    """Get the primary loss function for training.
    
    Uses Cross Entropy with label smoothing. Smoothing (default=0.1) acts as
    regularization, penalizing the model for making overly confident predictions.
    This is highly beneficial in datasets with human annotation noise or hard
    to distinguish patterns like subtle pen ink differences.
    
    Args:
        label_smoothing: Amount of smoothing (0.0 to 1.0).
        
    Returns:
        Configured PyTorch CrossEntropyLoss criterion.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
