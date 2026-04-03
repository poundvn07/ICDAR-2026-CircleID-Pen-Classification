"""Loss functions module for training.

Provides standard and multi-task loss functions.
"""
import torch
import torch.nn as nn


def get_loss_fn(label_smoothing: float = 0.1) -> nn.Module:
    """Get the primary loss function for training.
    
    Uses Cross Entropy with label smoothing to penalize overconfident
    predictions and counteract annotation noise.
    
    Args:
        label_smoothing: Amount of smoothing (0.0 to 1.0).
        
    Returns:
        Configured PyTorch CrossEntropyLoss criterion.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning (pen + writer classification).
    
    total_loss = pen_loss + alpha * writer_loss
    
    The writer auxiliary task forces the backbone to learn writer-invariant
    pen features by exposing it to writer identity signal.
    
    Args:
        alpha: Weight for the auxiliary writer loss (default 0.3).
        label_smoothing: Label smoothing for both CE losses.
    """
    
    def __init__(self, alpha: float = 0.3, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.pen_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.writer_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(
        self, 
        pen_logits: torch.Tensor, 
        pen_targets: torch.Tensor,
        writer_logits: torch.Tensor | None = None,
        writer_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined multi-task loss.
        
        Args:
            pen_logits: Pen class logits `(B, NUM_PEN_CLASSES)`.
            pen_targets: Pen class labels `(B,)`.
            writer_logits: Optional writer logits `(B, NUM_WRITERS)`.
            writer_targets: Optional writer labels `(B,)`.
            
        Returns:
            Combined scalar loss.
        """
        pen_loss = self.pen_criterion(pen_logits, pen_targets)
        
        if writer_logits is not None and writer_targets is not None:
            writer_loss = self.writer_criterion(writer_logits, writer_targets)
            return pen_loss + self.alpha * writer_loss
        
        return pen_loss
