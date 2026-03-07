"""Classification head module.

Implements a robust linear head for mapping extracted features to pen classes,
including regularization (LayerNorm, Dropout) as defined in OpenSpec.
"""
import torch
import torch.nn as nn


class PenClassifierHead(nn.Module):
    """Robust classification head for pen identification.

    Maps pooled feature vectors `(B, D)` to class logits `(B, NUM_CLASSES)`.
    Includes LayerNorm for stability and Dropout for regularization, as
    the dataset is relatively small and model capacity (ConvNeXt) is high.

    Args:
        in_features: Dimension of the input feature vector `(D,)`.
        num_classes: Number of output classes (8 for CircleID).
        p_dropout: Dropout probability.
    """

    def __init__(self, in_features: int, num_classes: int = 8, p_dropout: float = 0.3) -> None:
        super().__init__()
        
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features, num_classes)
        )

        # Initialize linear layer
        nn.init.xavier_uniform_(self.head[2].weight)
        if self.head[2].bias is not None:
            nn.init.zeros_(self.head[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Feature tensor of shape `(B, D)`.
            
        Returns:
            Logits tensor of shape `(B, NUM_CLASSES)`.
        """
        return self.head(x)
