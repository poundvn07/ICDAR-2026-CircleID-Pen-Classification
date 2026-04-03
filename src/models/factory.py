"""Model factory module.

Combines the backbone and the classification head into a cohesive model,
and provides utilities for freezing/unfreezing layers for transfer learning.
Supports optional multi-task learning with an auxiliary writer-identification head.
"""
import torch
import torch.nn as nn

from src.models.backbone import create_backbone
from src.models.classifier import PenClassifierHead


class PenClassificationModel(nn.Module):
    """End-to-End model for Pen Classification.
    
    Supports optional multi-task learning: when `writer_head` is present,
    the forward pass returns `(pen_logits, writer_logits)` during training
    and only `pen_logits` during eval.
    
    Args:
        backbone: Feature extraction nn.Module (output shape: `(B, D)`).
        head: Classification head nn.Module (maps `(B, D)` -> `(B, C)`).
        writer_head: Optional auxiliary head for writer identification.
    """

    def __init__(
        self, 
        backbone: nn.Module, 
        head: nn.Module,
        writer_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.writer_head = writer_head

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images `(B, 3, H, W)`.
            
        Returns:
            If multi-task and training: tuple `(pen_logits, writer_logits)`.
            Otherwise: pen_logits `(B, NUM_CLASSES)`.
        """
        features = self.backbone(x)
        pen_logits = self.head(features)
        
        if self.writer_head is not None and self.training:
            writer_logits = self.writer_head(features)
            return pen_logits, writer_logits
        
        return pen_logits

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    model_name: str = "convnext_tiny",
    pretrained: bool = True,
    num_classes: int = 8,
    p_dropout: float = 0.3,
    num_writers: int = 0,
) -> PenClassificationModel:
    """Factory function to build the complete model.
    
    Args:
        model_name: Name of the `timm` backbone architecture.
        pretrained: Whether to load ImageNet pretrained weights.
        num_classes: Number of output pen classes.
        p_dropout: Dropout probability in the classifier head.
        num_writers: Number of unique writers. If > 0, adds auxiliary writer head.
        
    Returns:
        Instantiated PenClassificationModel.
    """
    backbone, feature_dim = create_backbone(model_name, pretrained=pretrained)
    head = PenClassifierHead(
        in_features=feature_dim,
        num_classes=num_classes,
        p_dropout=p_dropout
    )
    
    writer_head = None
    if num_writers > 0:
        writer_head = PenClassifierHead(
            in_features=feature_dim,
            num_classes=num_writers,
            p_dropout=p_dropout
        )
    
    return PenClassificationModel(backbone, head, writer_head=writer_head)
