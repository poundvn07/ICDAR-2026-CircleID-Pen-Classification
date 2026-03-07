"""Model factory module.

Combines the backbone and the classification head into a cohesive model,
and provides utilities for freezing/unfreezing layers for transfer learning.
"""
import torch
import torch.nn as nn

from src.models.backbone import create_backbone
from src.models.classifier import PenClassifierHead


class PenClassificationModel(nn.Module):
    """End-to-End model for Pen Classification.
    
    Args:
        backbone: Feature extraction nn.Module (output shape: `(B, D)`).
        head: Classification head nn.Module (maps `(B, D)` -> `(B, C)`).
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images `(B, 3, H, W)`.
            
        Returns:
            Class logits `(B, NUM_CLASSES)`.
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone.
        
        Used for Stage 1 of transfer learning (warmup classifier head).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the backbone.
        
        Used for Stage 2 of transfer learning (fine-tune everything).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    model_name: str = "convnext_tiny",
    pretrained: bool = True,
    num_classes: int = 8,
    p_dropout: float = 0.3
) -> PenClassificationModel:
    """Factory function to build the complete model.
    
    Args:
        model_name: Name of the `timm` backbone architecture.
        pretrained: Whether to load ImageNet pretrained weights.
        num_classes: Number of output classes.
        p_dropout: Dropout probability in the classifier head.
        
    Returns:
        Instantiated PenClassificationModel.
    """
    backbone, feature_dim = create_backbone(model_name, pretrained=pretrained)
    head = PenClassifierHead(
        in_features=feature_dim,
        num_classes=num_classes,
        p_dropout=p_dropout
    )
    
    return PenClassificationModel(backbone, head)
