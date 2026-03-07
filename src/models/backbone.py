"""Feature extraction backbone module.

Wraps timm architectures (like ConvNeXt-Tiny) to output global pooled feature vectors
instead of classification logits.
"""
from typing import Tuple
import torch.nn as nn
import timm


def create_backbone(model_name: str = "convnext_tiny", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Create a feature extraction backbone.

    Uses `timm` to instantiate the model. The classifier head is completely
    removed (`num_classes=0`), so the forward pass returns a 2D tensor `(B, D)`
    of global pooled features.

    Args:
        model_name: Name of the `timm` model architecture.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        Tuple containing:
            - backbone: Feature extraction nn.Module
            - feature_dim: Dimension of the output feature vector `(D,)`.
    """
    # num_classes=0 strips the final linear layer and pooling might be adjusted
    # by timm to output just the pooled feature vector (B, D).
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
    )
    
    # Introspect the feature dimension directly from the timm model
    feature_dim = backbone.num_features
    
    return backbone, feature_dim
