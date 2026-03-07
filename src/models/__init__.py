"""src.models — Pen Classification Architectures

Public API
----------
- create_model: Factory function for the complete E2E model
- PenClassificationModel: E2E Model class with freeze/unfreeze methods
- create_backbone: Factory for timm feature extractors
- PenClassifierHead: Robust classification head
"""

from src.models.backbone import create_backbone
from src.models.classifier import PenClassifierHead
from src.models.factory import create_model, PenClassificationModel

__all__ = [
    "create_model",
    "PenClassificationModel",
    "create_backbone",
    "PenClassifierHead",
]
