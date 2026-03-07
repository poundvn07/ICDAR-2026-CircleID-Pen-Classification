"""Test suite for Model Architecture (src/models/).

TDD: These tests define the contract BEFORE implementation.
Validates backbone feature extraction, classifier head shape, and factory assembly.
"""
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants matching dataset
# ---------------------------------------------------------------------------
NUM_CLASSES = 8
DEFAULT_IMAGE_SIZE = 224
BATCH_SIZE = 4
IN_CHANNELS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """Return a dummy batch of images: (B, C, H, W)."""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Test: Backbone Contract
# ---------------------------------------------------------------------------
class TestBackbone:
    """Tests for the feature extraction backbone."""

    def test_backbone_output_shape(self, dummy_batch):
        """Backbone must output a 2D tensor of shape (B, D)."""
        from src.models.backbone import create_backbone

        # Try with a lightweight model for fast testing
        backbone, feature_dim = create_backbone("resnet18", pretrained=False)
        output = backbone(dummy_batch)

        assert output.ndim == 2, f"Expected 2D output (B, D), got {output.ndim}D"
        assert output.shape[0] == BATCH_SIZE
        assert output.shape[1] == feature_dim
        assert feature_dim == 512  # ResNet18 feature dim

    def test_convnext_tiny_feature_dim(self):
        """Our primary backbone (ConvNeXt-Tiny) must have dim=768."""
        from src.models.backbone import create_backbone

        _, feature_dim = create_backbone("convnext_tiny", pretrained=False)
        assert feature_dim == 768


# ---------------------------------------------------------------------------
# Test: Classifier Head Contract
# ---------------------------------------------------------------------------
class TestClassifierHead:
    """Tests for the custom classification head."""

    def test_head_output_shape(self):
        """Classifier head must map (B, D) to (B, NUM_CLASSES)."""
        from src.models.classifier import PenClassifierHead

        feature_dim = 768
        head = PenClassifierHead(in_features=feature_dim, num_classes=NUM_CLASSES)
        
        dummy_features = torch.randn(BATCH_SIZE, feature_dim)
        logits = head(dummy_features)

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_head_contains_layernorm_and_dropout(self):
        """Head must include LayerNorm and Dropout for robustness."""
        from src.models.classifier import PenClassifierHead

        head = PenClassifierHead(in_features=768, num_classes=8, p_dropout=0.3)
        
        has_ln = any(isinstance(m, nn.LayerNorm) for m in head.modules())
        has_dp = any(isinstance(m, nn.Dropout) for m in head.modules())
        
        assert has_ln, "Classifier head must include LayerNorm"
        assert has_dp, "Classifier head must include Dropout"


# ---------------------------------------------------------------------------
# Test: Factory / Full Model Assembly
# ---------------------------------------------------------------------------
class TestModelFactory:
    """Tests for the full model builder."""

    def test_full_model_forward(self, dummy_batch):
        """Full model must take images and return logits."""
        from src.models.factory import create_model

        model = create_model(model_name="resnet18", pretrained=False, num_classes=NUM_CLASSES)
        logits = model(dummy_batch)

        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_model_train_val_modes(self):
        """Model must properly set train/eval modes."""
        from src.models.factory import create_model

        model = create_model("resnet18", pretrained=False)
        
        model.train()
        assert model.training is True
        
        model.eval()
        assert model.training is False

    def test_freeze_backbone(self):
        """Model must be able to freeze the backbone for Stage 1 fine-tuning."""
        from src.models.factory import create_model

        model = create_model("resnet18", pretrained=False)
        model.freeze_backbone()

        for param in model.backbone.parameters():
            assert param.requires_grad is False
            
        for param in model.head.parameters():
            assert param.requires_grad is True

    def test_unfreeze_backbone(self):
        """Model must be able to unfreeze the backbone for Stage 2 fine-tuning."""
        from src.models.factory import create_model

        model = create_model("resnet18", pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()

        for param in model.backbone.parameters():
            assert param.requires_grad is True
