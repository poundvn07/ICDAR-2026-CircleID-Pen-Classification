"""Evaluation metrics module.

Computes Top-1 Accuracy and Macro F1 score for the pen classification task.
"""
from typing import Dict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics for the batch/epoch.
    
    Args:
        preds: Logits or probabilities tensor of shape `(N, NUM_CLASSES)`.
        targets: Ground truth labels tensor of shape `(N,)`.
        
    Returns:
        Dictionary containing `accuracy` and `macro_f1` floats.
    """
    # Detach from graph, ensure supported dtype for numpy (float32 instead of bfloat16), move to CPU numpy
    preds_np = preds.detach().to(torch.float32).cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Get Top-1 predictions
    if preds_np.ndim == 2:
        pred_labels = np.argmax(preds_np, axis=1)
    else:
        pred_labels = preds_np

    acc = accuracy_score(targets_np, pred_labels)
    
    # Macro F1 is the secondary metric per OpenSpec, handles class imbalance well
    f1 = f1_score(targets_np, pred_labels, average="macro", zero_division=0)
    
    return {
        "accuracy": acc,
        "macro_f1": f1
    }
