"""Trainer module.

Implements the training loop, validation loop, and tracking logic with
support for Automatic Mixed Precision (AMP) and Multi-task Learning.
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable
from collections import defaultdict
import numpy as np

from src.training.metrics import compute_metrics


class Trainer:
    """Handles the model training and validation loops.
    
    Features:
    - Train / Eval epochs
    - Automatic Mixed Precision (AMP) via `torch.amp`
    - Multi-task Learning support (pen + writer classification)
    - Metric aggregation and logging
    - Best model checkpoint saving
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Dict[str, Callable]] = None,
        use_amp: bool = True,
        save_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.scheduler_cfg = scheduler
        self.use_amp = use_amp
        self.save_path = save_path
        
        # Detect multi-task mode
        self.multi_task = hasattr(model, 'writer_head') and model.writer_head is not None
        
        # Determine AMP scaler based on device
        self.scaler = None
        if self.use_amp:
            if self.device.type == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda')
                
        # History
        self.history = defaultdict(list)

    def _get_amp_context(self):
        """Return the appropriate autocast context."""
        if self.use_amp and self.device.type in ('cuda', 'mps'):
            device_type = self.device.type
            dtype = torch.bfloat16 if device_type == 'mps' else torch.float16
            return torch.autocast(device_type=device_type, dtype=dtype)
        # Fallback: no-op context
        import contextlib
        return contextlib.nullcontext()
    
    def _compute_loss(self, model_output, batch):
        """Compute loss for single or multi-task mode."""
        labels = batch["label"].to(self.device)
        
        if self.multi_task and isinstance(model_output, tuple):
            pen_logits, writer_logits = model_output
            writer_labels = batch["writer_label"].to(self.device)
            loss = self.criterion(pen_logits, labels, writer_logits, writer_labels)
            return loss, pen_logits
        else:
            # Single-task or eval mode (model returns only pen_logits)
            pen_logits = model_output
            if self.multi_task:
                # MultiTaskLoss handles missing writer args gracefully
                loss = self.criterion(pen_logits, labels)
            else:
                loss = self.criterion(pen_logits, labels)
            return loss, pen_logits

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with self._get_amp_context():
                model_output = self.model(images)
                loss, pen_logits = self._compute_loss(model_output, batch)
                
            # Backward and Optimize
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            # Step the scheduler if it's step-based
            if self.scheduler_cfg and self.scheduler_cfg.get("interval") == "step":
                self.scheduler_cfg["scheduler"].step()
                
            epoch_loss += loss.item()
            all_preds.append(pen_logits.detach())
            all_targets.append(batch["label"].detach())
            
        # compute epoch metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_preds, all_targets)
        metrics["loss"] = epoch_loss / len(dataloader)
        
        return metrics

    @torch.no_grad()
    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            with self._get_amp_context():
                pen_logits = self.model(images)  # eval mode -> only pen_logits
                loss = self.criterion(pen_logits, labels) if not self.multi_task else \
                       self.criterion.pen_criterion(pen_logits, labels)
                
            epoch_loss += loss.item()
            all_preds.append(pen_logits.detach())
            all_targets.append(labels.detach())
            
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_preds, all_targets)
        metrics["loss"] = epoch_loss / len(dataloader)
        
        return metrics

    def fit(self, train_dl: DataLoader, val_dl: DataLoader, epochs: int) -> Dict[str, list]:
        """Train the model for a specified number of epochs.
        
        Args:
            train_dl: Training DataLoader.
            val_dl: Validation DataLoader.
            epochs: Total epochs to train.
            
        Returns:
            Dictionary containing the history of metrics.
        """
        best_val_f1 = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\n[{'-'*10} Epoch {epoch}/{epochs} {'-'*10}]")
            
            # Train
            train_metrics = self._train_epoch(train_dl)
            print(f"Train | Loss: {train_metrics['loss']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f} | "
                  f"F1: {train_metrics['macro_f1']:.4f}")
            
            # Validate
            val_metrics = self._validate_epoch(val_dl)
            print(f"Val   | Loss: {val_metrics['loss']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f} | "
                  f"F1: {val_metrics['macro_f1']:.4f}")
            
            # Track history
            for k, v in train_metrics.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)
                
            # Log LR info
            lrs = [f"{pg.get('name', f'g{i}')}: {pg['lr']:.2e}" 
                   for i, pg in enumerate(self.optimizer.param_groups)]
            self.history["lr"].append(self.optimizer.param_groups[-1]["lr"])
            print(f"LR: {' | '.join(lrs)}")
            
            # Save best model
            if val_metrics["macro_f1"] > best_val_f1:
                print(f"🌟 New best F1: {val_metrics['macro_f1']:.4f} (prev: {best_val_f1:.4f})")
                best_val_f1 = val_metrics["macro_f1"]
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"   Saved best model to {self.save_path}")
            
        return dict(self.history)
