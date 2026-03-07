"""Trainer module.

Implements the training loop, validation loop, and tracking logic with
support for Automatic Mixed Precision (AMP).
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
    - Metric aggregation and logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Dict[str, Callable]] = None,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.scheduler_cfg = scheduler
        self.use_amp = use_amp
        
        # Determine AMP scaler based on device
        self.scaler = None
        if self.use_amp:
            if self.device.type == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda')
            elif self.device.type == 'mps':
                # MPS currently doesn't implement or recommend GradScaler,
                # as natively bfloat16 handles ranges well, but we set flag.
                pass 
                
        # History
        self.history = defaultdict(list)

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        
        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            # the device_type kwarg handles cuda/cpu/mps context targeting
            if self.use_amp and self.device.type in ('cuda', 'mps'):
                # Note: bfloat16 is preferred for MPS but mps doesn't support torch.autocast natively in the same way yet.
                # We use generic fallback:
                device_type = self.device.type
                dtype = torch.bfloat16 if device_type == 'mps' else torch .float16
                
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
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
            
            # Detach to free up VRAM
            all_preds.append(logits.detach())
            all_targets.append(labels.detach())
            
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
            
            if self.use_amp and self.device.type in ('cuda', 'mps'):
                device_type = self.device.type
                dtype = torch.bfloat16 if device_type == 'mps' else torch.float16
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            epoch_loss += loss.item()
            
            all_preds.append(logits.detach())
            all_targets.append(labels.detach())
            
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_preds, all_targets)
        metrics["loss"] = epoch_loss / len(dataloader)
        
        # Step the scheduler if it's epoch-based
        if self.scheduler_cfg and self.scheduler_cfg.get("interval") == "epoch":
            # Some schedulers like ReduceLROnPlateau need the validation loss
            scheduler = self.scheduler_cfg["scheduler"]
            if hasattr(scheduler, "step_with_metrics"):
               pass # Not standard OneCycle
               
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
                
            # Log latest LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            print(f"LR: {current_lr:.6e}")
            
            # Keep track of best model loosely
            if val_metrics["macro_f1"] > best_val_f1:
                print(f"🌟 New best validation F1: {val_metrics['macro_f1']:.4f} (improved from {best_val_f1:.4f})")
                best_val_f1 = val_metrics["macro_f1"]
                # A full save_checkpoint method would go here
            
        return dict(self.history)
