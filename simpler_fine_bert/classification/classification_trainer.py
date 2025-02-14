from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional, Union
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import optuna

from simpler_fine_bert.common.base_trainer import BaseTrainer
from simpler_fine_bert.common.cuda_utils import metrics_manager
from simpler_fine_bert.common.wandb_manager import WandbManager

logger = logging.getLogger(__name__)

class ClassificationTrainer(BaseTrainer):
    """BERT trainer with classification-specific functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        metrics_dir: Path,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_trial: bool = False,
        trial: Optional[optuna.Trial] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None
    ) -> None:
        """Initialize classification trainer.
        
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
            metrics_dir: Directory to save metrics
            optimizer: Optional pre-configured optimizer
            scheduler: Optional learning rate scheduler
            is_trial: Whether this is an Optuna trial
            trial: Optuna trial object if is_trial is True
            wandb_manager: Optional Weights & Biases manager
            job_id: Optional job identifier
            train_dataset: Optional training dataset
            val_dataset: Optional validation dataset
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            scheduler=scheduler,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        self._optimizer = optimizer
        self.best_accuracy = 0.0
    
    def create_optimizer(self) -> Optimizer:
        """Create or return optimizer."""
        if self._optimizer is not None:
            return self._optimizer
        return super().create_optimizer()
    
    def compute_task_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute classification-specific metrics.
        
        Args:
            outputs: Model outputs (on device)
            batch: Input batch (already moved to device)
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = metrics_manager.compute_classification_metrics(outputs, batch)
        if metrics.get('accuracy', 0.0) > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
        return metrics
