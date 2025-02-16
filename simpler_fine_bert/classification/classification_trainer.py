from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from simpler_fine_bert.common import get_resource_manager, get_wandb_manager
from simpler_fine_bert.common.managers.metrics_manager import metrics_manager

# Get manager instances
resource_manager = get_resource_manager()

# Optional wandb support
try:
    WandbManager = get_wandb_manager().__class__
except ImportError:
    WandbManager = None

logger = logging.getLogger(__name__)

# Import BaseTrainer at runtime to avoid circular import
def get_base_trainer():
    from simpler_fine_bert.common.base_trainer import BaseTrainer
    return BaseTrainer

class ClassificationTrainer(get_base_trainer()):
    """Trainer for fine-tuning BERT for classification tasks."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        metrics_dir: Optional[str] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional['Dataset'] = None,
        val_dataset: Optional['Dataset'] = None
    ) -> None:
        """Initialize classification trainer."""
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        self.best_accuracy = 0.0
        self._optimizer = self.create_optimizer()

    def compute_batch_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        device_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute classification-specific metrics."""
        try:
            # Get classification metrics from metrics manager
            metrics = metrics_manager.compute_classification_metrics(outputs, batch)
            
            # Update best accuracy if needed
            if metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = metrics['accuracy']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing classification metrics: {str(e)}")
            logger.error(f"Output type: {type(outputs)}")
            logger.error(f"Output keys: {outputs.keys() if isinstance(outputs, dict) else 'Not a dict'}")
            logger.error(f"Batch type: {type(batch)}")
            logger.error(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dict'}")
            raise

    def get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self._optimizer.param_groups[0]['lr'] if self._optimizer else 0.0

    def cleanup_memory(self, aggressive: bool = False) -> None:
        """Clean up classification-specific memory resources."""
        super().cleanup_memory(aggressive)
        if aggressive:
            self.best_accuracy = 0.0
