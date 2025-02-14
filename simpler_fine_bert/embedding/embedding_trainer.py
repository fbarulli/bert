from __future__ import annotations

import logging
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from simpler_fine_bert.common.managers import get_metrics_manager

# Get manager instance
metrics_manager = get_metrics_manager()

logger = logging.getLogger(__name__)

# Import BaseTrainer at runtime to avoid circular import
def get_base_trainer():
    from simpler_fine_bert.common.base_trainer import BaseTrainer
    return BaseTrainer

class EmbeddingTrainer(get_base_trainer()):
    """Trainer for learning embeddings through masked language modeling."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        metrics_dir: Optional[str] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional['simpler_fine_bert.common.managers.wandb_manager.WandbManager'] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional['Dataset'] = None,
        val_dataset: Optional['Dataset'] = None
    ) -> None:
        """Initialize embedding trainer."""
        # Create metrics directory before initialization
        self.max_grad_norm = config['training']['max_grad_norm']
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
        self.best_embedding_loss = float('inf')
        # Create optimizer
        self._optimizer = self.create_optimizer()
        
        # Scale learning rate based on batch size
        base_batch_size = 32  # Standard BERT batch size
        current_batch_size = config['training']['batch_size']
        if current_batch_size != base_batch_size:
            scale_factor = current_batch_size / base_batch_size
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= scale_factor
            logger.info(f"Scaled learning rate by {scale_factor} for batch size {current_batch_size}")

    def compute_batch_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        device_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute embedding-specific metrics."""
        try:
            # Get embedding metrics from metrics manager
            metrics = metrics_manager.compute_embedding_metrics(outputs, batch)
            
            # Update best embedding loss if needed
            if metrics['embedding_loss'] < self.best_embedding_loss:
                self.best_embedding_loss = metrics['embedding_loss']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing embedding metrics: {str(e)}")
            logger.error(f"Output type: {type(outputs)}")
            logger.error(f"Output keys: {outputs.keys() if isinstance(outputs, dict) else 'Not a dict'}")
            logger.error(f"Batch type: {type(batch)}")
            logger.error(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'Not a dict'}")
            raise

    def get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self._optimizer.param_groups[0]['lr'] if self._optimizer else 0.0

    def cleanup_memory(self, aggressive: bool = False) -> None:
        """Clean up embedding-specific memory resources."""
        super().cleanup_memory(aggressive)
        if aggressive:
            self.best_embedding_loss = float('inf')
