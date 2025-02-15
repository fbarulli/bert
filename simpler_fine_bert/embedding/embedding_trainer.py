from __future__ import annotations

import logging
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from simpler_fine_bert.common.managers import get_metrics_manager

# Get manager instance
metrics_manager = get_metrics_manager()

# Optional wandb support
try:
    from simpler_fine_bert.common.managers import get_wandb_manager
    WandbManager = get_wandb_manager().__class__
except ImportError:
    WandbManager = None

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
        wandb_manager: Optional[WandbManager] = None,
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
        self.best_val_acc = 0.0
        # Create optimizer with scaled learning rate
        base_batch_size = 32  # Standard BERT batch size
        current_batch_size = config['training']['batch_size']
        effective_batch_size = current_batch_size * config['training']['gradient_accumulation_steps']
        
        # Use linear scaling for learning rate (BERT paper)
        if effective_batch_size != base_batch_size:
            scale_factor = effective_batch_size / base_batch_size
            config['training']['learning_rate'] *= scale_factor
            logger.info(
                f"Scaled learning rate by {scale_factor:.3f} "
                f"(batch_size={current_batch_size}, "
                f"grad_accum={config['training']['gradient_accumulation_steps']}, "
                f"effective_batch={effective_batch_size})"
            )
        
        # Create optimizer
        self._optimizer = self.create_optimizer()
        
        # Create scheduler
        if config['training'].get('scheduler', {}).get('use_scheduler', False):
            num_training_steps = len(train_loader) * config['training']['num_epochs']
            num_warmup_steps = int(num_training_steps * config['training']['scheduler']['warmup_ratio'])
            
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(
                f"Created linear scheduler with warmup "
                f"(warmup_steps={num_warmup_steps}, "
                f"total_steps={num_training_steps})"
            )

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics and track best values."""
        metrics = super().compute_metrics(outputs, batch)
        
        # Update best metrics
        if metrics['embedding_loss'] < self.best_embedding_loss:
            self.best_embedding_loss = metrics['embedding_loss']
        if metrics['accuracy'] > self.best_val_acc:
            self.best_val_acc = metrics['accuracy']
            if self.trial:
                self.trial.set_user_attr('best_val_acc', self.best_val_acc)
                self.trial.set_user_attr('epoch_metrics', self.metrics_logger.epoch_metrics)
        
        # Log metrics for debugging
        logger.debug(
            f"Batch Metrics:\n"
            f"- Loss: {metrics['loss']:.4f}\n"
            f"- PPL: {metrics.get('ppl', float('inf')):.2f}\n"
            f"- Accuracy: {metrics['accuracy']:.4%}\n"
            f"- Top-5 Accuracy: {metrics['top5_accuracy']:.4%}"
        )
        
        return metrics

    def get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self._optimizer.param_groups[0]['lr'] if self._optimizer else 0.0

    def cleanup_memory(self, aggressive: bool = False) -> None:
        """Clean up embedding-specific memory resources."""
        super().cleanup_memory(aggressive)
        if aggressive:
            self.best_embedding_loss = float('inf')
            self.best_val_acc = 0.0
