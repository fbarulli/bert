from __future__ import annotations

import logging
import json
import math
import torch
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from simpler_fine_bert.wandb_manager import WandbManager
from simpler_fine_bert.cuda_utils import metrics_manager

logger = logging.getLogger(__name__)

class MetricsLogger:
    def __init__(
        self,
        metrics_dir: Optional[Path],
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional[WandbManager] = None,
        job_id: Optional[int] = None
    ):
        self.metrics_dir = metrics_dir
        self.is_trial = is_trial
        self.trial = trial
        self.wandb_manager = wandb_manager
        self.job_id = job_id
        self.epoch_metrics = []
        self.current_step = 0
        
        if metrics_dir:
            try:
                metrics_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create metrics directory: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], phase: str) -> None:
        """Log metrics for a single phase."""
        try:
            # Validate metrics based on phase
            if phase.startswith('train'):
                self._validate_metrics(metrics, is_training=True)
            elif phase.startswith('val'):
                self._validate_metrics(metrics, is_training=False)
            
            # Log to wandb if enabled
            if self.wandb_manager and self.wandb_manager.enabled:
                wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
                self.wandb_manager.log_metrics(wandb_metrics, step=self.current_step)
                
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        global_step: int
    ) -> None:
        """Log metrics for a complete epoch."""
        try:
            # Update current step
            self.current_step = global_step
            
            # Validate metrics
            self._validate_metrics(train_metrics, is_training=True)
            self._validate_metrics(val_metrics, is_training=False)

            # Log metrics
            self.log_metrics(train_metrics, f'train_epoch_{epoch}')
            self.log_metrics(val_metrics, f'val_epoch_{epoch}')

            # Store for plotting
            self.epoch_metrics.append({
                'epoch': epoch,
                'train': train_metrics,
                'validation': val_metrics,
                'step': global_step
            })

            # Log to progress bar
            desc = f"Epoch {epoch+1}"
            for k, v in val_metrics.items():
                desc += f" | {k}: {v:.4f}"
                
        except Exception as e:
            logger.error(f"Error logging epoch metrics: {str(e)}")
            logger.error(traceback.format_exc())

    def _validate_metrics(self, metrics: Dict[str, float], is_training: bool) -> None:
        """Validate metrics based on training phase."""
        try:
            # Handle transformers output objects
            processed_metrics = {}
            for k, v in metrics.items():
                if hasattr(v, 'loss'):  # Handle transformers output objects
                    processed_metrics['loss'] = v.loss.item() if isinstance(v.loss, torch.Tensor) else v.loss
                    if hasattr(v, 'perplexity'):
                        processed_metrics['perplexity'] = v.perplexity.item() if isinstance(v.perplexity, torch.Tensor) else v.perplexity
                    if hasattr(v, 'accuracy'):
                        processed_metrics['accuracy'] = v.accuracy.item() if isinstance(v.accuracy, torch.Tensor) else v.accuracy
                elif isinstance(v, torch.Tensor):
                    processed_metrics[k] = v.item()
                else:
                    processed_metrics[k] = v
                    
            metrics = processed_metrics
            
            # Check all values are numeric and valid
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Metric {metric} must be numeric, got {type(value)}")
                if math.isnan(value) or math.isinf(value):
                    logger.warning(f"Invalid value for metric {metric}: {value}, setting to 0.0")
                    metrics[metric] = 0.0
                    
            # Validate required metrics based on phase
            if is_training:
                required = {'loss', 'learning_rate'}
                missing = required - set(metrics.keys())
                if missing:
                    raise ValueError(f"Missing required training metrics: {missing}")
            else:
                # For validation, only loss is required
                if 'loss' not in metrics:
                    raise ValueError("Validation metrics must include 'loss'")
                
            # Validate metric ranges
            if 'accuracy' in metrics:
                if not 0 <= metrics['accuracy'] <= 1:
                    raise ValueError(f"Accuracy must be between 0 and 1, got {metrics['accuracy']}")
                    
            if 'loss' in metrics:
                if metrics['loss'] < 0:
                    raise ValueError(f"Loss must be non-negative, got {metrics['loss']}")
                    
            if 'perplexity' in metrics:
                if metrics['perplexity'] <= 0:
                    raise ValueError(f"Perplexity must be positive, got {metrics['perplexity']}")
                    
        except Exception as e:
            logger.error(f"Error validating metrics: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def flush(self) -> None:
        """Flush metrics to disk and wandb."""
        try:
            if self.metrics_dir:
                try:
                    epoch_file = self.metrics_dir / 'epoch_metrics.json'
                    with open(epoch_file, 'w') as f:
                        json.dump(self.epoch_metrics, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to save epoch metrics: {e}")
            
            if self.wandb_manager:
                try:
                    self.wandb_manager.flush()
                except Exception as e:
                    logger.warning(f"Failed to flush wandb: {e}")
                
        except Exception as e:
            logger.warning(f"Error flushing metrics: {e}")
