from __future__ import annotations

import logging
import gc
import os
import traceback
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

from simpler_fine_bert.common.metrics_logger import MetricsLogger
from simpler_fine_bert.common.managers import get_storage_manager

# Get manager instance
storage_manager = get_storage_manager()

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        metrics_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_trial: bool = False,
        trial: Optional['optuna.Trial'] = None,
        wandb_manager: Optional['simpler_fine_bert.common.managers.wandb_manager.WandbManager'] = None,
        job_id: Optional[int] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None
    ):
        """Initialize trainer."""
        # Import managers at runtime
        from simpler_fine_bert.common.managers import (
            get_cuda_manager,
            get_batch_manager,
            get_amp_manager,
            get_tokenizer_manager,
            get_wandb_manager
        )

        # Optional wandb support
        try:
            wandb_manager_class = get_wandb_manager().__class__
        except ImportError:
            wandb_manager_class = None
        
        # Get manager instances
        cuda_manager = get_cuda_manager()
        batch_manager = get_batch_manager()
        amp_manager = get_amp_manager()
        tokenizer_manager = get_tokenizer_manager()
        
        # Store basic attributes
        self.model = model
        self.config = config
        self.metrics_dir = metrics_dir
        self.is_trial = is_trial
        self.trial = trial
        self.wandb_manager = wandb_manager
        self.job_id = job_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Store manager references
        self._cuda_manager = cuda_manager
        self._batch_manager = batch_manager
        self._amp_manager = amp_manager
        self._tokenizer_manager = tokenizer_manager
        
        # Initialize storage manager
        self.storage_manager = storage_manager
        
        # Get training config
        self.grad_norm = config['training']['max_grad_norm']
        self.gradient_accumulation = config['training']['gradient_accumulation_steps']
        self.log_interval = config['training']['logging_steps']
        self.eval_interval = config['training']['eval_steps']
        self.use_amp = config['training']['fp16']
        
        # Setup feature configurations
        profiler_config = config.get('training', {}).get('profiler', {})
        self.use_profiler = profiler_config.get('enabled', False)
        self.profiler_config = profiler_config
        
        cuda_graph_config = config.get('training', {}).get('cuda_graph', {})
        self.use_cuda_graph = cuda_graph_config.get('enabled', False)
        self.cuda_graph_config = cuda_graph_config
        self.static_batch = None
        self.static_graph = None
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device and model
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Initialize model weights if needed
        self._initialize_weights()
        
        # Create metrics logger
        self.metrics_logger = MetricsLogger(
            metrics_dir=metrics_dir,
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id
        )
        
        # Initialize training state
        self.current_step = 0
        self.accumulation_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.scheduler = scheduler
        self.pbar = None
        self.global_step = 0
        self.train_steps = len(train_loader)
        self.val_steps = len(val_loader)
        
        # Early stopping configuration
        self.early_stopping_triggered = False
        self.early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 3)
        self.early_stopping_min_delta = config.get('training', {}).get('early_stopping_min_delta', 1e-4)
        
        # Create CUDA graph if enabled
        if self.use_cuda_graph:
            self._cuda_manager.create_graph(self.forward_static)
            
        logger.info(
            f"Initialized trainer with:\n"
            f"- Device: {self.device}\n"
            f"- AMP: {self.use_amp}\n"
            f"- Gradient accumulation: {self.gradient_accumulation}\n"
            f"- CUDA graph: {self.use_cuda_graph}"
        )

    def _initialize_weights(self):
        """Initialize model weights properly."""
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
        self.model.apply(_init_weights)

    def create_optimizer(self, params=None) -> Optimizer:
        """Create optimizer with proper parameter initialization."""
        # Initialize optimizer with weight decay for non-bias parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer_type = self.config['training'].get('optimizer_type', 'adamw').lower()
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config['training']['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.config['training']['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config['training']['learning_rate'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
        logger.info(
            f"Created optimizer: {optimizer_type}\n"
            f"Learning rate: {self.config['training']['learning_rate']}\n"
            f"Weight decay: {self.config['training']['weight_decay']}"
        )
        
        return optimizer

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Union[Dict[str, torch.Tensor], 'transformers.modeling_outputs.ModelOutput'], Dict[str, torch.Tensor]]:
        """Forward pass with device management and error handling."""
        try:
            with record_function("batch_to_device"):
                device_batch = self._batch_manager.prepare_batch(batch, self.device)
            
            if self.use_cuda_graph:
                with record_function("cuda_graph_replay"):
                    self._cuda_manager.replay_graph()
                    return self.static_outputs, device_batch
                    
            with record_function("model_forward"), self._amp_manager.autocast():
                outputs = self.model(**device_batch)
                
                if not (hasattr(outputs, 'loss') or (isinstance(outputs, dict) and 'loss' in outputs)):
                    logger.error(f"Model outputs missing loss: {type(outputs)}")
                    if hasattr(outputs, '__dict__'):
                        logger.error(f"Available attributes: {outputs.__dict__.keys()}")
                    elif isinstance(outputs, dict):
                        logger.error(f"Available keys: {outputs.keys()}")
                    raise RuntimeError("Model outputs missing loss")
                
                if not isinstance(outputs, dict):
                    outputs = {
                        'loss': outputs.loss,
                        'logits': outputs.logits,
                        'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                        'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
                    }
                
                return outputs, device_batch
                
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            logger.error(f"Batch keys: {batch.keys()}")
            logger.error(f"Model type: {type(self.model)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """Backward pass with gradient accumulation."""
        with record_function("loss_scale"):
            loss = loss / self.gradient_accumulation
        
        with record_function("backward_step"):
            self._amp_manager.backward_step(
                loss=loss,
                model=self.model,
                optimizer=optimizer,
                grad_norm=self.grad_norm
            )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: Optimizer
    ) -> Dict[str, float]:
        """Single training step."""
        try:
            outputs, device_batch = self.forward(batch)
            
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")
            
            self.backward_step(loss, optimizer)
            
            self.accumulation_step = (self.accumulation_step + 1) % self.gradient_accumulation
            
            metrics = self.compute_metrics(outputs, device_batch)
            metrics['loss'] = loss.item() * self.gradient_accumulation
            
            current_lr = optimizer.param_groups[0]['lr']
            metrics['learning_rate'] = current_lr
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_metrics = []
        
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    outputs, device_batch = self.forward(batch)
                    metrics = self.compute_metrics(outputs, device_batch)
                    val_metrics.append(metrics)
            
            avg_metrics = {}
            for key in val_metrics[0].keys():
                values = [m[key] for m in val_metrics]
                avg_metrics[key] = sum(values) / len(values)
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise
        finally:
            self.model.train()

    def train_epoch(
        self,
        epoch: int,
        optimizer: Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch."""
        try:
            # Initialize validation metrics
            self.current_val_metrics = {'loss': float('inf'), 'acc': 0.0}
            
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/1 [Train]",
                dynamic_ncols=True
            )
            
            train_metrics = []
            profiler = self._create_profiler() if self.use_profiler else nullcontext()
            
            with profiler as prof:
                for batch in pbar:
                    with record_function("train_step"):
                        metrics = self.train_step(batch, optimizer)
                        train_metrics.append(metrics)
                        
                        # Format metrics for display
                        display_metrics = {
                            'loss': f"{metrics['loss']:.4f}",
                            'ppl': f"{metrics.get('ppl', float('inf')):.2f}",
                            'acc': f"{metrics.get('accuracy', 0):.2%}",  # Use accuracy from metrics_manager
                            'val_loss': f"{self.current_val_metrics['loss']:.4f}",
                            'val_acc': f"{self.current_val_metrics.get('accuracy', 0):.2%}"  # Use accuracy from metrics_manager
                        }
                        
                        pbar.set_postfix(display_metrics)
                        
                        if self.current_step % self.log_interval == 0:
                            self.log_metrics(metrics, 'train')
                        
                        if self.current_step % self.eval_interval == 0:
                            self.current_val_metrics = self.validate()
                            self.log_metrics(self.current_val_metrics, 'val')
                            
                            if self.current_val_metrics['loss'] < self.best_val_loss:
                                self.best_val_loss = self.current_val_metrics['loss']
                                if not self.is_trial:
                                    self.save_model()
                        
                        self.current_step += 1
                        
                        if self.use_profiler:
                            prof.step()
            
            if self.use_profiler:
                logger.info(prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=10
                ))
            
            avg_metrics = {}
            for key in train_metrics[0].keys():
                values = [m[key] for m in train_metrics]
                avg_metrics[key] = sum(values) / len(values)
            
            # Step scheduler at end of epoch
            if self.scheduler is not None:
                self.scheduler.step()
                avg_metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {str(e)}")
            raise

    def train(self, num_epochs: int) -> None:
        """Train for multiple epochs."""
        try:
            with record_function("train_setup"):
                # Use existing optimizer if created in __init__
                optimizer = getattr(self, '_optimizer', None)
                if optimizer is None:
                    optimizer = self.create_optimizer()
                # Use existing scheduler if created in __init__
                scheduler = getattr(self, 'scheduler', None)
                
                logger.info("Starting training with profiling enabled" if self.use_profiler else "Starting training")
            
            for epoch in range(num_epochs):
                with record_function(f"epoch_{epoch}"):
                    train_metrics = self.train_epoch(epoch, optimizer)
                    self.log_metrics(train_metrics, f'train_epoch_{epoch}')
                    
                    with record_function("validation"):
                        val_metrics = self.validate()
                        self.log_metrics(val_metrics, f'val_epoch_{epoch}')
                    
                    with record_function("model_saving"):
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                            if not self.is_trial:
                                self.save_model()
                    
                    # Step scheduler at end of epoch if it exists
                    if scheduler is not None:
                        scheduler.step()
                        logger.debug(f"Stepped scheduler, new LR: {scheduler.get_last_lr()[0]:.2e}")
                    
                    with record_function("memory_cleanup"):
                        self.cleanup_memory()
                        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.use_profiler:
                self._cleanup_profiler()
            
    def log_metrics(self, metrics: Dict[str, float], phase: str) -> None:
        """Log metrics for current phase."""
        try:
            self.metrics_logger.log_metrics(metrics, phase)
            
            if self.is_trial and self.trial and phase.startswith('val'):
                self.trial.report(metrics['loss'], step=self.current_step)
                
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
                    
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_profiler(self):
        """Create PyTorch profiler with configuration."""
        schedule_config = self.profiler_config.get('schedule', {})
        activities = []
        if 'cpu' in self.profiler_config.get('activities', []):
            activities.append(ProfilerActivity.CPU)
        if 'cuda' in self.profiler_config.get('activities', []):
            activities.append(ProfilerActivity.CUDA)

        # Get profiler directory from storage manager
        profiler_dir = self.storage_manager.get_profiler_dir(
            self.trial.number if self.is_trial else None
        )

        profiler = profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=schedule_config.get('wait', 2),
                warmup=schedule_config.get('warmup', 2),
                active=schedule_config.get('active', 6),
                repeat=schedule_config.get('repeat', 1)
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
            record_shapes=self.profiler_config.get('record_shapes', True),
            profile_memory=self.profiler_config.get('profile_memory', True),
            with_stack=self.profiler_config.get('with_stack', True),
            with_flops=self.profiler_config.get('with_flops', True)
        )

        if self.profiler_config.get('export_chrome_trace', False):
            def export_chrome_trace(prof):
                prof.export_chrome_trace(str(profiler_dir / 'trace.json'))
            profiler.on_trace_ready = export_chrome_trace

        return profiler

    def _cleanup_profiler(self):
        """Clean up profiler resources."""
        try:
            if hasattr(self, '_profiler') and self._profiler is not None:
                self._profiler.stop()
                del self._profiler
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            # Clean up profiler directory
            if self.is_trial:
                self.storage_manager.cleanup_profiler(self.trial.number)
            else:
                self.storage_manager.cleanup_profiler()
                
        except Exception as e:
            logger.error(f"Error cleaning up profiler: {str(e)}")
            logger.error(traceback.format_exc())

    def compute_metrics(
        self,
        outputs: Union[Dict[str, torch.Tensor], 'transformers.modeling_outputs.ModelOutput'],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics from outputs (deprecated - use metrics_manager directly)."""
        from simpler_fine_bert.common.managers import get_metrics_manager
        metrics_manager = get_metrics_manager()
        return metrics_manager.compute_embedding_metrics(outputs, batch)

    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    def save_model(self) -> None:
        """Save model."""
        if not self.is_trial:
            save_dir = self.metrics_dir.parent / 'model'
            save_dir.mkdir(exist_ok=True)
            
            # Save with safe_serialization=False to handle shared tensors
            self.model.save_pretrained(save_dir, safe_serialization=False)
            logger.info(f"Saved model to {save_dir}")

    def cleanup_memory(self, aggressive: bool = False) -> None:
        """Clean up memory resources."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            
            if aggressive and not hasattr(self, '_cleaned_up'):
                if hasattr(self, 'model') and self.model is not None:
                    self.model.cpu()
                
                if hasattr(self, 'train_loader'):
                    del self.train_loader
                if hasattr(self, 'val_loader'):
                    del self.val_loader
                
                if hasattr(self, 'train_dataset'):
                    del self.train_dataset
                if hasattr(self, 'val_dataset'):
                    del self.val_dataset
                
                if self.is_trial and self.trial:
                    self._tokenizer_manager.cleanup_worker(self.trial.number)
                
                gc.collect()
                self._cleaned_up = True
                
        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            logger.error(traceback.format_exc())

    def plot_metrics(self) -> None:
        """Plot training metrics (deprecated - use trial_analyzer.plot_trial_curves instead)."""
        pass

class nullcontext:
    """Context manager that does nothing."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
