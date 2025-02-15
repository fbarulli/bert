from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from simpler_fine_bert.common.managers import get_wandb_manager
from simpler_fine_bert.common.utils import measure_memory, clear_memory

# Optional wandb support
try:
    WandbManager = get_wandb_manager().__class__
except ImportError:
    WandbManager = None

logger = logging.getLogger(__name__)

def train_embeddings(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    metrics_dir: Optional[str] = None,
    is_trial: bool = False,
    trial: Optional['optuna.Trial'] = None,
    wandb_manager: Optional[WandbManager] = None,
    job_id: Optional[int] = None,
    train_dataset: Optional['Dataset'] = None,
    val_dataset: Optional['Dataset'] = None
) -> None:
    """Train embedding model with masked language modeling."""
    try:
        from simpler_fine_bert.embedding.embedding_trainer import EmbeddingTrainer
        
        trainer = EmbeddingTrainer(
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
        
        # Train for specified number of epochs
        num_epochs = config['training']['num_epochs']
        trainer.train(num_epochs)
        
        # Plot best trial metrics
        from simpler_fine_bert.common.study.trial_analyzer import TrialAnalyzer
        analyzer = TrialAnalyzer(Path(metrics_dir) if metrics_dir else Path.cwd())
        analyzer.plot_trial_curves([trial] if trial else [], "Embedding Training")
        
        # Clean up resources
        trainer.cleanup_memory(aggressive=True)
        clear_memory()
        
    except Exception as e:
        logger.error(f"Error in embedding training: {str(e)}")
        raise

def validate_embeddings(
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: Dict[str, Any],
    metrics_dir: Optional[str] = None,
    wandb_manager: Optional[WandbManager] = None,
    job_id: Optional[int] = None,
    val_dataset: Optional['Dataset'] = None
) -> Dict[str, float]:
    """Validate embedding model on validation set."""
    try:
        from simpler_fine_bert.embedding.embedding_trainer import EmbeddingTrainer
        
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=None,  # No training loader needed for validation
            val_loader=val_loader,
            config=config,
            metrics_dir=metrics_dir,
            is_trial=False,
            trial=None,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=None,
            val_dataset=val_dataset
        )
        
        # Run validation
        val_metrics = trainer.validate()
        
        # Clean up resources
        trainer.cleanup_memory(aggressive=True)
        clear_memory()
        
        return val_metrics
        
    except Exception as e:
        logger.error(f"Error in embedding validation: {str(e)}")
        raise
