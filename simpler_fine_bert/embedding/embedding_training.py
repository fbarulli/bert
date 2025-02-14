from __future__ import annotations

import logging
import os
import torch
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transformers import BertConfig

from simpler_fine_bert.embedding.embedding_trainer import EmbeddingTrainer
from simpler_fine_bert.common.managers import get_resource_manager

# Get manager instance
resource_manager = get_resource_manager()
from simpler_fine_bert.common.wandb_manager import WandbManager
from simpler_fine_bert.common.utils import measure_memory, clear_memory
from simpler_fine_bert.common.resource import resource_factory

logger = logging.getLogger(__name__)

@measure_memory
def train_embeddings(
    config: Dict[str, Any],
    output_dir: Path,
    is_trial: bool = False,
    trial: Optional['optuna.Trial'] = None,
    wandb_manager: Optional[WandbManager] = None,
    job_id: Optional[int] = None
) -> Tuple[float, Dict[str, Any]]:
    """Train a BERT model to learn embeddings through masked token prediction.
    
    Args:
        config: Training configuration
        output_dir: Directory to save outputs
        is_trial: Whether this is an Optuna trial
        trial: Optuna trial object if is_trial=True
        wandb_manager: Optional W&B logging manager
        job_id: Optional job ID for parallel training
        
    Returns:
        Tuple of (best validation loss, metrics dictionary)
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create resources through factory
        try:
            train_dataset, val_dataset = resource_manager.create_datasets(
                config,
                stage='embedding'
            )
            train_loader, val_loader = resource_manager.create_dataloaders(
                config,
                train_dataset,
                val_dataset
            )
            
            model = resource_factory.create_resource('model', config)
            
            logger.info("Successfully created all resources")
        except Exception as e:
            logger.error(f"Failed to create resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialize trainer
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=output_dir / 'metrics',
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        try:
            # Train model
            trainer.train(config['training']['num_epochs'])
            
            # Get best metrics
            metrics = {
                'best_embedding_loss': trainer.best_embedding_loss,
                'final_learning_rate': trainer.get_current_lr()
            }
            
            return trainer.best_embedding_loss, metrics
            
        finally:
            clear_memory()
        
    except Exception as e:
        logger.error(
            f"Embedding training failed:\n"
            f"Error: {str(e)}\n"
            f"Config: {config}\n"
            f"Trial: {trial.number if trial else None}"
        )
        if trial and hasattr(trial, 'set_user_attr'):
            trial.set_user_attr('error', str(e))
        raise
    finally:
        clear_memory()

@measure_memory
def validate_embeddings(
    model_path: str,
    data_path: str,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Validate a trained embedding model.
    
    Args:
        model_path: Path to saved model
        data_path: Path to validation data
        config: Model configuration
        output_dir: Optional output directory for metrics
        
    Returns:
        Dictionary of validation metrics
    """
    try:
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with validation data path
        val_config = {**config, 'data': {**config['data'], 'csv_path': data_path}}
        
        # Create resources through factory
        try:
            _, val_dataset = resource_manager.create_datasets(
                val_config,
                stage='embedding'
            )
            _, val_loader = resource_manager.create_dataloaders(
                val_config,
                None,
                val_dataset
            )
            
            # For validation, we need to load a specific model checkpoint
            val_config['model']['path'] = model_path
            model = resource_factory.create_resource('model', val_config)
            
            logger.info("Successfully created validation resources")
        except Exception as e:
            logger.error(f"Failed to create validation resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialize trainer
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=None,
            val_loader=val_loader,
            config=config,
            metrics_dir=output_dir / 'metrics' if output_dir else None,
            is_trial=False,
            val_dataset=val_dataset
        )
        
        try:
            # Run validation
            metrics = trainer.validate()
            
            # Log results
            logger.info(
                f"Validation results:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- Embedding Loss: {metrics['embedding_loss']:.4f}\n"
                f"- Perplexity: {metrics['perplexity']:.2f}\n"
                f"- Accuracy: {metrics['accuracy']:.4f}\n"
                f"- Top-5 Accuracy: {metrics['top5_accuracy']:.4f}"
            )
            
            return metrics
            
        finally:
            clear_memory()
        
    except Exception as e:
        logger.error(
            f"Embedding validation failed:\n"
            f"Error: {str(e)}\n"
            f"Model path: {model_path}\n"
            f"Data path: {data_path}"
        )
        raise
    finally:
        clear_memory()
