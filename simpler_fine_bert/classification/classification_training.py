from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import math

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
from simpler_fine_bert.classification.dataset import CSVDataset
from simpler_fine_bert.common.optuna_manager import OptunaManager
from simpler_fine_bert.common.utils import seed_everything as set_seed, create_optimizer, create_scheduler

from simpler_fine_bert.common.config_utils import load_config

from simpler_fine_bert.common.cuda_utils import cuda_manager
from simpler_fine_bert.common.data_manager import dataloader_manager
from simpler_fine_bert.common.resource_manager import resource_manager
from simpler_fine_bert.classification.classification_trainer import ClassificationTrainer

def get_classification_model():
    """Get ClassificationBert model at runtime to avoid circular imports."""
    from simpler_fine_bert.classification.model import ClassificationBert
    return ClassificationBert

logger = logging.getLogger(__name__)

def configure_device():
    """Get device through cuda_manager."""
    return cuda_manager.get_device()

def get_texts_and_labels(config: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """Loads labels from CSV files using CSVDataset."""
    # Get tokenizer through manager
    tokenizer = tokenizer_manager.get_worker_tokenizer(
        worker_id=os.getpid(),
        model_name=config['model']['name'],
        model_type='classification'
    )
    
    train_dataset, val_dataset = resource_manager.create_datasets(
        config,
        stage='classification'
    )
    train_labels = [train_dataset[i]['label'] for i in range(len(train_dataset)) if 'label' in train_dataset[i]]
    val_labels = [val_dataset[i]['label'] for i in range(len(val_dataset)) if 'label' in val_dataset[i]]

    return train_labels, val_labels

def run_classification_optimization(embedding_model_path: str, config_path: str, study_name: Optional[str] = None) -> Dict[str, Any]:
    """Run classification optimization stage."""
    config = load_config(config_path)
    n_jobs = config['training']['n_jobs']  # Get n_jobs from config
    n_trials = config['training']['num_trials']  # Get num_trials from config
    study_name = study_name or 'classification_optimization'  # Use default name if not provided

    # Initialize output directory
    output_dir = Path(config['output']['dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_device()

    # Load labels
    logger.info("Loading labels for classification training")
    train_labels, val_labels = get_texts_and_labels(config)

    # Setup optimization manager
    study_manager = OptunaManager(
        study_name=study_name,
        config=config,
        storage_dir=output_dir
    )

    def objective(trial):
        # Initialize all variables to None at start
        local_vars = {
            'model': None,
            'tokenizer': None,
            'optimizer': None,
            'scheduler': None,
            'trainer': None,
            'train_loader': None,
            'val_loader': None
        }
        
        try:
            trial_config = study_manager.suggest_parameters(trial)
            trial_config['model']['name'] = embedding_model_path
            
            # Initialize components
            local_vars['tokenizer'] = tokenizer_manager.get_worker_tokenizer(
            worker_id=trial.number,
            model_name=embedding_model_path,
            model_type='classification'
            )
            local_vars['model'] = get_classification_model()(
                config=trial_config, 
                num_labels=trial_config['model']['num_labels']
            ).to(device)
            
            # Create datasets and loaders
            train_dataset, val_dataset = resource_manager.create_datasets(
                trial_config,
                stage='classification'
            )
            train_loader, val_loader = resource_manager.create_dataloaders(
                trial_config,
                train_dataset,
                val_dataset
            )
            local_vars['train_loader'] = train_loader
            local_vars['val_loader'] = val_loader

            # Create optimizer and scheduler
            local_vars['optimizer'] = create_optimizer(local_vars['model'], trial_config)
            local_vars['scheduler'] = create_scheduler(local_vars['optimizer'], len(local_vars['train_loader'].dataset), trial_config)

            # Initialize trainer
            local_vars['trainer'] = ClassificationTrainer(
                model=local_vars['model'],
                train_loader=local_vars['train_loader'],
                val_loader=local_vars['val_loader'],
                optimizer=local_vars['optimizer'],
                scheduler=local_vars['scheduler'],
                device=device,
                config=trial_config,
                checkpoints_dir=output_dir / 'classification_stage',
                is_trial=True,
                trial=trial,
                wandb_manager=study_manager.wandb_manager,
                job_id=trial.number
            )

            try:
                # Train and get validation metrics
                local_vars['trainer'].train(int(trial_config['training']['num_epochs']))
                val_metrics = local_vars['trainer'].validate()
                
                # Use a combination of metrics for optimization
                val_loss = val_metrics['loss']
                val_accuracy = val_metrics['accuracy']
                val_mse = val_metrics['mse']
                val_mae = val_metrics['mae']
                
                # Weighted combination of metrics
                objective_value = (
                    0.4 * val_loss +  # Weight loss more heavily
                    0.3 * (1 - val_accuracy) +  # Convert accuracy to error
                    0.15 * val_mse +  # Include MSE with lower weight
                    0.15 * val_mae  # Include MAE with lower weight
                )
                
                # Early stopping check
                if local_vars['trainer'].early_stopping_triggered:
                    raise optuna.TrialPruned("Early stopping triggered")
                
                # Bad loss check
                if not math.isfinite(val_metrics['loss']):
                    raise optuna.TrialPruned("Non-finite loss detected")
                
                # Memory check
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                    if memory_used > config['resources']['max_memory_gb']:
                        raise optuna.TrialPruned(f"Memory limit exceeded: {memory_used:.2f}GB")
                
                return objective_value
                
            finally:
                # Clean up GPU memory after each trial
                if torch.cuda.is_available():
                    for var_name, var in local_vars.items():
                        if var is not None:
                            del var
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleaned after classification trial")

        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise optuna.TrialPruned(f"Trial failed: {str(e)}")
        finally:
            # Safe cleanup
            if torch.cuda.is_available():
                for var_name, var in local_vars.items():
                    if var is not None:
                        del var
                torch.cuda.empty_cache()
                logger.info("Cleaned up trial resources")

    # Run optimization
    best_trial = study_manager.optimize(objective)
    logger.info(f"Best classification trial parameters: {best_trial.params}")
    
    return best_trial.params

def train_final_model(embedding_model_path: str, best_params: Dict[str, Any], config_path: str, output_dir: Optional[Path] = None) -> None:
    """Train final classification model with best parameters."""
    config = load_config(config_path)
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = configure_device()
    set_seed(config['training']['seed'])
    
    # Update config with best parameters and embedding model path
    config.update(best_params)
    config['model']['name'] = embedding_model_path
    config['training']['batch_size'] = 16  # Fixed batch size for classification
    
    # Initialize tokenizer and model
    tokenizer = tokenizer_manager.get_worker_tokenizer(
            worker_id=os.getpid(),
            model_name=embedding_model_path,
            model_type='classification'
    )
    model = get_classification_model()(config=config, num_labels=config['model']['num_labels']).to(device)

    # Create datasets and loaders
    train_dataset, val_dataset = resource_manager.create_datasets(
        config,
        stage='classification'
    )
    train_loader, val_loader = resource_manager.create_dataloaders(
        config,
        train_dataset,
        val_dataset
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, len(train_loader), config)

    # Initialize trainer
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        checkpoints_dir=output_dir / 'classification_stage',
        is_trial=False,
        trial=None,
        wandb_manager=None,
        job_id=0  # Final training uses job_id 0
    )

    try:
        # Train model
        trainer.train(int(config['training']['num_epochs']))
        
        # Get final validation metrics
        val_metrics = trainer.validate()
        logger.info("Final classification metrics:")
        logger.info(f"- Loss: {val_metrics['loss']:.4f}")
        logger.info(f"- Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"- MSE: {val_metrics['mse']:.4f}")
        logger.info(f"- MAE: {val_metrics['mae']:.4f}")
        
    finally:
        # Clean up GPU memory after training
        if torch.cuda.is_available():
            del model
            del optimizer
            del scheduler
            del trainer
            del tokenizer
            torch.cuda.empty_cache()
            logger.info("GPU memory cleaned after final training")
