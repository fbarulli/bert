"""Factory for creating Optuna objectives."""

from __future__ import annotations

import logging
import os
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import optuna
from torch.utils.data import DataLoader, Dataset
from optuna.trial import FixedTrial
from torch.nn import Module
from torch.optim import Optimizer

from simpler_fine_bert.common.cuda_utils import cuda_manager
from simpler_fine_bert.common.dataloader_manager import dataloader_manager
from simpler_fine_bert.common.model_manager import model_manager
from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
from simpler_fine_bert.embedding import EmbeddingDataset, EmbeddingTrainer
from simpler_fine_bert.common.utils import create_optimizer

logger = logging.getLogger(__name__)

class ObjectiveFactory:
    """Factory for creating and managing Optuna objectives."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str) -> None:
        """Initialize the factory.
        
        Args:
            config: Training configuration
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.output_dir = output_dir
        self.pid = os.getpid()
        logger.info(f"ObjectiveFactory initialized for process {self.pid}")

    def objective_method(self, trial: optuna.Trial) -> float:
        """Process-local objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Best embedding loss achieved during training
            
        Raises:
            optuna.TrialPruned: If trial fails or should be pruned
        """
        local_vars: Dict[str, Optional[Union[Module, Optimizer, EmbeddingTrainer]]] = {
            'model': None,
            'optimizer': None,
            'trainer': None
        }
        current_pid = os.getpid()

        try:
            # Initialize all process resources
            from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
            ResourceInitializer.initialize_process()
            device = cuda_manager.get_device()  # Now safe to use after initialization
            logger.info(f"Running trial {trial.number} on {device} in process {current_pid}")
            
            # Get tokenizer through manager
            tokenizer = tokenizer_manager.get_worker_tokenizer(
                worker_id=trial.number,
                model_name=self.config['model']['name']
            )

            # Get trial configuration from parameter manager
            from simpler_fine_bert.common.parameter_manager import ParameterManager
            param_manager = ParameterManager(self.config)
            trial_config = param_manager.get_trial_config(trial)

            # Initialize datasets with process-specific error handling
            logger.info(f"Creating datasets in process {current_pid}")
            train_dataset = EmbeddingDataset(
                data_path=Path(trial_config['data']['csv_path']),
                tokenizer=tokenizer,
                max_length=trial_config['data']['max_length'],
                split='train',
                train_ratio=trial_config['data']['train_ratio'],
                mask_prob=trial_config['data']['embedding_mask_probability'],
                max_predictions=trial_config['data']['max_predictions'],
                max_span_length=trial_config['data']['max_span_length']
            )
            val_dataset = EmbeddingDataset(
                data_path=Path(trial_config['data']['csv_path']),
                tokenizer=tokenizer,
                max_length=trial_config['data']['max_length'],
                split='val',
                train_ratio=trial_config['data']['train_ratio'],
                mask_prob=trial_config['data']['embedding_mask_probability'],
                max_predictions=trial_config['data']['max_predictions'],
                max_span_length=trial_config['data']['max_span_length']
            )
            logger.info(f"Datasets created successfully in process {current_pid}")

            batch_size = trial_config['training']['batch_size']
            train_loader = dataloader_manager.create_dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=trial_config['data']['num_workers']
            )
            val_loader = dataloader_manager.create_dataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=trial_config['data']['num_workers']
            )

            # Create model and optimizer
            model = model_manager.get_worker_model(
                worker_id=trial.number,
                model_name=trial_config['model']['name'],
                device_id=device.index if device.type == 'cuda' else None,
                config=trial_config
            )
            optimizer = create_optimizer(model, trial_config['training'])
            
            # Store in local vars for cleanup
            local_vars['model'] = model
            local_vars['optimizer'] = optimizer
            
            logger.info(f"Created model and optimizer in process {current_pid}")
            
            # Create trainer with pre-configured optimizer
            local_vars['trainer'] = EmbeddingTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=trial_config,
                metrics_dir=Path(self.output_dir) / f'trial_{trial.number}',
                optimizer=optimizer,
                is_trial=True,
                trial=trial,
                wandb_manager=None,
                job_id=trial.number,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            local_vars['trainer'].train(trial_config['training']['num_epochs'])
            return local_vars['trainer'].best_embedding_loss

        except Exception as e:
            logger.error(f"Trial failed in process {current_pid}: {str(e)}")
            raise optuna.TrialPruned()
        finally:
            # Thorough cleanup of process-local resources
            try:
                # Clean up local variables first
                for var in local_vars.values():
                    if var is not None:
                        del var
                
                # Clean up model and tokenizer manager resources
                model_manager.cleanup_worker(trial.number)
                tokenizer_manager.cleanup_worker(trial.number)
                
                # Clean up all process resources in proper order
                from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
                ResourceInitializer.cleanup_process()
                
                logger.info(f"Cleaned up resources for trial {trial.number} in process {current_pid}")
            except Exception as e:
                logger.error(f"Error during cleanup in process {current_pid}: {str(e)}")
                # Attempt resource cleanup even if other cleanup fails
                try:
                    ResourceInitializer.cleanup_process()
                except Exception as cleanup_error:
                    logger.error(f"Additional error during resource cleanup: {str(cleanup_error)}")
