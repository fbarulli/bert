from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
import torch

from simpler_fine_bert.data_manager import data_manager
from simpler_fine_bert.model_manager import model_manager
from simpler_fine_bert.common import get_cuda_manager
from simpler_fine_bert.embedding import EmbeddingTrainer
from simpler_fine_bert.common.utils import create_optimizer

# Get manager instances
data_manager = get_data_manager()
model_manager = get_model_manager()
cuda_manager = get_cuda_manager()

logger = logging.getLogger(__name__)

def init_shared_resources(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize only data resources using DataManager."""
    try:
        # Validate config
        required_config = ['data', 'training', 'model']
        if not all(k in config for k in required_config):
            raise ValueError(f"Missing required config sections: {[k for k in required_config if k not in config]}")

        # Use DataManager to create all resources
        resources = data_manager.init_shared_resources(config)
        
        # Validate returned resources
        required_resources = ['tokenizer', 'train_dataset', 'val_dataset', 'train_loader', 'val_loader']
        missing = [r for r in required_resources if r not in resources]
        if missing:
            raise ValueError(f"DataManager failed to provide required resources: {missing}")
        
        # Log initialization success
        logger.info("Initialized shared data resources using DataManager:")
        logger.info(f"- Train dataset size: {len(resources['train_dataset'])}")
        logger.info(f"- Val dataset size: {len(resources['val_dataset'])}")
        logger.info(f"- Using tokenizer: {config['model']['name']}")
        
        return resources
        
    except Exception as e:
        logger.error(f"Failed to initialize shared resources: {str(e)}")
        raise

def objective_fn(trial: optuna.Trial, resources: Dict[str, Any]) -> float:
    """Objective function that uses model_manager for model creation."""
    try:
        # Get worker-specific device and model
        worker_id = trial.number
        device_id = worker_id % torch.cuda.device_count() if torch.cuda.is_available() else None
        
        # Use model_manager to get model - no direct model creation
        model = model_manager.get_worker_model(
            worker_id=worker_id,
            model_name=config['model']['name'],
            device_id=device_id
        )
        
        # Get hyperparameter ranges from config
        hp_config = config['hyperparameters']
        
        # Create optimizer with trial parameters from config
        optimizer = create_optimizer(
            model=model,
            learning_rate=trial.suggest_float(
                'learning_rate',
                hp_config['learning_rate']['min'],
                hp_config['learning_rate']['max'],
                log=hp_config['learning_rate']['type'] == 'log'
            ),
            weight_decay=trial.suggest_float(
                'weight_decay',
                hp_config['weight_decay']['min'],
                hp_config['weight_decay']['max'],
                log=hp_config['weight_decay']['type'] == 'log'
            ),
            optimizer_type=config['training']['optimizer_type']
        )
        
        # Create trainer with provided resources
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=resources['train_loader'],
            val_loader=resources['val_loader'],
            optimizer=optimizer,
            scheduler=None,  # We don't use scheduler in trials
            device=cuda_manager.get_device(device_id),
            config=config,
            metrics_dir=Path(config['output']['dir']) / f'trial_{trial.number}',
            is_trial=True,
            trial=trial,
            job_id=worker_id
        )
        
        # Train and get validation metrics
        trainer.train(num_epochs=config['training']['num_epochs'])
        val_metrics = trainer.validate()
        
        # Return the metric to optimize
        return val_metrics['embedding_loss']
        
    except Exception as e:
        logger.error(f"Trial failed: {str(e)}")
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")
    finally:
        # Clean up resources
        model_manager.cleanup_worker(worker_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_embedding_trials(config: Dict[str, Any]) -> Optional[optuna.trial.FrozenTrial]:
    """Run optimization with deferred resource initialization."""
    try:
        # Validate required config fields
        if 'training' not in config:
            raise ValueError("Missing 'training' section in config")
        if 'n_jobs' not in config['training']:
            raise ValueError("Missing 'n_jobs' in training config")
        if 'study_name' not in config['training']:
            raise ValueError("Missing 'study_name' in training config")
            
        # Create study manager with direct config access
        study_manager = get_optuna_manager()(
            study_name=config['training']['study_name'],  # Direct access
            config=config,
            storage_dir=Path(config['output']['dir'])
        )
        
        # Run optimization - each worker will initialize its own resources
        best_trial = study_manager.optimize(objective_fn)
        return best_trial
        
    except Exception as e:
        logger.error(f"Embedding trials failed: {e}")
        raise
