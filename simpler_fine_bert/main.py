from __future__ import annotations

"""Main entry point for training."""

# Initialize TensorFlow and suppress warnings before any other imports
from simpler_fine_bert.common.tensorflow_init import *

# Process setup must happen before any other imports
from simpler_fine_bert.common.process.multiprocessing_setup import setup_multiprocessing
setup_multiprocessing()

import logging
import os
import traceback
from pathlib import Path
from typing import Dict, Any

from simpler_fine_bert.common.utils import setup_logging, seed_everything
from simpler_fine_bert.common.config_utils import load_config

logger = logging.getLogger(__name__)

def train_model(config: Dict[str, Any]) -> None:
    """Train model with proper process isolation."""
    try:
        # Import training functions at runtime
        from simpler_fine_bert.embedding import train_embeddings, validate_embeddings
        from simpler_fine_bert.classification import (
            run_classification_optimization,
            train_final_model
        )
        
        # Set random seed
        seed_everything(config['training']['seed'])
        
        # Create output directory from config
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train embedding model
        if config['model']['stage'] == 'embedding':
            logger.info("\n=== Starting Embedding Training ===")
            
            # Get managers
            from simpler_fine_bert.common.managers import get_resource_manager, get_model_manager
            resource_manager = get_resource_manager()
            model_manager = get_model_manager()
            
            # Initialize process resources for resource manager
            resource_manager.initialize_process(process_id=os.getpid())
            
            # Create datasets and dataloaders
            train_dataset, val_dataset = resource_manager.create_datasets(config, stage='embedding')
            train_loader, val_loader = resource_manager.create_dataloaders(config, train_dataset, val_dataset)
            
            # Get model
            model = model_manager.get_worker_model(
                worker_id=0,  # Main process
                model_name=config['model']['name'],
                model_type=config['model']['stage'],  # 'embedding' or 'classification'
                device_id=None,  # Let manager handle device
                config=config
            )
            
            # Train model
            train_embeddings(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                metrics_dir=output_dir,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            
        # Train classification model
        elif config['model']['stage'] == 'classification':
            logger.info("\n=== Starting Classification Training ===")
            run_classification_optimization(
                embedding_model_path=config['model']['embedding_model_path'],
                config_path=config['model']['config_path'],
                study_name=config
            )
            
        else:
            raise ValueError(f"Unknown training stage: {config['model']['stage']}")
        
    except Exception as e:
        logger.error(f"Error initializing training: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def initialize_managers(config: Dict[str, Any]) -> None:
    """Initialize all managers with config."""
    try:
        # Import managers at runtime
        from simpler_fine_bert.common.managers import (
            get_resource_manager,
            get_parameter_manager,
            get_worker_manager,
            get_storage_manager,
            get_data_manager,
            get_model_manager
        )
        
        # Get manager instances
        resource_manager = get_resource_manager()
        parameter_manager = get_parameter_manager()
        worker_manager = get_worker_manager()
        storage_manager = get_storage_manager()
        data_manager = get_data_manager()
        model_manager = get_model_manager()
        from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
        
        # Initialize resource manager with config first
        resource_manager.config = config
        
        # Initialize process resources with config
        ResourceInitializer.initialize_process(config)
        
        # Initialize data manager before model manager since it provides tokenizer
        data_manager.config = config
        
        # Initialize remaining managers
        parameter_manager.base_config = config
        model_manager.config = config
        worker_manager.n_jobs = config['training']['n_jobs']
        storage_manager.storage_dir = Path(config['output']['dir']) / 'storage'
        
        logger.info("All managers initialized with config")
        
    except Exception as e:
        logger.error(f"Failed to initialize managers: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def cleanup_resources() -> None:
    """Clean up all resources."""
    try:
        # Import ResourceInitializer at runtime
        from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
        ResourceInitializer.cleanup_process()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def main():
    """Main entry point."""
    try:
        logger.info(f"Main Process ID: {os.getpid()}")
        logger.info("Loading configuration...")

        config = load_config("config_embedding.yaml")
        logger.info("Configuration loaded successfully")

        # Initialize all managers
        initialize_managers(config)

        # --- INSERT DEBUG CHECK HERE (USING logger.debug) ---
        masking_logger = logging.getLogger("simpler_fine_bert.embedding.masking")
        print(f"DEBUG CHECK: Masking Logger Level: {logging.getLevelName(masking_logger.level)}") # Changed to logger.debug
        # ------------------------------

        # --- FORCE DEBUG LEVEL ON MASKING LOGGER ---
        masking_logger.setLevel(logging.DEBUG)
        print(f"DEBUG CHECK (AFTER SETLEVEL): Masking Logger Level: {logging.getLevelName(masking_logger.level)}") # Added another debug print to confirm
        # -----------------------------------------

        logger.info("\n=== Starting Training ===")  # <--- DEBUG LINES SHOULD BE *ABOVE* THIS LINE

        train_model(config)
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        logger.info("Cleaning up resources...")
        cleanup_resources()
