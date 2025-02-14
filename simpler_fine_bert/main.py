from __future__ import annotations

"""Main entry point for training."""

# Process setup must happen before any other imports
from simpler_fine_bert.common.process.multiprocessing_setup import setup_multiprocessing
setup_multiprocessing()

import logging
import os
from pathlib import Path
from typing import Dict, Any

from simpler_fine_bert.common.utils import setup_logging, seed_everything
from simpler_fine_bert.common.config_utils import load_config
from simpler_fine_bert.embedding import train_embeddings, validate_embeddings
from simpler_fine_bert.common.resource import resource_factory
from simpler_fine_bert.common.parameter_manager import parameter_manager
from simpler_fine_bert.common.resource_manager import resource_manager
from simpler_fine_bert.common.worker_manager import worker_manager
from simpler_fine_bert.common.storage_manager import storage_manager
from simpler_fine_bert.common.metrics_manager import metrics_manager
from simpler_fine_bert.common.cuda_manager import cuda_manager
from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
from simpler_fine_bert.classification import (
    run_classification_optimization,
    train_final_model
)

logger = logging.getLogger(__name__)

def train_model(config: Dict[str, Any]) -> None:
    """Train model with proper process isolation."""
    try:
        # Set random seed
        seed_everything(config['training']['seed'])
        
        # Create output directory from config
        output_dir = Path(config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train embedding model
        if config['model']['stage'] == 'embedding':
            logger.info("\n=== Starting Embedding Training ===")
            train_embeddings(config, output_dir)
            
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
        raise

def main():
    """Main entry point."""
    try:
        logger.info(f"Main Process ID: {os.getpid()}")
        logger.info("Loading configuration...")
        
        config = load_config("config_embedding.yaml")
        logger.info("Configuration loaded successfully")
        
        # Initialize resource manager with config first
        resource_manager.config = config
        
        # Initialize process resources with config
        ResourceInitializer.initialize_process(config)
        
        # Initialize remaining managers
        parameter_manager.base_config = config
        worker_manager.n_jobs = config['training']['n_jobs']
        storage_manager.storage_dir = Path(config['output']['dir']) / 'storage'
        
        logger.info("All managers initialized with config")
        
        logger.info("\n=== Starting Training ===")
        train_model(config)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up resources...")
        ResourceInitializer.cleanup_process()

if __name__ == '__main__':
    # Setup logging first for visibility
    setup_logging()
    main()
