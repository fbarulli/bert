from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from simpler_fine_bert.common.utils import setup_logging, seed_everything
from simpler_fine_bert.common.config_utils import load_config
from simpler_fine_bert.embedding import train_embeddings, validate_embeddings
from simpler_fine_bert.common.resource import resource_factory
from simpler_fine_bert.classification import (
    run_classification_optimization,
    train_final_model
)

logger = logging.getLogger(__name__)

def train_model(config: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """Train model with proper process isolation."""
    try:
        # Set random seed
        seed_everything(config['training']['seed'])
        
        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
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
                study_name=config.get('study_name')
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
        
        logger.info(f"\n=== Creating {config['model']['stage'].title()} Resources ===")
        try:
            # Create dataset through resource factory
            dataset = resource_factory.create_resource('dataset', config)
            logger.info(f"Created dataset with {len(dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to create resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        logger.info("\n=== Starting Training ===")
        train_model(config)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up resources...")

if __name__ == '__main__':
    # Setup logging
    setup_logging()
    main()
