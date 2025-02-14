from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Callable
import torch
import optuna

logger = logging.getLogger(__name__)

class ParallelStudy:
    """Handles parallel training using multiple processes."""
    
    def __init__(
        self,
        study_name: str,
        storage_dir: Path,
        n_gpus: int,
        config: Dict[str, Any]
    ):
        self.study_name = study_name
        self.storage_dir = storage_dir
        self.config = config
        
        # Always use single GPU
        if n_gpus > 1:
            logger.warning("Multi-GPU support is disabled. Using single GPU.")
        
    def run(self, objective_fn: Callable):
        """Run study on single GPU."""
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        try:
            # Create storage URL
            storage_url = f"sqlite:///{self.storage_dir}/optuna.db"
            
            # Create study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage_url,
                load_if_exists=True
            )
            
            # Run trials
            study.optimize(
                objective_fn,
                n_trials=self.config['training']['num_trials']
            )
            
        except Exception as e:
            logger.error(f"Study failed: {str(e)}")
            raise
