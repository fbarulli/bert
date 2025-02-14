from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Callable
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import optuna

logger = logging.getLogger(__name__)

class ParallelStudy:
    """Handles distributed training across multiple GPUs/processes."""
    
    def __init__(
        self,
        study_name: str,
        storage_dir: Path,
        n_gpus: int,
        config: Dict[str, Any]
    ):
        self.study_name = study_name
        self.storage_dir = storage_dir
        self.n_gpus = n_gpus
        self.config = config
        self.world_size = n_gpus if torch.cuda.is_available() else 1
        
        # Set distributed environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
    def run(self, objective_fn: Callable):
        """Run study using torch.distributed."""
        if self.world_size > 1:
            mp.spawn(
                self._distributed_objective,
                args=(objective_fn,),
                nprocs=self.world_size,
                join=True
            )
        else:
            self._distributed_objective(0, objective_fn)

    def _distributed_objective(self, rank: int, objective_fn: Callable):
        """Initialize process group and run objective."""
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        
        # Initialize process group
        init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo'
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        
        try:
            # Create storage URL
            storage_url = f"sqlite:///{self.storage_dir}/optuna.db"
            
            # Create study for this process
            study = optuna.create_study(
                study_name=f"{self.study_name}_rank{rank}",
                storage=storage_url,
                load_if_exists=True
            )
            
            # Run trials for this process
            study.optimize(
                objective_fn,
                n_trials=self.config['training']['num_trials'] // self.world_size
            )
            
        finally:
            destroy_process_group()
