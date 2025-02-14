"""Worker initialization utilities."""

from __future__ import annotations
import logging
import os
from typing import Optional, Callable

from simpler_fine_bert.common.cuda_manager import cuda_manager
from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

class WorkerInitializer:
    """Handles worker process initialization."""
    
    @staticmethod
    def initialize(worker_id: int) -> None:
        """Initialize worker process.
        
        This function is designed to be used as a worker_init_fn for DataLoader.
        It ensures all necessary resources are properly initialized in the worker process.
        
        Args:
            worker_id: The ID of the worker process
        """
        try:
            logger.info(f"Initializing worker {worker_id} (PID: {os.getpid()})")
            
            # Initialize CUDA first
            cuda_manager.initialize()
            logger.debug(f"CUDA initialized for worker {worker_id}")
            
            # Initialize tokenizer
            tokenizer_manager.initialize()
            logger.debug(f"Tokenizer initialized for worker {worker_id}")
            
            # Initialize remaining resources
            ResourceInitializer.initialize_process()
            logger.debug(f"Resources initialized for worker {worker_id}")
            
            logger.info(f"Worker {worker_id} initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize worker {worker_id}: {str(e)}")
            raise

def get_worker_init_fn(num_workers: int) -> Optional[Callable[[int], None]]:
    """Get worker initialization function if needed.
    
    Args:
        num_workers: Number of workers to use
        
    Returns:
        Worker initialization function if workers are used, None otherwise
    """
    if num_workers > 0:
        return WorkerInitializer.initialize
    return None

__all__ = ['WorkerInitializer', 'get_worker_init_fn']
