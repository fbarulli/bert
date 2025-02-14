"""Core initialization utilities."""

from __future__ import annotations
import os
import logging
import multiprocessing as mp
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def initialize_worker(worker_id: int) -> None:
    """Initialize worker process.
    
    This function handles the core initialization sequence for worker processes.
    It uses lazy imports to avoid circular dependencies.
    
    Args:
        worker_id: The ID of the worker process
    """
    try:
        logger.info(f"Initializing worker {worker_id} (PID: {os.getpid()})")
        
        # Import managers at runtime to avoid circular imports
        from simpler_fine_bert.common.cuda_manager import cuda_manager
        from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
        from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
        
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

def initialize_process() -> Tuple[int, int]:
    """Initialize process-specific settings.
    
    This function handles the core initialization sequence for any process.
    It uses lazy imports to avoid circular dependencies.
    
    Returns:
        Tuple of (current_pid, parent_pid)
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        current_pid = os.getpid()
        parent_pid = os.getppid()
        
        # Set spawn method for any sub-processes
        if mp.get_start_method(allow_none=True) != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn'")
            except RuntimeError as e:
                logger.warning(f"Could not set spawn method: {e}")
        
        # Import ResourceInitializer at runtime to avoid circular imports
        from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
        
        # Initialize process resources
        ResourceInitializer.initialize_process()
        
        return current_pid, parent_pid
        
    except Exception as e:
        logger.error(f"Failed to initialize process {os.getpid()}: {str(e)}")
        raise

def cleanup_process() -> None:
    """Clean up process-specific resources.
    
    This function handles the cleanup sequence for any process.
    It uses lazy imports to avoid circular dependencies.
    """
    try:
        # Import ResourceInitializer at runtime to avoid circular imports
        from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
        
        ResourceInitializer.cleanup_process()
        logger.info(f"Process {os.getpid()} resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error cleaning up process {os.getpid()}: {str(e)}")
        raise

def get_worker_init_fn(num_workers: int) -> Optional[callable]:
    """Get worker initialization function if needed.
    
    Args:
        num_workers: Number of workers to use
        
    Returns:
        Worker initialization function if workers are used, None otherwise
    """
    if num_workers > 0:
        return initialize_worker
    return None

__all__ = [
    'initialize_worker',
    'initialize_process',
    'cleanup_process',
    'get_worker_init_fn'
]
