"""Multiprocessing initialization utilities."""

import multiprocessing
import os
import logging
import gc

logger = logging.getLogger(__name__)

def _cleanup_cuda_state():
    """Clean up CUDA state before worker processes."""
    try:
        import torch.cuda
        if torch.cuda.is_available():
            # Clear CUDA memory
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # Ensure CUDA is not initialized
            if torch.cuda.is_initialized():
                torch.cuda._lazy_init()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug(f"Cleaned up CUDA state in process {os.getpid()}")
    except Exception as e:
        logger.warning(f"Could not clean up CUDA state: {e}")

def setup_multiprocessing():
    """Set up multiprocessing with spawn method.
    
    This should be called at the very start of the program, before any other imports
    that might initialize multiprocessing or CUDA.
    
    Raises:
        RuntimeError: If spawn method cannot be set and current method is not spawn
    """
    try:
        # Clean up any existing CUDA state first
        _cleanup_cuda_state()
        
        # Set spawn method
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            logger.info(f"Set multiprocessing start method to 'spawn' in process {os.getpid()}")
        else:
            logger.debug(f"Multiprocessing already using spawn method in process {os.getpid()}")
            
        # Verify CUDA libraries are not loaded
        import torch._C
        if not hasattr(torch._C, '_cuda_isInited'):
            logger.debug("CUDA libraries not yet initialized")
        
    except RuntimeError as e:
        current_method = multiprocessing.get_start_method()
        if current_method != 'spawn':
            logger.error(f"Failed to set spawn method, using {current_method}")
            raise RuntimeError(f"Multiprocessing must use spawn method, got {current_method}")
        logger.debug(f"Multiprocessing already initialized with spawn method in process {os.getpid()}")

def verify_spawn_method():
    """Verify that spawn method is being used.
    
    This can be called in worker processes to verify proper initialization.
    
    Raises:
        RuntimeError: If current method is not spawn
    """
    current_method = multiprocessing.get_start_method()
    if current_method != 'spawn':
        raise RuntimeError(f"Multiprocessing must use spawn method, got {current_method}")
    logger.debug(f"Verified spawn method in process {os.getpid()}")
