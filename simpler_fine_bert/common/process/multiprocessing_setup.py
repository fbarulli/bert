"""Multiprocessing initialization utilities."""

import multiprocessing
import os
import logging

logger = logging.getLogger(__name__)

def setup_multiprocessing():
    """Set up multiprocessing with spawn method.
    
    This should be called at the very start of the program, before any other imports
    that might initialize multiprocessing.
    
    Raises:
        RuntimeError: If spawn method cannot be set and current method is not spawn
    """
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            logger.info(f"Set multiprocessing start method to 'spawn' in process {os.getpid()}")
        else:
            logger.debug(f"Multiprocessing already using spawn method in process {os.getpid()}")
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
