"""Process initialization utilities."""

import logging
import os
from typing import Tuple

from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

def initialize_process_resources() -> None:
    """Initialize all process-local resources.
    
    This should be called at the start of any new process to ensure proper setup.
    Uses ResourceInitializer to handle the actual initialization.
    """
    try:
        ResourceInitializer.initialize_process()
        logger.info("Process resources initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize process resources: {str(e)}")
        raise

def initialize_process() -> Tuple[int, int]:
    """Initialize process-specific settings.
    
    This should be called at the start of any new process to ensure proper setup.
    
    Returns:
        Tuple of (current_pid, parent_pid)
    """
    current_pid = ResourceInitializer.initialize_process()
    parent_pid = os.getppid()
    return current_pid, parent_pid

def cleanup_process_resources() -> None:
    """Clean up process-specific resources.
    
    This should be called when cleaning up a process to ensure proper resource cleanup.
    Uses ResourceInitializer to handle the actual cleanup.
    """
    try:
        ResourceInitializer.cleanup_process()
        logger.info(f"Process {os.getpid()} resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up process resources: {str(e)}")
        raise

__all__ = [
    'initialize_process',
    'initialize_process_resources',
    'cleanup_process_resources'
]
