"""CUDA utilities and process resource management."""

import logging
from typing import Optional

from simpler_fine_bert.common.cuda_manager import cuda_manager
from simpler_fine_bert.common.tensor_manager import tensor_manager
from simpler_fine_bert.common.batch_manager import batch_manager
from simpler_fine_bert.common.metrics_manager import metrics_manager
from simpler_fine_bert.common.amp_manager import amp_manager
from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

def initialize_process_resources() -> None:
    """Initialize all process-local resources in the correct order."""
    try:
        ResourceInitializer.initialize_process()
        logger.info("Process resources initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize process resources: {str(e)}")
        raise

def get_cuda_device():
    """Get the current CUDA device after initialization."""
    return cuda_manager.get_device()

def cleanup_process_resources() -> None:
    """Clean up all process-local resources."""
    try:
        ResourceInitializer.cleanup_process()
        logger.info("Process resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up process resources: {str(e)}")
        raise

__all__ = [
    'cuda_manager',
    'tensor_manager',
    'batch_manager',
    'metrics_manager',
    'amp_manager',
    'initialize_process_resources',
    'get_cuda_device',
    'cleanup_process_resources'
]
