"""CUDA utilities and process resource management."""

import logging
from typing import Optional

from simpler_fine_bert.common.cuda_manager import cuda_manager
from simpler_fine_bert.common.tensor_manager import tensor_manager
from simpler_fine_bert.common.batch_manager import batch_manager
from simpler_fine_bert.common.metrics_manager import metrics_manager
from simpler_fine_bert.common.amp_manager import amp_manager
from simpler_fine_bert.common.process.initialization import (
    initialize_process,
    cleanup_process
)

logger = logging.getLogger(__name__)

def get_cuda_device():
    """Get the current CUDA device after initialization."""
    return cuda_manager.get_device()

__all__ = [
    'cuda_manager',
    'tensor_manager',
    'batch_manager',
    'metrics_manager',
    'amp_manager',
    'initialize_process',
    'get_cuda_device',
    'cleanup_process'
]
