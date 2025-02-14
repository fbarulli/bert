"""CUDA utilities and process resource management."""

import logging
from typing import Optional, Dict, Any
import torch
import gc

logger = logging.getLogger(__name__)

def get_cuda_manager():
    """Get cuda manager instance at runtime."""
    from simpler_fine_bert.common.cuda_manager import cuda_manager
    return cuda_manager

def get_tensor_manager():
    """Get tensor manager instance at runtime."""
    from simpler_fine_bert.common.tensor_manager import tensor_manager
    return tensor_manager

def get_cuda_device(device_id: Optional[int] = None) -> torch.device:
    """Get the current CUDA device."""
    if torch.cuda.is_available():
        return torch.device(f'cuda:{device_id if device_id is not None else 0}')
    return torch.device('cpu')

def initialize_process(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize process resources."""
    # Import initialization utilities at runtime
    from simpler_fine_bert.common.process.initialization import initialize_process as _init
    return _init(config)

def cleanup_process() -> None:
    """Clean up process resources."""
    # Import initialization utilities at runtime
    from simpler_fine_bert.common.process.initialization import cleanup_process as _cleanup
    return _cleanup()

def clear_cuda_memory() -> None:
    """Clear CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_cuda_memory_stats() -> Dict[str, float]:
    """Get current CUDA memory statistics."""
    if not torch.cuda.is_available():
        return {}
        
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
        'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
    }

def reset_cuda_stats() -> None:
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
            torch.cuda.reset_accumulated_memory_stats()

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()

def get_cuda_device_count() -> int:
    """Get number of available CUDA devices."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def get_cuda_device_properties(device_id: Optional[int] = None) -> Dict[str, Any]:
    """Get properties of CUDA device."""
    if not torch.cuda.is_available():
        return {}
        
    device = device_id if device_id is not None else 0
    props = torch.cuda.get_device_properties(device)
    return {
        'name': props.name,
        'total_memory': props.total_memory / 1024**3,  # GB
        'major': props.major,
        'minor': props.minor,
        'multi_processor_count': props.multi_processor_count
    }

__all__ = [
    'get_cuda_manager',
    'get_tensor_manager',
    'get_cuda_device',
    'initialize_process',
    'cleanup_process',
    'clear_cuda_memory',
    'get_cuda_memory_stats',
    'reset_cuda_stats',
    'is_cuda_available',
    'get_cuda_device_count',
    'get_cuda_device_properties'
]
