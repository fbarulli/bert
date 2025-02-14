"""Process startup utilities."""

import os
import logging
import multiprocessing
import torch.distributed as dist

logger = logging.getLogger(__name__)

def set_start_method():
    """Set multiprocessing start method to spawn.
    
    This must be called before any other imports that might initialize multiprocessing.
    """
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info(f"Set multiprocessing start method to 'spawn' in process {os.getpid()}")
    except RuntimeError as e:
        current_method = multiprocessing.get_start_method()
        if current_method != 'spawn':
            logger.error(f"Failed to set spawn method, using {current_method}")
            raise RuntimeError("Must use spawn method") from e
        logger.debug(f"Multiprocessing already using spawn method in process {os.getpid()}")

def init_distributed(backend: str = "nccl", init_method: str = "env://"):
    """Initialize distributed training.
    
    Args:
        backend: PyTorch distributed backend ("nccl" for GPU, "gloo" for CPU)
        init_method: Process group initialization method
    """
    if 'RANK' not in os.environ:
        return

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        logger.info(
            f"Initialized distributed process group (rank {dist.get_rank()}"
            f"/{dist.get_world_size()}, local rank {os.environ['LOCAL_RANK']})"
        )
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise

def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group")

def setup_process(config: dict = None):
    """Set up process with proper initialization sequence.
    
    Args:
        config: Optional configuration dictionary for distributed settings
    """
    # Set spawn method first
    set_start_method()
    
    # Initialize distributed if running with torchrun
    if config and 'RANK' in os.environ:
        init_distributed(
            backend=config['training']['distributed']['backend'],
            init_method=config['training']['distributed']['init_method']
        )
