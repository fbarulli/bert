from __future__ import annotations

import logging
import os
import gc
import random
import psutil
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TypeVar, Iterator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
from torch.optim.lr_scheduler import _LRScheduler
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')

def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    config: Optional[Dict[str, Any]] = None
) -> None:
    import coloredlogs
    
    # Determine log level from config or default to INFO
    if config and config.get('training', {}).get('debug_logging', False):
        level = "DEBUG"
    elif level is None:
        level = "INFO"
    
    # Set level
    numeric_level = getattr(logging, level.upper())
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    # Add colored logs for terminal
    coloredlogs.install(level=numeric_level)
    
    # Log the level being used
    logger.info(f"Logging level set to: {level}")

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_memory_usage() -> Dict[str, float]:
    process = psutil.Process()
    memory_info = process.memory_info()
    
    usage = {
        'rss': memory_info.rss / (1024 ** 3),  # Resident Set Size
        'vms': memory_info.vms / (1024 ** 3),  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        usage.update({
            'cuda_allocated': torch.cuda.memory_allocated() / (1024 ** 3),
            'cuda_reserved': torch.cuda.memory_reserved() / (1024 ** 3)
        })
    
    return usage

def clear_memory() -> None:
    # Clean up tokenizer resources
    from simpler_fine_bert.common.managers import get_tokenizer_manager
    tokenizer_manager = get_tokenizer_manager()
    tokenizer_manager.cleanup_worker(os.getpid())
    
    # Standard memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    desc: Optional[str] = None,
    chunk_size: Optional[int] = None,
    wandb_manager: Optional[Any] = None
) -> List[Any]:
    if max_workers is None:
        max_workers = os.cpu_count() or 4
        
    if chunk_size is None:
        chunk_size = max(1, len(items) // (max_workers * 4))
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        results = []
        total = len(items)
        
        for i, result in enumerate(executor.map(func, items, chunksize=chunk_size)):
            results.append(result)
            if wandb_manager:
                wandb_manager.log_progress(i + 1, total, prefix='parallel_')
                if desc:
                    wandb_manager.log_metrics({'parallel_task': desc})
        
        return results

def batch_iterator(
    items: List[T],
    batch_size: int,
    drop_last: bool = False
) -> Iterator[List[T]]:
    length = len(items)
    for ndx in range(0, length, batch_size):
        batch = items[ndx:min(ndx + batch_size, length)]
        if drop_last and len(batch) < batch_size:
            break
        yield batch

def create_memmap_array(
    path: Path,
    shape: tuple,
    dtype: np.dtype = np.float32,
    data: Optional[np.ndarray] = None
) -> np.ndarray:
    if data is not None:
        array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        array[:] = data[:]
        array.flush()
    else:
        array = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    
    return array

def load_memmap_array(
    path: Path,
    shape: tuple,
    dtype: np.dtype = np.float32,
    mode: str = 'r'
) -> np.ndarray:
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)

def measure_memory(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = get_memory_usage()
        try:
            result = func(*args, **kwargs)
            after = get_memory_usage()
            
            # Calculate differences
            diff = {
                k: after[k] - before[k]
                for k in before.keys()
            }
            
            logger.debug(
                f"Memory usage for {func.__name__}:\n"
                f"  Before: {before}\n"
                f"  After: {after}\n"
                f"  Difference: {diff}"
            )
            
            return result
        finally:
            clear_memory()
    
    return wrapper

def chunk_file(
    file_path: Path,
    chunk_size: str = '64MB'
) -> Iterator[bytes]:
    # Convert chunk size to bytes
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    size = int(chunk_size[:-2])
    unit = chunk_size[-2:].upper()
    chunk_bytes = size * units[unit]
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            yield chunk

def init_worker():
    print("DEBUG: init_worker() called in process:", os.getpid())
    # Ensure CUDA is initialized in worker
    if torch.cuda.is_available():
        torch.cuda.init()

def init_shared_resources(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize only non-CUDA resources in main process."""
    from simpler_fine_bert.common.managers import (
        get_data_manager,
        get_tokenizer_manager
    )
    
    # Get manager instances
    data_manager = get_data_manager()
    tokenizer_manager = get_tokenizer_manager()
    
    # Initialize data resources
    resources = data_manager.init_resources(config)
    
    # Initialize tokenizer resources
    tokenizer = tokenizer_manager.get_worker_tokenizer(
        worker_id=os.getpid(),
        model_name=config['model']['name']
    )
    resources['tokenizer'] = tokenizer
    
    return resources

import torch
from torch.optim.lr_scheduler import _LRScheduler, LRScheduler
from typing import Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)

def create_optimizer(model: torch.nn.Module, training_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer with proper configuration.
    
    Args:
        model: The model whose parameters will be optimized
        training_config: The training section of the configuration containing optimizer parameters
    """
    try:
        # Get optimizer parameters from training config
        optimizer_type = training_config['optimizer_type'].lower()
        lr = float(training_config['learning_rate'])
        weight_decay = float(training_config['weight_decay'])
        eps = 1e-8  # Fixed epsilon value

        # Get parameters to optimize
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # Create optimizer
        optimizer_map = {
            'adamw': torch.optim.AdamW,
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }

        if optimizer_type not in optimizer_map:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available options: {list(optimizer_map.keys())}")

        optimizer_cls = optimizer_map[optimizer_type]
        optimizer_kwargs = {'lr': lr, 'eps': eps} if 'adam' in optimizer_type else {'lr': lr}

        optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )

        logger.debug(f"Created {optimizer_type} optimizer with lr={lr}, weight_decay={weight_decay}")
        return optimizer

    except Exception as e:
        logger.error(f"Error creating optimizer: {str(e)}")
        raise

def create_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int, config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Number of training steps
        config: Configuration dictionary containing scheduler settings

    Returns:
        Learning rate scheduler or None if not configured
    """
    try:
        if not config['training']['scheduler']['use_scheduler']:
            return None

        scheduler_type = config['training']['scheduler']['type']
        warmup_ratio = config['training']['warmup_ratio']
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        min_lr_ratio = float(config['training']['scheduler']['min_lr_ratio'])
        
        logger.info(f"Creating {scheduler_type} scheduler:")
        logger.info(f"- warmup_ratio: {warmup_ratio}")
        logger.info(f"- num_warmup_steps: {num_warmup_steps}")
        logger.info(f"- num_training_steps: {num_training_steps}")
        logger.info(f"- min_lr_ratio: {min_lr_ratio}")

        if scheduler_type == 'cosine':
            return WarmupCosineScheduler(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=float(config['training']['scheduler']['num_cycles']),
                min_lr_ratio=min_lr_ratio
            )
        elif scheduler_type == 'linear':
            return WarmupLinearScheduler(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_ratio=min_lr_ratio
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    except Exception as e:
        logger.error(f"Error creating scheduler: {str(e)}")
        raise

class WarmupLinearScheduler(LRScheduler):
    """Linear learning rate scheduler with warmup."""
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase
            alpha = float(self.last_epoch) / float(max(1, self.num_warmup_steps))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Linear decay phase
            progress = float(self.last_epoch - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            return [base_lr * max(self.min_lr_ratio, 1.0 - progress) for base_lr in self.base_lrs]

class WarmupCosineScheduler(LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase
            alpha = float(self.last_epoch) / float(max(1, self.num_warmup_steps))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            progress = float(self.last_epoch - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay) for base_lr in self.base_lrs]

__all__ = [
    # Core utilities
    'setup_logging',
    'seed_everything',
    'get_memory_usage',
    'clear_memory',
    
    # Optimization-related
    'create_optimizer',
    'create_scheduler',
    'WarmupLinearScheduler',
    'WarmupCosineScheduler',
    
    # Data processing
    'parallel_map',
    'batch_iterator',
    'create_memmap_array',
    'load_memmap_array',
    'chunk_file',
    
    # Memory management
    'measure_memory',
    'init_worker',
    'init_shared_resources'
]
