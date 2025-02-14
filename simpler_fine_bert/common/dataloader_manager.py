from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
import os
import logging
import traceback
from typing import Any, Optional
from torch.utils.data import Dataset, DataLoader

from simpler_fine_bert.base_manager import BaseManager
from simpler_fine_bert.cuda_manager import cuda_manager

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id: int) -> None:
    """Initialize worker process with centralized resource management."""
    try:
        # Initialize all process resources through ResourceInitializer
        from simpler_fine_bert.resource_initializer import ResourceInitializer
        current_pid = ResourceInitializer.initialize_process()
        logger.info(f"Initialized DataLoader worker {worker_id} (PID: {current_pid})")
    except Exception as e:
        logger.error(f"Failed to initialize DataLoader worker {worker_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Attempt cleanup on failure
        try:
            ResourceInitializer.cleanup_process()
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up worker {worker_id}: {str(cleanup_error)}")
        raise

class DataLoaderManager(BaseManager):
    """Process-local dataloader manager."""
    
    def _initialize_process_local(self):
        """Initialize process-local attributes."""
        try:
            # Call parent's initialization first
            super()._initialize_process_local()
            
            logger.info("Initializing DataLoaderManager for process %s", os.getpid())
            
            # Initialize thread-local storage with defaults before accessing
            if not hasattr(self._local, 'num_workers'):
                self._local.num_workers = 0
            if not hasattr(self._local, 'pin_memory'):
                self._local.pin_memory = False  # Safe default
                
            # Ensure cuda_manager is initialized
            if not hasattr(cuda_manager._local, 'initialized'):
                cuda_manager.initialize()
                
            # Update pin_memory based on CUDA availability
            try:
                self._local.pin_memory = cuda_manager.is_available()
            except Exception as e:
                logger.warning(f"Could not check CUDA availability, defaulting pin_memory to False: {e}")
                self._local.pin_memory = False
                
            self._local.settings_initialized = True

            logger.info(
                "DataLoader settings initialized for process %s (pin_memory=%s)",
                self._local.pid,
                self._local.pin_memory,
            )
        except Exception as e:
            logger.error("Failed to initialize DataLoader settings: %s", str(e))
            logger.error(traceback.format_exc())
            # Clean up partial initialization
            if hasattr(self._local, 'settings_initialized'):
                delattr(self._local, 'settings_initialized')
            raise

    def is_initialized(self) -> bool:
        """Check if manager is fully initialized."""
        return (
            super().is_initialized() and
            hasattr(self._local, 'settings_initialized') and
            self._local.settings_initialized
        )
            
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **kwargs: Any
    ) -> DataLoader:
        """Create dataloader with proper settings."""
        self.ensure_initialized()
        try:
            # Ensure thread-local storage is properly initialized
            if not hasattr(self._local, 'num_workers'):
                self._initialize_process_local()
                
            # Use instance defaults if not specified
            if num_workers is None:
                num_workers = self._local.num_workers
            if pin_memory is None:
                # Only enable pin_memory if explicitly set or if CUDA is available
                try:
                    pin_memory = hasattr(self._local, 'pin_memory') and self._local.pin_memory
                except AttributeError:
                    logger.warning("pin_memory not initialized, defaulting to False")
                    pin_memory = False
                
            # Create dataloader with settings and worker initialization
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                **kwargs
            )
            
            logger.debug(
                f"Created DataLoader with batch_size={batch_size}, "
                f"num_workers={num_workers}, pin_memory={pin_memory}"
            )
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def set_worker_settings(
        self,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None
    ) -> None:
        """Update default worker settings."""
        self.ensure_initialized()
        try:
            self._local.num_workers = num_workers
            if pin_memory is not None:
                # Only enable pin_memory if CUDA is available
                self._local.pin_memory = pin_memory and cuda_manager.is_available()
                
            logger.debug(
                f"Updated DataLoader settings: num_workers={self._local.num_workers}, "
                f"pin_memory={self._local.pin_memory}"
            )
            
        except Exception as e:
            logger.error(f"Error updating worker settings: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Global instance
dataloader_manager = DataLoaderManager()
