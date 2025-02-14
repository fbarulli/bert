from __future__ import annotations
import torch
import logging
import traceback
from typing import Optional, Union, List, Tuple
import numpy as np

from simpler_fine_bert.common.base_manager import BaseManager
from simpler_fine_bert.common.cuda_manager import cuda_manager  # Correct import


logger = logging.getLogger(__name__)

class TensorManager(BaseManager):
    """Process-local tensor manager for device placement and memory management."""
    
    def _initialize_process_local(self):
        """Initialize process-local attributes."""
        # Initialize cuda_manager first since we depend on it
        cuda_manager.ensure_initialized()
        self._local.device = None
            
    def create_tensor(
        self,
        data: Union[torch.Tensor, List, np.ndarray],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Create tensor with proper device placement."""
        self.ensure_initialized()
        try:
            # Convert data to tensor if needed
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
                
            # Set device
            if device is None:
                device = self.get_device()
                
            # Move tensor to device
            data = data.to(device=device, dtype=dtype)
            data.requires_grad = requires_grad
            
            return data
            
        except Exception as e:
            logger.error(f"Error creating tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def get_device(self) -> torch.device:
        """Get current device."""
        self.ensure_initialized()
        if self._local.device is None:
            if cuda_manager.is_available():
                self._local.device = torch.device('cuda')
            else:
                self._local.device = torch.device('cpu')
        return self._local.device
        
    def create_cpu_tensor(
        self,
        data: Union[torch.Tensor, List, np.ndarray],
        dtype: Optional[torch.dtype] = None,
        pin_memory: bool = True
    ) -> torch.Tensor:
        """Create tensor on CPU with optional pinned memory."""
        self.ensure_initialized()
        try:
            # Convert data to tensor if needed
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
                
            # Move to CPU and set dtype
            data = data.cpu()
            if dtype is not None:
                data = data.to(dtype=dtype)
                
            # Pin memory if requested and CUDA is available
            if pin_memory and cuda_manager.is_available():
                data = data.pin_memory()
                
            return data
            
        except Exception as e:
            logger.error(f"Error creating CPU tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_random(
        self,
        size: Union[Tuple[int, ...], List[int]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Create random tensor between 0 and 1."""
        self.ensure_initialized()
        try:
            if device is None:
                device = self.get_device()
            return torch.rand(size, device=device)
        except Exception as e:
            logger.error(f"Error creating random tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def create_random_int(
        self,
        low: int,
        high: int,
        size: Union[Tuple[int, ...], List[int]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create random integer tensor between low and high."""
        self.ensure_initialized()
        try:
            if device is None:
                device = self.get_device()
            return torch.randint(low, high, size, device=device, dtype=dtype)
        except Exception as e:
            logger.error(f"Error creating random int tensor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def clear_memory(self) -> None:
        """Clear CUDA memory cache."""
        self.ensure_initialized()
        if cuda_manager.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

# Global instance
tensor_manager = TensorManager()

__all__ = ['TensorManager', 'tensor_manager']
