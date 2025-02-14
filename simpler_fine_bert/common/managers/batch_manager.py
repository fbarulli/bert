from __future__ import annotations
import torch
import logging
import traceback
from typing import Dict, Any, Optional, Union
from torch.utils.data import DataLoader

from simpler_fine_bert.common.managers.base_manager import BaseManager
from simpler_fine_bert.common.managers import get_cuda_manager, get_tensor_manager

logger = logging.getLogger(__name__)

class BatchManager(BaseManager):
    """Process-local batch manager for device placement and memory management."""
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        # Call parent's initialization first
        super()._initialize_process_local(config)
        
        # Get managers at runtime
        cuda_manager = get_cuda_manager()
        tensor_manager = get_tensor_manager()
        
        # Initialize dependencies
        cuda_manager.ensure_initialized()
        tensor_manager.ensure_initialized()
        
        # Initialize local state
        self._local.device = None
        
    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors to target device."""
        self.ensure_initialized()
        try:
            if device is None:
                cuda_manager = get_cuda_manager()
                device = cuda_manager.get_device()
                
            # Move each tensor to device
            return {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
        except Exception as e:
            logger.error(f"Error preparing batch: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def get_batch_size(self, batch: Union[Dict[str, torch.Tensor], DataLoader]) -> int:
        """Get batch size from batch dict or dataloader."""
        self.ensure_initialized()
        try:
            if isinstance(batch, dict):
                # Get first tensor's batch size
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        return v.size(0)
                raise ValueError("No tensors found in batch dict")
            elif isinstance(batch, DataLoader):
                return batch.batch_size
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")
                
        except Exception as e:
            logger.error(f"Error getting batch size: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Global instance
batch_manager = BatchManager()

__all__ = ['BatchManager', 'batch_manager']
