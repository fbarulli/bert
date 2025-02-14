from __future__ import annotations
import torch
import logging
import traceback
from typing import Optional, Dict

from simpler_fine_bert.common.base_manager import BaseManager
from simpler_fine_bert.common.tensor_manager import tensor_manager

logger = logging.getLogger(__name__)

class BatchManager(BaseManager):
    """Process-local batch manager."""
    
    def _initialize_process_local(self):
        """Initialize process-local attributes."""
        # Initialize tensor manager first since we depend on it
        tensor_manager.ensure_initialized()
        
        # No state needed yet, but good practice to have the method
        pass
    
    def move_batch_to_device(
        self,
        batch: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Move batch to device and handle memory."""
        self.ensure_initialized()
        try:
            if device is None:
                device = tensor_manager.get_device()
                
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
                   
        except Exception as e:
            logger.error(f"Error moving batch to device: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    # Alias for backward compatibility
    prepare_batch = move_batch_to_device

# Global instance
batch_manager = BatchManager()
