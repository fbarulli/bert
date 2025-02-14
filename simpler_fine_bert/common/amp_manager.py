from __future__ import annotations
import torch
import logging
import traceback
from typing import Optional, Dict, Any
from contextlib import contextmanager

from simpler_fine_bert.common.base_manager import BaseManager
from simpler_fine_bert.common.cuda_manager import cuda_manager

logger = logging.getLogger(__name__)

class AMPManager(BaseManager):
    """Process-local automatic mixed precision manager."""
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        try:
            # Call parent's initialization first
            super()._initialize_process_local(config)
            
            # Initialize scaler only if CUDA is available and FP16 is enabled
            if cuda_manager.is_available():
                training_config = self.get_config_section(config, 'training')
                if training_config.get('fp16', False):
                    self._local.scaler = torch.cuda.amp.GradScaler()
                    logger.info(f"Initialized GradScaler for process {self._local.pid}")
                else:
                    self._local.scaler = None
                    logger.info("FP16 not enabled, AMP disabled")
            else:
                self._local.scaler = None
                logger.warning("CUDA not available, AMP disabled")
        except Exception as e:
            logger.error(f"Failed to initialize GradScaler: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def is_enabled(self) -> bool:
        """Check if AMP is enabled."""
        self.ensure_initialized()
        return hasattr(self._local, 'scaler') and self._local.scaler is not None
            
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        self.ensure_initialized()
        if not self.is_enabled():
            return loss
        return self._local.scaler.scale(loss)
        
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with gradient unscaling."""
        self.ensure_initialized()
        if self.is_enabled():
            self._local.scaler.step(optimizer)
            self._local.scaler.update()
        else:
            optimizer.step()
        
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for clipping."""
        self.ensure_initialized()
        if self.is_enabled():
            self._local.scaler.unscale_(optimizer)
        
    def backward_step(
        self,
        loss: torch.Tensor,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_norm: Optional[float] = None
    ) -> None:
        """Perform backward pass with gradient scaling and clipping."""
        self.ensure_initialized()
            
        try:
            if self.is_enabled():
                # Scale loss and backward pass
                scaled_loss = self.scale_loss(loss)
                scaled_loss.backward()
                
                # Unscale gradients before clipping
                self.unscale_gradients(optimizer)
                
                # Clip gradients if requested
                if grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                
                # Step optimizer and update scaler
                self.step(optimizer)
            else:
                # Regular backward pass without AMP
                loss.backward()
                
                # Clip gradients if requested
                if grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                
                # Regular optimizer step
                optimizer.step()
            
        except Exception as e:
            logger.error(f"Error in backward step: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    @contextmanager
    def autocast(self) -> None:
        """Context manager for automatic mixed precision."""
        self.ensure_initialized()
        if self.is_enabled():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                yield
        else:
            yield

# Global instance
amp_manager = AMPManager()

__all__ = ['AMPManager', 'amp_manager']
