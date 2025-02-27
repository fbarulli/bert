import torch
import logging
import os
import threading
from typing import Dict, Optional
import gc
import time

logger = logging.getLogger(__name__)

class ResourcePool:
    """Process-local CUDA resource manager."""
    def __init__(self, memory_limit_gb: float = 5.5):  # Increased to 5.5GB per process
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            logger.warning("CUDA not available, using CPU")
            return

        # Convert GB to bytes
        self.memory_limit = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.lock = threading.Lock()
        self.last_cleanup = 0
        self.cleanup_interval = 0.1  # seconds

        # Log initial state
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"CUDA memory limit per process: {memory_limit_gb:.2f}GB "
                       f"(Total available: {total_memory/1e9:.2f}GB)")

    def check_memory(self, size_bytes: Optional[int] = None) -> bool:
        """Check if memory usage is within limits."""
        if self.device.type == 'cpu':
            return True

        try:
            current_allocated = torch.cuda.memory_allocated(self.device)
            
            # If size is specified, check if we have enough space
            if size_bytes is not None:
                return (current_allocated + size_bytes) <= self.memory_limit
                
            return current_allocated <= self.memory_limit
            
        except Exception as e:
            logger.error(f"Error checking memory: {str(e)}")
            return False

    def request_memory(self, size_bytes: int) -> bool:
        """Request memory allocation."""
        if self.device.type == 'cpu':
            return True

        try:
            with self.lock:
                # Check if we need cleanup
                if not self.check_memory(size_bytes):
                    current_time = time.time()
                    if current_time - self.last_cleanup >= self.cleanup_interval:
                        self.cleanup()
                        self.last_cleanup = current_time
                    
                    # Check again after cleanup
                    if not self.check_memory(size_bytes):
                        logger.warning(f"Memory request for {size_bytes/1e9:.2f}GB exceeds limit")
                        return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error requesting memory: {str(e)}")
            return False

    def cleanup(self):
        """Clean up CUDA memory."""
        if self.device.type == 'cpu':
            return

        try:
            # Force garbage collection first
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log memory state
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            logger.debug(f"Memory after cleanup - Allocated: {allocated/1e9:.2f}GB, "
                        f"Reserved: {reserved/1e9:.2f}GB")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_stats(self) -> Dict[str, int]:
        """Get current memory statistics."""
        if self.device.type == 'cpu':
            return {'allocated': 0, 'reserved': 0, 'limit': 0}
            
        return {
            'allocated': torch.cuda.memory_allocated(self.device),
            'reserved': torch.cuda.memory_reserved(self.device),
            'limit': self.memory_limit
        }

# Global resource pool instance
resource_pool = ResourcePool()
