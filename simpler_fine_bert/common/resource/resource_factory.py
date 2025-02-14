from __future__ import annotations
import torch 
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from simpler_fine_bert.common.cuda_utils import cuda_manager
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

def get_data_manager():
    """Get data manager instance at runtime to avoid circular imports."""
    from simpler_fine_bert.common.data_manager import data_manager
    return data_manager

class ResourceFactory:
    """Factory for creating process-local resources."""
    
    @staticmethod
    def create_resources(
        config: Dict[str, Any],
        device_id: Optional[int] = None,
        cache_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create fresh resources for a process."""
        try:
            # Get process-specific device
            device = cuda_manager.get_device(device_id if device_id is not None else 0)
            
            # Get data manager and create resources
            data_resources = get_data_manager().create_process_resources(config)
            
            return {
                'device': device,
                **data_resources
            }
            
        except Exception as e:
            logger.error(f"Error creating resources: {e}")
            raise
    
    @classmethod
    def get_resource_config(cls, resource: Any) -> Dict[str, Any]:
        """Extract configuration that can be used to recreate a resource."""
        if isinstance(resource, DataLoader):
            return {
                'type': 'DataLoader',
                'batch_size': resource.batch_size,
                'num_workers': resource.num_workers,
                'dataset': cls.get_resource_config(resource.dataset),
                'shuffle': resource.shuffle
            }
        elif isinstance(resource, Dataset):
            return {
                'type': resource.__class__.__name__,
                'params': {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in resource.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }
            }
        return None

resource_factory = ResourceFactory()
