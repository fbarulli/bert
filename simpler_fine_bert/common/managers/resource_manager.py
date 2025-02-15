from __future__ import annotations
import torch
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging
import os
import traceback
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
from simpler_fine_bert.common.managers import get_cuda_manager

# Get manager instance
cuda_manager = get_cuda_manager()
from simpler_fine_bert.common.resource import resource_factory

logger = logging.getLogger(__name__)

# Resource types that can be managed
RESOURCE_TYPES = {
    'dataset': 'Dataset resources',
    'model': 'Model resources',
    'optimizer': 'Optimizer resources',
    'dataloader': 'DataLoader resources'
}

@dataclass
class ResourceConfig:
    """Serializable configuration for resources."""
    dataset_params: Dict[str, Any]
    dataloader_params: Dict[str, Any]
    model_params: Dict[str, Any]
    device_id: Optional[int] = None

class ProcessResourceManager:
    """Manages per-process resource creation and cleanup."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process_id = None
        self.resources = {}
        import inspect
        logger.debug(f"DEBUG: ProcessResourceManager.__init__() called in process: {os.getpid()} - caller: {inspect.stack()[1].filename} : {inspect.stack()[1].lineno}")

    def create_datasets(
        self,
        config: Dict[str, Any],
        stage: str = 'embedding'  # 'embedding' or 'classification'
    ) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
        train_dataset = resource_factory.create_resource(
            'dataset', 
            config,
            split='train',
            stage=stage
        )
        val_dataset = resource_factory.create_resource(
            'dataset',
            config,
            split='val',
            stage=stage
        )
        return train_dataset, val_dataset

    def create_dataloaders(
        self,
        config: Dict[str, Any],
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        train_loader = resource_factory.create_resource(
            'dataloader',
            config,
            dataset=train_dataset,
            split='train'
        )
        val_loader = resource_factory.create_resource(
            'dataloader',
            config,
            dataset=val_dataset,
            split='val'
        )
        return train_loader, val_loader

    def initialize_process(self, process_id: int, device_id: Optional[int] = None) -> None:
        """Initialize resources for this process."""
        # Use ResourceInitializer for process initialization with config
        ResourceInitializer.initialize_process(self.config)
        
        self.process_id = process_id
        self.device_id = device_id
        
        # Create process-specific resources after initialization
        self._create_process_resources()
        
    def _create_process_resources(self) -> None:
        """Create new resources specific to this process."""
        try:
            from simpler_fine_bert.common.managers.data_manager import DataManager

            # Get process-specific device using initialized cuda_manager
            device = cuda_manager.get_device()

            # Get data manager instance from BaseManager registry
            from simpler_fine_bert.common.managers import get_data_manager
            data_mgr = get_data_manager()
            
            # Create and track resources
            self.resources = {
                'device': device,
                **data_mgr.init_process_resources(self.config)
            }
            
            # Log created resources
            for resource_type, resource in self.resources.items():
                logger.debug(
                    f"Created resource {resource_type} of type {type(resource).__name__} "
                    f"for process {self.process_id}"
                )
                
        except Exception as e:
            logger.error(f"Failed to create process resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_resource(self, name: str) -> Any:
        """Get a resource, creating it if needed."""
        if self.process_id is None:
            raise RuntimeError("Process not initialized")
            
        if name not in self.resources:
            raise KeyError(f"Resource {name} not available")
        return self.resources[name]
    
    def cleanup(self) -> None:
        """Clean up process resources."""
        try:
            # Clean up individual resources first
            for resource_type, resource in self.resources.items():
                try:
                    if hasattr(resource, 'cleanup'):
                        resource.cleanup()
                    logger.debug(f"Cleaned up resource {resource_type}")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_type}: {str(e)}")
            
            self.resources.clear()
            
            # Use ResourceInitializer for final cleanup
            ResourceInitializer.cleanup_process()
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Create singleton instance with empty config (will be initialized later)
resource_manager = ProcessResourceManager({})

__all__ = ['ProcessResourceManager', 'ResourceConfig', 'resource_manager', 'RESOURCE_TYPES']
