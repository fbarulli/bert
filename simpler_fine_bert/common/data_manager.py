from __future__ import annotations
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import logging
import os
import traceback
from pathlib import Path
from filelock import FileLock
from typing import Dict, Any, Tuple
from transformers import PreTrainedTokenizerFast
from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
from torch.utils.data.dataloader import default_collate
import threading

from simpler_fine_bert.common.base_manager import BaseManager
from simpler_fine_bert.embedding import EmbeddingDataset
from simpler_fine_bert.common.dataloader_manager import dataloader_manager
from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

class DataManager(BaseManager):
    """Manages data resources with proper process and thread synchronization."""
    
    _shared_datasets = {}
    _lock = threading.Lock()
    
    def _initialize_process_local(self):
        """Initialize process-local attributes."""
        try:
            super()._initialize_process_local()
            self._local.resources = None
            logger.info(f"DataManager initialized for process {self._local.pid}")
        except Exception as e:
            logger.error(f"Failed to initialize DataManager: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _create_datasets(
        self,
        config: Dict[str, Any]
    ) -> Tuple[EmbeddingDataset, EmbeddingDataset]:
        """Create train and validation datasets."""
        try:
            # Get tokenizer through manager
            tokenizer = tokenizer_manager.get_worker_tokenizer(
                worker_id=os.getpid(),
                model_name=config['model']['name']
            )

            # Create datasets with shared memory tensors
            data_path = Path(config['data']['csv_path'])
            
            train_dataset = EmbeddingDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                split='train',
                train_ratio=config['data'].get('train_ratio', 0.9),
                max_length=config['data']['max_length']
            )
            
            val_dataset = EmbeddingDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                split='val',
                train_ratio=config['data'].get('train_ratio', 0.9),
                max_length=config['data']['max_length']
            )

            # Move tensors to shared memory
            for dataset in [train_dataset, val_dataset]:
                for key in dataset.data:
                    if isinstance(dataset.data[key], torch.Tensor):
                        dataset.data[key] = dataset.data[key].share_memory_()

            return train_dataset, val_dataset

        except Exception as e:
            logger.error(f"Error creating datasets: {str(e)}")
            raise

    def _create_dataloaders(
        self,
        train_dataset: EmbeddingDataset,
        val_dataset: EmbeddingDataset,
        config: Dict[str, Any],
        world_size: int = 1,
        rank: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        try:
            # Create samplers
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            ) if world_size > 1 else None

            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            ) if world_size > 1 else None

            # Create dataloaders with efficient settings
            train_loader = dataloader_manager.create_dataloader(
                dataset=train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=0,  # No workers needed since data is in shared memory
                collate_fn=default_collate,
                persistent_workers=False,  # No persistence needed
                prefetch_factor=2  # Prefetch 2 batches
            )

            val_loader = dataloader_manager.create_dataloader(
                dataset=val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                collate_fn=default_collate,
                persistent_workers=False,
                prefetch_factor=2
            )

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Error creating dataloaders: {str(e)}")
            raise

    def init_shared_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize shared data resources."""
        with self._lock:
            if not self._shared_datasets:
                logger.info("Initializing shared datasets")
                try:
                    # Create shared datasets
                    train_dataset, val_dataset = self._create_datasets(config)
                    self._shared_datasets['train'] = train_dataset
                    self._shared_datasets['val'] = val_dataset
                    logger.info("Successfully created shared datasets")
                except Exception as e:
                    logger.error(f"Failed to create shared datasets: {str(e)}")
                    raise

        return self._shared_datasets

    def init_process_resources(
        self,
        config: Dict[str, Any],
        world_size: int = 1,
        rank: int = 0
    ) -> Dict[str, Any]:
        """Initialize process-local resources using shared datasets."""
        self.ensure_initialized()
        
        try:
            # Ensure all process resources are initialized first
            ResourceInitializer.initialize_process()
            
            # Get or create shared datasets
            shared = self.init_shared_resources(config)
            
            # Get tokenizer through manager
            tokenizer = tokenizer_manager.get_worker_tokenizer(
                worker_id=os.getpid(),
                model_name=config['model']['name']
            )

            # Create process-local dataloaders
            train_loader, val_loader = self._create_dataloaders(
                shared['train'],
                shared['val'],
                config,
                world_size,
                rank
            )

            # Store process-local resources
            self._local.resources = {
                'tokenizer': tokenizer,
                'train_dataset': shared['train'],
                'val_dataset': shared['val'],
                'train_loader': train_loader,
                'val_loader': val_loader
            }
            
            # Validate resources
            self._validate_resources(self._local.resources)
            logger.info(f"Successfully initialized process {self._local.pid} resources")
            
        except Exception as e:
            logger.error(f"Error initializing process {self._local.pid} resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        return self._local.resources

    def _validate_resources(self, resources: Dict[str, Any]) -> None:
        """Validate that all required resources exist and are of correct type."""
        required = {
            'tokenizer': PreTrainedTokenizerFast,
            'train_dataset': EmbeddingDataset,
            'val_dataset': EmbeddingDataset,
            'train_loader': DataLoader,
            'val_loader': DataLoader
        }

        for name, expected_type in required.items():
            if name not in resources:
                raise ValueError(f"Missing required resource: {name}")
            if not isinstance(resources[name], expected_type):
                raise TypeError(f"Resource {name} has wrong type: {type(resources[name])}, expected {expected_type}")

data_manager = DataManager()
