from __future__ import annotations

import logging
import csv
import numpy as np
import torch
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Iterator
from torch.utils.data import IterableDataset, Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import functools
import gc

from simpler_fine_bert.common.utils import parallel_map, create_memmap_array, load_memmap_array, measure_memory
from simpler_fine_bert.common.directory_manager import DirectoryManager
from simpler_fine_bert.common.cuda_utils import cuda_manager
from simpler_fine_bert.common.tensor_manager import tensor_manager

logger = logging.getLogger(__name__)

@dataclass
class TokenizedBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    labels: Optional[np.ndarray] = None

def process_batch_parallel(tokenizer: PreTrainedTokenizerFast, texts: List[str], max_length: int) -> Dict[str, np.ndarray]:
    """Process a batch of texts in parallel using ThreadPoolExecutor."""
    def _tokenize_text(text: str) -> Dict[str, np.ndarray]:
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'  # Return numpy arrays directly
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        encodings = list(executor.map(_tokenize_text, texts))
    
    # Stack results
    return {
        'input_ids': np.stack([e['input_ids'] for e in encodings]),
        'attention_mask': np.stack([e['attention_mask'] for e in encodings])
    }

class CSVDataset(Dataset):
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        split: str = 'train',
        train_ratio: float = 0.9
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.train_ratio = train_ratio
        # Create cache directory in the project root
        cache_dir = Path(os.getcwd()) / '.cache'
        cache_dir.mkdir(exist_ok=True)
        
        # Generate cache paths using absolute paths
        data_hash = hash(str(self.data_path.absolute()) + str(max_length))
        self.cache_prefix = cache_dir / f"dataset_{data_hash}"
        
        # Load or create memmapped arrays
        input_ids_path = self.cache_prefix.parent / f"{self.cache_prefix.name}_input_ids.npy"
        if not input_ids_path.exists():
            logger.info("Creating memory-mapped arrays for dataset...")
            self._create_memmap_arrays()
        else:
            logger.info("Loading existing memory-mapped arrays...")
            try:
                self._load_memmap_arrays()
            except Exception as e:
                logger.error(f"Error loading memory-mapped arrays: {str(e)}")
                logger.info("Cleaning up cache and recreating arrays...")
                self._cleanup_cache()
                self._create_memmap_arrays()
            
        # Calculate split indices
        self.split_idx = int(self.total_rows * train_ratio)
        if split == 'train':
            self.start_idx = 0
            self.end_idx = self.split_idx
        else:  # val
            self.start_idx = self.split_idx
            self.end_idx = self.total_rows
    
    def _cleanup_cache(self):
        """Clean up cache files."""
        for file in [
            f"{self.cache_prefix}_input_ids.npy",
            f"{self.cache_prefix}_attention_mask.npy",
            f"{self.cache_prefix}_labels.npy",
            f"{self.cache_prefix}_shapes.npy"
        ]:
            try:
                Path(file).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete {file}: {str(e)}")
            
    def _create_memmap_arrays(self):
        """Create memory-mapped arrays from scratch."""
        try:
            # First pass: count valid rows and collect texts
            texts = []
            labels = []
            with open(self.data_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'rating' in row:
                        try:
                            rating = int(row['rating'])
                            if not 1 <= rating <= 5:
                                logger.error(f"Rating must be between 1-5, got: {rating}")
                                continue
                            label = rating - 1  # Convert 1-5 to 0-4
                            texts.append(row['text'])
                            labels.append(label)
                        except ValueError:
                            logger.error(f"Rating must be an integer, got: {row['rating']}")
                            continue
                    else:
                        texts.append(row['text'])
                        labels.append(-1)  # Use -1 for no label
            
            # Tokenize in batches to save memory
            logger.info("Tokenizing texts...")
            batch_size = 1000
            self.total_rows = len(texts)
            
            # Create memmapped arrays and save shapes
            input_shape = (self.total_rows, self.max_length)
            labels_shape = (self.total_rows,)
            
            # Save shapes to metadata file
            shapes = {
                'input_ids': input_shape,
                'attention_mask': input_shape,
                'labels': labels_shape
            }
            np.save(f"{self.cache_prefix}_shapes.npy", shapes)
            
            self.input_ids = create_memmap_array(f"{self.cache_prefix}_input_ids.npy", input_shape, dtype=np.int64)
            self.attention_mask = create_memmap_array(f"{self.cache_prefix}_attention_mask.npy", input_shape, dtype=np.int64)
            self.labels = create_memmap_array(f"{self.cache_prefix}_labels.npy", labels_shape, dtype=np.int64)
            
            # Process in batches using parallel processing
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encodings = process_batch_parallel(self.tokenizer, batch_texts, self.max_length)
                
                # Store in memmap arrays
                self.input_ids[i:i + len(batch_texts)] = encodings['input_ids']
                self.attention_mask[i:i + len(batch_texts)] = encodings['attention_mask']
                self.labels[i:i + len(batch_texts)] = labels[i:i + len(batch_texts)]
                
                # Force memory cleanup
                gc.collect()
                
            logger.info(f"Created memory-mapped dataset with {self.total_rows} examples")
            
        except Exception as e:
            logger.error(f"Error creating memory-mapped arrays: {str(e)}")
            self._cleanup_cache()
            raise
        
    def _load_memmap_arrays(self):
        """Load existing memory-mapped arrays."""
        try:
            # Load shapes from metadata file
            shapes = np.load(f"{self.cache_prefix}_shapes.npy", allow_pickle=True).item()
            
            # Load arrays with their shapes
            self.input_ids = load_memmap_array(f"{self.cache_prefix}_input_ids.npy", shapes['input_ids'], dtype=np.int64)
            self.attention_mask = load_memmap_array(f"{self.cache_prefix}_attention_mask.npy", shapes['attention_mask'], dtype=np.int64)
            self.labels = load_memmap_array(f"{self.cache_prefix}_labels.npy", shapes['labels'], dtype=np.int64)
            self.total_rows = shapes['input_ids'][0]  # First dimension is number of rows
            
        except Exception as e:
            logger.error(f"Error loading memory-mapped arrays: {str(e)}")
            raise
    
    def __len__(self) -> int:
        """Return the total size of the dataset."""
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        # Convert split-relative index to absolute index
        idx = idx + self.start_idx
        
        # Create tensors on CPU with pinned memory
        item = {
            'input_ids': tensor_manager.create_cpu_tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': tensor_manager.create_cpu_tensor(self.attention_mask[idx], dtype=torch.long)
        }
        
        if self.labels[idx] != -1:  # If we have a valid label
            item['labels'] = tensor_manager.create_cpu_tensor(self.labels[idx], dtype=torch.long)
        
        return item

def create_dataloaders(
    data_path: Path,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    batch_size: int,
    train_ratio: float = 0.9,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, CSVDataset, CSVDataset]:
    """Create train and validation dataloaders.
    
    Args:
        data_path: Path to data file
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size
        train_ratio: Ratio of data to use for training
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Create datasets
    train_dataset = CSVDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split='train',
        train_ratio=train_ratio
    )
    val_dataset = CSVDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split='val',
        train_ratio=train_ratio
    )
    
    # Create dataloaders using manager
    from simpler_fine_bert.common.dataloader_manager import dataloader_manager
    
    train_loader = dataloader_manager.create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = dataloader_manager.create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset, val_dataset
