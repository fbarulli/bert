from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple
import torch
from transformers import PreTrainedTokenizerFast
import os

from simpler_fine_bert.common.data import CSVDataset
from simpler_fine_bert.embedding.masking import SpanMaskingModule

logger = logging.getLogger(__name__)

class EmbeddingDataset(CSVDataset):
    """Dataset for learning embeddings through masked token prediction."""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        split: str = 'train',
        train_ratio: float = 0.9,
        mask_prob: float = 0.15,
        max_predictions: int = 20,
        max_span_length: int = 1
    ):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            train_ratio=train_ratio
        )
        self.mask_prob = mask_prob
        self.max_predictions = max_predictions
        self.max_span_length = max_span_length
        # Initialize masking module
        self.masking_module = SpanMaskingModule(
            tokenizer=self.tokenizer,
            mask_prob=self.mask_prob,
            max_span_length=self.max_span_length,
            max_predictions=self.max_predictions,
            worker_id=os.getpid()
        )
    
    def _mask_tokens(self, item: Dict[str, torch.Tensor], idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masking using the SpanMaskingModule.
        
        Args:
            item: Dictionary containing input tensors
            idx: Index into the dataset
            
        Returns:
            Tuple of (masked inputs, labels) both of shape [seq_len]
        """
        if item['input_ids'].dim() != 1:
            raise ValueError(f"Expected 1D input tensor, got shape: {item['input_ids'].shape}")
            
        # Apply masking using the module (returns CPU tensors)
        masked_inputs, labels = self.masking_module(item)
        
        return masked_inputs, labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset with masking applied.
        
        Args:
            idx: Index into the dataset split
            
        Returns:
            Dictionary containing masked input_ids, attention_mask, and labels
        """
        # Get base item from parent class
        item = super().__getitem__(idx)
        
        # Apply masking with validation check
        input_ids, embedding_labels = self._mask_tokens(item, idx)
        
        # Verify masking ratio
        mask = embedding_labels != -100
        mask_ratio = mask.sum().item() / len(embedding_labels)
        if mask_ratio < 0.1 or mask_ratio > 0.2:  # Allow some variance around 0.15
            logger.warning(
                f"Unusual masking ratio {mask_ratio:.2%} at index {idx}\n"
                f"- Total tokens: {len(embedding_labels)}\n"
                f"- Masked tokens: {mask.sum().item()}"
            )
        
        item['input_ids'] = input_ids
        item['labels'] = embedding_labels
        
        return item

__all__ = ['EmbeddingDataset']
