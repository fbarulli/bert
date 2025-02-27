"""Embedding module for BERT model training."""

from simpler_fine_bert.embedding.model import EmbeddingBert
from simpler_fine_bert.embedding.embedding_trainer import EmbeddingTrainer
from simpler_fine_bert.embedding.embedding_training import train_embeddings, validate_embeddings
from simpler_fine_bert.embedding.dataset import EmbeddingDataset
from simpler_fine_bert.embedding.masking import (
    MaskingModule,
    WholeWordMaskingModule,
    SpanMaskingModule
)
from simpler_fine_bert.embedding.losses import SimCSEEmbeddingLoss

__all__ = [
    'EmbeddingBert',
    'EmbeddingTrainer',
    'train_embeddings',
    'validate_embeddings',
    'EmbeddingDataset',
    'MaskingModule',
    'WholeWordMaskingModule',
    'SpanMaskingModule',
    'SimCSEEmbeddingLoss'
]
