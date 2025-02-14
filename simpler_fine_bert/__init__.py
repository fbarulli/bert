# Common utilities
from simpler_fine_bert.common.config_utils import load_config
from simpler_fine_bert.common.utils import setup_logging, seed_everything, create_optimizer

# Embedding components
from simpler_fine_bert.embedding import (
    EmbeddingBert,
    EmbeddingTrainer,
    train_embeddings,
    validate_embeddings,
    EmbeddingDataset,
    MaskingModule,
    WholeWordMaskingModule,
    SpanMaskingModule,
    SimCSEEmbeddingLoss
)

# Classification components
from simpler_fine_bert.classification import (
    ClassificationBert,
    run_classification_optimization,
    train_final_model,
    CSVDataset,
    FocalLoss
)

__all__ = [
    # Common
    'load_config',
    'setup_logging',
    'seed_everything',
    'create_optimizer',
    
    # Embedding
    'EmbeddingBert',
    'EmbeddingTrainer',
    'train_embeddings',
    'validate_embeddings',
    'EmbeddingDataset',
    'MaskingModule',
    'WholeWordMaskingModule',
    'SpanMaskingModule',
    'SimCSEEmbeddingLoss',
    
    # Classification
    'ClassificationBert',
    'run_classification_optimization',
    'train_final_model',
    'CSVDataset',
    'FocalLoss'
]

# Configuration key mappings
CONFIG_KEY_MAPPINGS = {
    'learning_rate': 'training',
    'batch_size': 'training',
    'num_epochs': 'training',
    'warmup_steps': 'training',
    'max_grad_norm': 'training',
    'weight_decay': 'training',
    'hidden_dropout_prob': 'training',
    'attention_probs_dropout_prob': 'training',
    'num_workers': 'training',
    'max_length': 'model',
    'num_labels': 'model',
    'embedding_mask_probability': 'data',
    'max_span_length': 'data'
}

# Default configuration sections
DEFAULT_CONFIG_SECTIONS = {
    'model': [
        'name', 'type', 'max_length', 'num_labels'
    ],
    'data': [
        'embedding_mask_probability', 'max_span_length', 'csv_path',
        'train_ratio', 'max_length', 'max_predictions', 'num_workers'
    ],
    'training': [
        'learning_rate', 'batch_size', 'num_epochs', 'warmup_steps',
        'max_grad_norm', 'weight_decay', 'hidden_dropout_prob',
        'attention_probs_dropout_prob', 'num_workers'
    ],
    'output': [
        'dir', 'log_level'
    ],
    'resources': [
        'max_memory_gb', 'num_gpus'
    ]
}
