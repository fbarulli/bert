"""Root package initialization."""

# Common utilities
from simpler_fine_bert.common.config_utils import load_config
from simpler_fine_bert.common.utils import setup_logging, seed_everything, create_optimizer

# Manager access
from simpler_fine_bert.common.managers.amp_manager import amp_manager as get_amp_manager
from simpler_fine_bert.common.managers.batch_manager import batch_manager as get_batch_manager
from simpler_fine_bert.common.managers.cuda_manager import cuda_manager as get_cuda_manager
from simpler_fine_bert.common.managers.data_manager import data_manager as get_data_manager
from simpler_fine_bert.common.managers.dataloader_manager import dataloader_manager as get_dataloader_manager
from simpler_fine_bert.common.managers.directory_manager import directory_manager as get_directory_manager
from simpler_fine_bert.common.managers.metrics_manager import metrics_manager as get_metrics_manager
from simpler_fine_bert.common.managers.model_manager import model_manager as get_model_manager
from simpler_fine_bert.common.managers.parameter_manager import parameter_manager as get_parameter_manager
from simpler_fine_bert.common.managers.resource_manager import resource_manager as get_resource_manager
from simpler_fine_bert.common.managers.storage_manager import storage_manager as get_storage_manager
from simpler_fine_bert.common.managers.tensor_manager import tensor_manager as get_tensor_manager
from simpler_fine_bert.common.managers.tokenizer_manager import tokenizer_manager as get_tokenizer_manager
from simpler_fine_bert.common.managers.worker_manager import worker_manager as get_worker_manager

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
    
    # Managers
    'get_amp_manager',
    'get_batch_manager',
    'get_cuda_manager',
    'get_data_manager',
    'get_dataloader_manager',
    'get_directory_manager',
    'get_metrics_manager',
    'get_model_manager',
    'get_parameter_manager',
    'get_resource_manager',
    'get_storage_manager',
    'get_tensor_manager',
    'get_tokenizer_manager',
    'get_worker_manager',
    
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
