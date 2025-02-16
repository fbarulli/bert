"""Root package initialization with lazy loading."""

def get_common_utils():
    """Get common utilities with lazy loading."""
    from simpler_fine_bert.common.config_utils import load_config
    from simpler_fine_bert.common.utils import setup_logging, seed_everything, create_optimizer
    return {
        'load_config': load_config,
        'setup_logging': setup_logging,
        'seed_everything': seed_everything,
        'create_optimizer': create_optimizer
    }

def get_managers():
    """Get manager access functions with lazy loading."""
    from simpler_fine_bert.common.managers import (
        get_amp_manager,
        get_batch_manager,
        get_cuda_manager,
        get_data_manager,
        get_dataloader_manager,
        get_directory_manager,
        get_metrics_manager,
        get_model_manager,
        get_parameter_manager,
        get_resource_manager,
        get_storage_manager,
        get_tensor_manager,
        get_tokenizer_manager,
        get_worker_manager
    )
    return {
        'get_amp_manager': get_amp_manager,
        'get_batch_manager': get_batch_manager,
        'get_cuda_manager': get_cuda_manager,
        'get_data_manager': get_data_manager,
        'get_dataloader_manager': get_dataloader_manager,
        'get_directory_manager': get_directory_manager,
        'get_metrics_manager': get_metrics_manager,
        'get_model_manager': get_model_manager,
        'get_parameter_manager': get_parameter_manager,
        'get_resource_manager': get_resource_manager,
        'get_storage_manager': get_storage_manager,
        'get_tensor_manager': get_tensor_manager,
        'get_tokenizer_manager': get_tokenizer_manager,
        'get_worker_manager': get_worker_manager
    }

def get_embedding_components():
    """Get embedding components with lazy loading."""
    from simpler_fine_bert.embedding import get_embedding_components
    return get_embedding_components()

def get_classification_components():
    """Get classification components with lazy loading."""
    from simpler_fine_bert.classification import get_classification_components
    return get_classification_components()

__all__ = [
    # Lazy loading getters
    'get_common_utils',
    'get_managers',
    'get_embedding_components',
    'get_classification_components'
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
