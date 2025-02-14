# simpler_fine_bert/config_utils.py

import logging
import yaml
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define parameter types for type checking and conversion
PARAMETER_TYPES = {
    'data': {
        'csv_path': str,
        'train_ratio': float,
        'max_length': int,
        'embedding_mask_probability': float,
        'max_predictions': int,
        'num_workers': int
    },
    'training': {
        'batch_size': int,
        'num_epochs': int,
        'num_workers': int,
        'seed': int,
        'n_jobs': int,
        'num_trials': int,
        'n_startup_trials': int,
        'learning_rate': float,
        'weight_decay': float,
        'fp16': bool,
        "optimizer_type": str,
        "save_every_epoch": int,
        'early_stopping_patience': int,
        'early_stopping_min_delta': float,
        "max_grad_norm": float
    },
    'model': {
        'name': str
    },
    'optimizer': {
        'learning_rate': float,
        'weight_decay': float,
        'adam_beta1': float,
        'adam_beta2': float,
        'adam_epsilon': float,
        'max_grad_norm': float
    },
    'output': {
        'dir': str
    },
    'study_name': str,
    'hyperparameters': dict,
    'resources': {
        'max_memory_gb': float,
        'max_time_hours': float
    },
    'scheduler': dict
}

def _convert_value(value: any, target_type: type) -> any:
    """Convert value to target type."""
    try:
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting {value} to {target_type}: {str(e)}")
        raise

def _convert_config_types(config: dict, type_map: dict) -> dict:
    """Recursively convert config values to correct types."""
    converted = {}
    for key, value in config.items():
        if isinstance(value, dict):
            if key in type_map and isinstance(type_map[key], dict):
                converted[key] = _convert_config_types(value, type_map[key])
            else:
                converted[key] = value
        elif key in type_map:
            try:
                converted[key] = _convert_value(value, type_map[key])
                logger.debug(f"Converted {key}={value} to {type(converted[key])}")
            except Exception as e:
                logger.error(f"Error converting {key}={value}: {str(e)}")
                raise
        else:
            converted[key] = value
    return converted

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with type conversion."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config = _convert_config_types(config, PARAMETER_TYPES)
        logger.info("Loaded and converted config types successfully")
        return config

    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise
