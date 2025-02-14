from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Union, List, Tuple, TypedDict, Literal
import optuna
from optuna.trial import Trial

logger = logging.getLogger(__name__)

class FloatParam(TypedDict):
    min: float
    max: float
    log: bool

class FixedParam(TypedDict):
    value: Union[int, float, bool]

class ParameterSuggester:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stage = config.get('stage', 'embedding')
        
        self.common_float_params: Dict[str, FloatParam] = {
            'learning_rate': {'min': 1e-5, 'max': 5e-5, 'log': True},  # Narrowed range
            'weight_decay': {'min': 0.0, 'max': 0.3, 'log': False},    # From config
            'warmup_ratio': {'min': 0.0, 'max': 0.2, 'log': False}     # From config
        }
        
        self.embedding_float_params: Dict[str, FloatParam] = {
            'embedding_mask_probability': {'min': 0.15, 'max': 0.15, 'log': False}  # Fixed from config
        }
        
        self.float_params = self.common_float_params.copy()
        if self.stage == 'embedding':
            self.float_params.update(self.embedding_float_params)
            self.fixed_params = {
                'batch_size': {'value': 16},  # From config
                'max_length': {'value': 512},  # From config
                'max_predictions': {'value': 20}  # From config
            }
        else:
            self.fixed_params = {
                'batch_size': {'value': 16}
            }
    
    def suggest_parameters(
        self,
        trial: Trial,
        historical_ranges: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        try:
            config = self.config.copy()
            
            # Add fixed parameters
            for param_name, param_info in self.fixed_params.items():
                config[param_name] = param_info['value']
            
            # Suggest float parameters
            for param_name, param_info in self.float_params.items():
                value = trial.suggest_float(
                    param_name,
                    param_info['min'],
                    param_info['max'],
                    log=param_info['log']
                )
                config[param_name] = value
            
            return config
            
        except Exception as e:
            logger.error(f"Error suggesting parameters: {str(e)}")
            raise
    
    def check_constraints(
        self,
        params: Dict[str, Any],
        max_memory_gb: float,
        max_compute: Optional[int] = None
    ) -> bool:
        try:
            memory_usage = self._estimate_memory_usage(params)
            if memory_usage > max_memory_gb:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking constraints: {str(e)}")
            raise
    
    def _estimate_memory_usage(self, params: Dict[str, Any]) -> float:
        try:
            hidden_dim = 768  # Fixed for bert-base
            batch_size = params.get('batch_size', 16)
            sequence_length = params.get('max_length', 512)
            
            num_parameters = hidden_dim * hidden_dim * 12  # 12 layers
            activation_memory = batch_size * sequence_length * hidden_dim * 4  # 4 bytes per float
            
            total_memory = (num_parameters * 4 + activation_memory) / (1024 ** 3)  # Convert to GB
            total_memory *= 1.5  # Add buffer for optimizer states
            
            return total_memory
        except Exception as e:
            logger.error(f"Error estimating memory usage: {str(e)}")
            raise
