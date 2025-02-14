"""Manager access layer to prevent circular imports."""

from __future__ import annotations

def get_amp_manager():
    from simpler_fine_bert.common.amp_manager import amp_manager
    return amp_manager

def get_batch_manager():
    from simpler_fine_bert.common.batch_manager import batch_manager
    return batch_manager

def get_cuda_manager():
    from simpler_fine_bert.common.cuda_manager import cuda_manager
    return cuda_manager

def get_data_manager():
    from simpler_fine_bert.common.data_manager import data_manager
    return data_manager

def get_dataloader_manager():
    from simpler_fine_bert.common.dataloader_manager import dataloader_manager
    return dataloader_manager

def get_directory_manager():
    from simpler_fine_bert.common.directory_manager import directory_manager
    return directory_manager

def get_metrics_manager():
    from simpler_fine_bert.common.metrics_manager import metrics_manager
    return metrics_manager

def get_model_manager():
    from simpler_fine_bert.common.model_manager import model_manager
    return model_manager

def get_parameter_manager():
    from simpler_fine_bert.common.parameter_manager import parameter_manager
    return parameter_manager

def get_resource_manager():
    from simpler_fine_bert.common.resource_manager import resource_manager
    return resource_manager

def get_storage_manager():
    from simpler_fine_bert.common.storage_manager import storage_manager
    return storage_manager

def get_tensor_manager():
    from simpler_fine_bert.common.tensor_manager import tensor_manager
    return tensor_manager

def get_tokenizer_manager():
    from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
    return tokenizer_manager

def get_worker_manager():
    from simpler_fine_bert.common.worker_manager import worker_manager
    return worker_manager

__all__ = [
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
    'get_worker_manager'
]
