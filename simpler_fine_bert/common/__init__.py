"""Common utilities and components."""

# Direct imports that don't cause circular dependencies
from simpler_fine_bert.common.base_manager import BaseManager
from simpler_fine_bert.common.utils import (
    setup_logging,
    seed_everything,
    create_optimizer,
    measure_memory,
    clear_memory
)

# Lazy imports for everything else
def get_base_trainer():
    from simpler_fine_bert.common.base_trainer import BaseTrainer
    return BaseTrainer

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

def get_metrics_logger():
    from simpler_fine_bert.common.metrics_logger import metrics_logger
    return metrics_logger

def get_model_manager():
    from simpler_fine_bert.common.model_manager import model_manager
    return model_manager

def get_optuna_manager():
    from simpler_fine_bert.common.optuna_manager import OptunaManager
    return OptunaManager

def get_parameter_manager():
    from simpler_fine_bert.common.parameter_manager import parameter_manager
    return parameter_manager

def get_resource_manager():
    from simpler_fine_bert.common.resource_manager import resource_manager
    return resource_manager

def get_storage_manager():
    from simpler_fine_bert.common.storage_manager import storage_manager
    return storage_manager

def get_tokenizer_manager():
    from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
    return tokenizer_manager

def get_wandb_manager():
    from simpler_fine_bert.common.wandb_manager import WandbManager
    return WandbManager

def get_worker_manager():
    from simpler_fine_bert.common.worker_manager import worker_manager
    return worker_manager

# Study components
def get_study_components():
    from simpler_fine_bert.common.study.study_config import StudyConfig
    from simpler_fine_bert.common.study.study_storage import StudyStorage
    from simpler_fine_bert.common.study.trial_analyzer import TrialAnalyzer
    from simpler_fine_bert.common.study.trial_executor import TrialExecutor
    from simpler_fine_bert.common.study.trial_state_manager import TrialStateManager
    from simpler_fine_bert.common.study.parameter_suggester import ParameterSuggester
    from simpler_fine_bert.common.study.parallel_study import ParallelStudy
    from simpler_fine_bert.common.study.objective_factory import ObjectiveFactory
    return {
        'StudyConfig': StudyConfig,
        'StudyStorage': StudyStorage,
        'TrialAnalyzer': TrialAnalyzer,
        'TrialExecutor': TrialExecutor,
        'TrialStateManager': TrialStateManager,
        'ParameterSuggester': ParameterSuggester,
        'ParallelStudy': ParallelStudy,
        'ObjectiveFactory': ObjectiveFactory
    }

# Process components
def get_process_components():
    from simpler_fine_bert.common.process.process_init import initialize_process
    from simpler_fine_bert.common.process.process_utils import (
        set_process_name,
        get_process_name,
        is_main_process
    )
    return {
        'initialize_process': initialize_process,
        'set_process_name': set_process_name,
        'get_process_name': get_process_name,
        'is_main_process': is_main_process
    }

# Resource components
def get_resource_components():
    from simpler_fine_bert.common.resource.resource_factory import ResourceFactory
    from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
    return {
        'ResourceFactory': ResourceFactory,
        'ResourceInitializer': ResourceInitializer
    }

# Scheduler components
def get_scheduler_components():
    from simpler_fine_bert.common.scheduler.dynamic_scheduler import (
        WarmupCosineScheduler,
        WarmupLinearScheduler,
        create_scheduler,
        get_scheduler_config
    )
    return {
        'WarmupCosineScheduler': WarmupCosineScheduler,
        'WarmupLinearScheduler': WarmupLinearScheduler,
        'create_scheduler': create_scheduler,
        'get_scheduler_config': get_scheduler_config
    }

__all__ = [
    # Base
    'BaseManager',
    'get_base_trainer',
    
    # Managers
    'get_amp_manager',
    'get_batch_manager',
    'get_cuda_manager',
    'get_data_manager',
    'get_dataloader_manager',
    'get_directory_manager',
    'get_metrics_logger',
    'get_model_manager',
    'get_optuna_manager',
    'get_parameter_manager',
    'get_resource_manager',
    'get_storage_manager',
    'get_tokenizer_manager',
    'get_wandb_manager',
    'get_worker_manager',
    
    # Study
    'get_study_components',
    
    # Process
    'get_process_components',
    
    # Resource
    'get_resource_components',
    
    # Scheduler
    'get_scheduler_components',
    
    # Utils
    'setup_logging',
    'seed_everything',
    'create_optimizer',
    'measure_memory',
    'clear_memory'
]
