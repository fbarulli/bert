"""Common utilities and components."""

# Direct imports that don't cause circular dependencies
from simpler_fine_bert.common.managers.base_manager import BaseManager
from simpler_fine_bert.common.utils import (
    setup_logging,
    seed_everything,
    create_optimizer,
    measure_memory,
    clear_memory
)

# Import all manager getters
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

# Base trainer getter
def get_base_trainer():
    from simpler_fine_bert.common.base_trainer import BaseTrainer
    return BaseTrainer

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
    'get_metrics_manager',
    'get_model_manager',
    'get_parameter_manager',
    'get_resource_manager',
    'get_storage_manager',
    'get_tensor_manager',
    'get_tokenizer_manager',
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
