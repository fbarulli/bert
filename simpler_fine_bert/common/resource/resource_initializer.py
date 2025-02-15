"""Resource initialization with dependency tracking."""

from __future__ import annotations
import logging
import os
import torch
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, List, Type, Set
from dataclasses import dataclass, field

from simpler_fine_bert.common.managers.base_manager import BaseManager
from simpler_fine_bert.common.cuda_utils import (
    is_cuda_available,
    clear_cuda_memory,
    reset_cuda_stats
)

logger = logging.getLogger(__name__)

@dataclass
class ManagerDependency:
    """Represents a manager and its dependencies."""
    manager_cls: Type[BaseManager]
    depends_on: List[Type[BaseManager]] = field(default_factory=list)
    description: str = ""
    initialized: bool = False

class ResourceInitializer:
    """Centralizes process-local resource initialization with dependency tracking."""
    
    # Class-level storage for process config
    _config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def _get_manager_dependencies(cls) -> Dict[Type[BaseManager], ManagerDependency]:
        """Get manager dependencies lazily to avoid circular imports."""
        # Import managers at runtime to avoid circular imports
        from simpler_fine_bert.common.managers import (
            get_cuda_manager,
            get_tensor_manager,
            get_batch_manager,
            get_metrics_manager,
            get_amp_manager,
            get_tokenizer_manager,
            get_dataloader_manager,
            get_data_manager
        )
        
        cuda_manager = get_cuda_manager()
        tensor_manager = get_tensor_manager()
        batch_manager = get_batch_manager()
        metrics_manager = get_metrics_manager()
        amp_manager = get_amp_manager()
        tokenizer_manager = get_tokenizer_manager()
        dataloader_manager = get_dataloader_manager()
        data_manager = get_data_manager()
        
        # Define dependencies using manager classes
        return {
            cuda_manager.__class__: ManagerDependency(
                manager_cls=cuda_manager.__class__,
                description="CUDA system initialization"
            ),
            amp_manager.__class__: ManagerDependency(
                manager_cls=amp_manager.__class__,
                depends_on=[cuda_manager.__class__],
                description="Automatic mixed precision"
            ),
            tensor_manager.__class__: ManagerDependency(
                manager_cls=tensor_manager.__class__,
                depends_on=[cuda_manager.__class__, amp_manager.__class__],
                description="Tensor operations"
            ),
            batch_manager.__class__: ManagerDependency(
                manager_cls=batch_manager.__class__,
                depends_on=[tensor_manager.__class__],
                description="Batch processing"
            ),
            metrics_manager.__class__: ManagerDependency(
                manager_cls=metrics_manager.__class__,
                depends_on=[tensor_manager.__class__],
                description="Metrics tracking"
            ),
            tokenizer_manager.__class__: ManagerDependency(
                manager_cls=tokenizer_manager.__class__,
                description="Tokenizer management"
            ),
            dataloader_manager.__class__: ManagerDependency(
                manager_cls=dataloader_manager.__class__,
                depends_on=[cuda_manager.__class__, tensor_manager.__class__, tokenizer_manager.__class__],
                description="Data loading"
            ),
            data_manager.__class__: ManagerDependency(
                manager_cls=data_manager.__class__,
                depends_on=[dataloader_manager.__class__, tokenizer_manager.__class__],
                description="Data management"
            )
        }

    @classmethod
    def _verify_dependencies(cls, dependencies: Dict[Type[BaseManager], ManagerDependency]) -> None:
        """Verify that manager dependencies form a valid DAG."""
        visited: Set[Type[BaseManager]] = set()
        temp_visited: Set[Type[BaseManager]] = set()
        
        def visit(manager_cls: Type[BaseManager]) -> None:
            if manager_cls in temp_visited:
                cycle = [m.__name__ for m in temp_visited]
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
            if manager_cls not in visited:
                temp_visited.add(manager_cls)
                for dep in dependencies[manager_cls].depends_on:
                    if dep not in dependencies:
                        raise ValueError(f"Unknown dependency {dep.__name__}")
                    visit(dep)
                temp_visited.remove(manager_cls)
                visited.add(manager_cls)
        
        for manager_cls in dependencies:
            if manager_cls not in visited:
                visit(manager_cls)

    @classmethod
    def _initialize_cuda_and_amp(cls, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CUDA and AMP first."""
        # Import managers at runtime
        from simpler_fine_bert.common.managers import get_cuda_manager, get_amp_manager
        cuda_manager = get_cuda_manager()
        amp_manager = get_amp_manager()
        
        # Initialize CUDA first
        cuda_manager.ensure_initialized(config)
        if is_cuda_available():
            if torch.cuda.is_initialized():
                logger.info("CUDA initialized successfully")
                # Reset CUDA stats
                clear_cuda_memory()
                reset_cuda_stats()
            else:
                logger.warning("CUDA available but not initialized")
        else:
            logger.info("CUDA not available, running on CPU")
            
        # Initialize AMP right after CUDA
        amp_manager.ensure_initialized(config)
        logger.info("AMP initialized successfully")

    @classmethod
    def initialize_process(cls, config: Optional[Dict[str, Any]] = None) -> int:
        """Initialize all process-local resources in dependency order.
        
        Args:
            config: Optional configuration dictionary
        
        Returns:
            Current process ID
        """
        current_pid = os.getpid()
        parent_pid = os.getppid()
        
        # Store config for this process
        cls._config = config
        
        # Set multiprocessing start method if this is not a worker
        is_worker = mp.parent_process() is not None
        if not is_worker:
            try:
                mp.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn'")
            except RuntimeError as e:
                logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}")

        logger.info(f"Initializing process resources (PID: {current_pid}, Parent PID: {parent_pid})")
        
        try:
            # Get dependencies lazily
            dependencies = cls._get_manager_dependencies()
            
            # Verify dependencies are valid
            cls._verify_dependencies(dependencies)
            
            # Reset initialization state
            for dep in dependencies.values():
                dep.initialized = False
            
            # Initialize CUDA and AMP first
            cls._initialize_cuda_and_amp(config)
            
            # Import managers at runtime
            from simpler_fine_bert.common.managers import get_cuda_manager, get_amp_manager
            cuda_manager = get_cuda_manager()
            amp_manager = get_amp_manager()
            
            # Mark CUDA and AMP as initialized
            dependencies[cuda_manager.__class__].initialized = True
            dependencies[amp_manager.__class__].initialized = True
            
            # Initialize remaining managers in dependency order
            cls._initialize_remaining_managers(dependencies)
            
            # Verify all managers were initialized
            uninitialized = [
                manager_cls.__name__ 
                for manager_cls, dep in dependencies.items() 
                if not dep.initialized
            ]
            if uninitialized:
                raise RuntimeError(f"Failed to initialize managers: {', '.join(uninitialized)}")
            
            logger.info(f"All process resources initialized successfully for PID {current_pid}")
            return current_pid
            
        except Exception as e:
            logger.error(f"Failed to initialize process resources: {str(e)}")
            # Attempt cleanup on failure
            cls.cleanup_process()
            raise

    @classmethod
    def _initialize_manager(cls, dep: ManagerDependency) -> None:
        """Initialize a single manager and verify its state."""
        try:
            # Get manager instance from BaseManager registry
            manager = BaseManager._instances[dep.manager_cls]
            
            # Verify dependencies are initialized
            for parent_cls in dep.depends_on:
                parent_manager = BaseManager._instances[parent_cls]
                if not parent_manager.is_initialized():
                    raise RuntimeError(
                        f"Cannot initialize {manager.__class__.__name__} - "
                        f"dependency {parent_cls.__name__} not initialized"
                    )
            
            # Initialize manager with config
            manager.ensure_initialized(cls._config)
            
            # Verify initialization
            if not manager.is_initialized():
                raise RuntimeError(f"{manager.__class__.__name__} failed to initialize properly")
            
            dep.initialized = True
            logger.info(f"Initialized {manager.__class__.__name__} ({dep.description})")
            
        except Exception as e:
            logger.error(f"Failed to initialize {dep.manager_cls.__name__}: {str(e)}")
            raise

    @classmethod
    def _initialize_remaining_managers(cls, dependencies: Dict[Type[BaseManager], ManagerDependency]) -> None:
        """Initialize remaining managers in dependency order."""
        while True:
            progress = False
            
            for manager_cls, dep in dependencies.items():
                if not dep.initialized and all(
                    dependencies[parent_cls].initialized 
                    for parent_cls in dep.depends_on
                ):
                    cls._initialize_manager(dep)
                    progress = True
            
            if not progress:
                break

    @classmethod
    def cleanup_process(cls) -> None:
        """Clean up all process resources in reverse dependency order."""
        try:
            # Get dependencies lazily
            dependencies = cls._get_manager_dependencies()
            
            # Clean up in reverse initialization order
            for manager_cls, dep in reversed(list(dependencies.items())):
                if dep.initialized:
                    try:
                        # Get manager instance from BaseManager registry
                        manager = BaseManager._instances[dep.manager_cls]
                        # Use BaseManager's cleanup mechanism
                        manager.cleanup_all()
                        dep.initialized = False
                        logger.info(f"Cleaned up {manager_cls.__name__}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {manager_cls.__name__}: {str(e)}")
            
            # Clear stored config
            cls._config = None
            
            # Clean up CUDA memory
            if is_cuda_available():
                clear_cuda_memory()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
