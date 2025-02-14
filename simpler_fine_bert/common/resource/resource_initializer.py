from __future__ import annotations

import logging
import os
import torch
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, List, Type, Set
from dataclasses import dataclass, field

from simpler_fine_bert.common.base_manager import BaseManager
from simpler_fine_bert.common.cuda_manager import cuda_manager
from simpler_fine_bert.common.tensor_manager import tensor_manager
from simpler_fine_bert.common.batch_manager import batch_manager
from simpler_fine_bert.common.metrics_manager import metrics_manager
from simpler_fine_bert.common.amp_manager import amp_manager
from simpler_fine_bert.common.dataloader_manager import dataloader_manager
from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager

logger = logging.getLogger(__name__)

@dataclass
class ManagerDependency:
    """Represents a manager and its dependencies."""
    manager: BaseManager
    depends_on: List[Type[BaseManager]] = field(default_factory=list)
    description: str = ""
    initialized: bool = False

class ResourceInitializer:
    """Centralizes process-local resource initialization with dependency tracking."""
    
    # Define initialization order and dependencies
    _manager_dependencies = {
        cuda_manager.__class__: ManagerDependency(
            manager=cuda_manager,
            description="CUDA system initialization"
        ),
        tensor_manager.__class__: ManagerDependency(
            manager=tensor_manager,
            depends_on=[cuda_manager.__class__],
            description="Tensor operations"
        ),
        batch_manager.__class__: ManagerDependency(
            manager=batch_manager,
            depends_on=[tensor_manager.__class__],
            description="Batch processing"
        ),
        metrics_manager.__class__: ManagerDependency(
            manager=metrics_manager,
            depends_on=[tensor_manager.__class__],
            description="Metrics tracking"
        ),
        amp_manager.__class__: ManagerDependency(
            manager=amp_manager,
            depends_on=[cuda_manager.__class__],
            description="Automatic mixed precision"
        ),
        tokenizer_manager.__class__: ManagerDependency(
            manager=tokenizer_manager,
            description="Tokenizer management"
        ),
        dataloader_manager.__class__: ManagerDependency(
            manager=dataloader_manager,
            depends_on=[cuda_manager.__class__, tensor_manager.__class__, tokenizer_manager.__class__],
            description="Data loading"
        )
    }

    @classmethod
    def _verify_dependencies(cls) -> None:
        """Verify that manager dependencies form a valid DAG."""
        visited: Set[Type[BaseManager]] = set()
        temp_visited: Set[Type[BaseManager]] = set()
        
        def visit(manager_cls: Type[BaseManager]) -> None:
            if manager_cls in temp_visited:
                cycle = [m.__name__ for m in temp_visited]
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
            if manager_cls not in visited:
                temp_visited.add(manager_cls)
                for dep in cls._manager_dependencies[manager_cls].depends_on:
                    if dep not in cls._manager_dependencies:
                        raise ValueError(f"Unknown dependency {dep.__name__}")
                    visit(dep)
                temp_visited.remove(manager_cls)
                visited.add(manager_cls)
        
        for manager_cls in cls._manager_dependencies:
            if manager_cls not in visited:
                visit(manager_cls)

    @classmethod
    def initialize_process(cls) -> int:
        """Initialize all process-local resources in dependency order.
        
        Returns:
            Current process ID
        """
        current_pid = os.getpid()
        parent_pid = os.getppid()
        
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
            # Verify dependencies are valid
            cls._verify_dependencies()
            
            # Reset initialization state
            for dep in cls._manager_dependencies.values():
                dep.initialized = False
            
            # Initialize CUDA first
            if not torch.cuda.is_initialized():
                cuda_dep = cls._manager_dependencies[cuda_manager.__class__]
                cls._initialize_manager(cuda_dep)
            else:
                logger.info("CUDA already initialized by parent process")
                cls._manager_dependencies[cuda_manager.__class__].initialized = True

            # Initialize remaining managers in dependency order
            cls._initialize_remaining_managers()
            
            # Verify all managers were initialized
            uninitialized = [
                manager_cls.__name__ 
                for manager_cls, dep in cls._manager_dependencies.items() 
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
            # Verify dependencies are initialized
            for parent_cls in dep.depends_on:
                parent_dep = cls._manager_dependencies[parent_cls]
                if not parent_dep.initialized:
                    raise RuntimeError(
                        f"Cannot initialize {dep.manager.__class__.__name__} - "
                        f"dependency {parent_cls.__name__} not initialized"
                    )
            
            # Initialize manager
            dep.manager.ensure_initialized()
            
            # Verify initialization
            if not dep.manager.is_initialized():
                raise RuntimeError(f"{dep.manager.__class__.__name__} failed to initialize properly")
            
            dep.initialized = True
            logger.info(f"Initialized {dep.manager.__class__.__name__} ({dep.description})")
            
        except Exception as e:
            logger.error(f"Failed to initialize {dep.manager.__class__.__name__}: {str(e)}")
            raise

    @classmethod
    def _initialize_remaining_managers(cls) -> None:
        """Initialize remaining managers in dependency order."""
        while True:
            progress = False
            
            for manager_cls, dep in cls._manager_dependencies.items():
                if not dep.initialized and all(
                    cls._manager_dependencies[parent_cls].initialized 
                    for parent_cls in dep.depends_on
                ):
                    cls._initialize_manager(dep)
                    progress = True
            
            if not progress:
                break

    @classmethod
    def cleanup_process(cls) -> None:
        """Clean up all process resources in reverse dependency order."""
        # Clean up in reverse initialization order
        for manager_cls, dep in reversed(list(cls._manager_dependencies.items())):
            if dep.initialized:
                try:
                    # Use BaseManager's cleanup mechanism
                    dep.manager.cleanup_all()
                    dep.initialized = False
                    logger.info(f"Cleaned up {manager_cls.__name__}")
                except Exception as e:
                    logger.error(f"Error cleaning up {manager_cls.__name__}: {str(e)}")
