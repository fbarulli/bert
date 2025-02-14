from __future__ import annotations
import threading
import logging
import os
import weakref
from typing import Dict, Type, ClassVar

logger = logging.getLogger(__name__)

class BaseManager:
    """Base class for process-local managers with isolated storage."""
    
    # Registry to track all manager instances
    _instances: ClassVar[Dict[Type['BaseManager'], 'BaseManager']] = weakref.WeakValueDictionary()
    
    # Registry for class-specific thread-local storage
    _storage_registry: ClassVar[Dict[Type['BaseManager'], threading.local]] = {}
    
    def __new__(cls):
        """Ensure singleton instance per manager class."""
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
            # Initialize class-specific storage
            cls._storage_registry[cls] = threading.local()
        return cls._instances[cls]
    
    @property
    def _local(self) -> threading.local:
        """Get class-specific thread-local storage."""
        return self.__class__._storage_registry[self.__class__]
    
    def ensure_initialized(self) -> None:
        """Ensure manager is initialized for current process."""
        current_pid = os.getpid()
        if not hasattr(self._local, 'initialized') or self._local.pid != current_pid:
            logger.debug(f"Initializing {self.__class__.__name__} for process {current_pid}")
            self._local.pid = current_pid
            self._local.initialized = False  # Set to True only after successful initialization
            try:
                self._initialize_process_local()
                self._local.initialized = True
                logger.info(f"{self.__class__.__name__} initialized for process {current_pid}")
            except Exception as e:
                logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
                # Clean up partial initialization
                if hasattr(self._local, 'initialized'):
                    delattr(self._local, 'initialized')
                raise
    
    def _initialize_process_local(self) -> None:
        """Initialize process-local attributes. Override in subclasses."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if manager is initialized for current process."""
        return (
            hasattr(self._local, 'initialized') and 
            self._local.initialized and 
            self._local.pid == os.getpid()
        )
    
    @classmethod
    def cleanup_all(cls) -> None:
        """Clean up all manager instances."""
        for manager_cls, storage in cls._storage_registry.items():
            try:
                if hasattr(storage, 'initialized'):
                    delattr(storage, 'initialized')
                logger.debug(f"Cleaned up storage for {manager_cls.__name__}")
            except Exception as e:
                logger.error(f"Error cleaning up {manager_cls.__name__}: {str(e)}")
