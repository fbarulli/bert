"""Process utility functions for managing process names, IDs, and priorities."""

from __future__ import annotations
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

def set_process_name(name: str) -> None:
    """Set process name.
    
    Args:
        name: The name to set for the current process
    """
    try:
        import setproctitle
        setproctitle.setproctitle(name)
        logger.info(f"Set process name to: {name}")
    except ImportError:
        logger.warning("setproctitle not available, process name not set")

def get_process_name() -> str:
    """Get current process name.
    
    Returns:
        The current process name, or a default if not set
    """
    try:
        import setproctitle
        return setproctitle.getproctitle()
    except ImportError:
        return f"Process-{os.getpid()}"

def is_main_process() -> bool:
    """Check if this is the main process.
    
    Returns:
        True if this is the main process, False otherwise
    """
    return os.getpid() == os.getppid()

def get_process_id() -> int:
    """Get current process ID.
    
    Returns:
        The current process ID
    """
    return os.getpid()

def get_parent_process_id() -> int:
    """Get parent process ID.
    
    Returns:
        The parent process ID
    """
    return os.getppid()

def set_process_priority(priority: int) -> None:
    """Set process priority (nice value).
    
    Args:
        priority: The nice value to set (-20 to 19, lower is higher priority)
    """
    try:
        os.nice(priority)
        logger.info(f"Set process priority to: {priority}")
    except OSError as e:
        logger.warning(f"Failed to set process priority: {e}")

__all__ = [
    'set_process_name',
    'get_process_name',
    'is_main_process',
    'get_process_id',
    'get_parent_process_id',
    'set_process_priority'
]
