"""Process initialization utilities."""

import logging
import os
from typing import Tuple

from simpler_fine_bert.resource_initializer import ResourceInitializer

logger = logging.getLogger(__name__)

def initialize_process() -> Tuple[int, int]:
    """Initialize process-specific settings.
    
    This should be called at the start of any new process to ensure proper setup.
    
    Returns:
        Tuple of (current_pid, parent_pid)
    """
    current_pid = ResourceInitializer.initialize_process()
    parent_pid = os.getppid()
    
    return current_pid, parent_pid
