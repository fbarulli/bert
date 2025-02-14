"""Process initialization utilities."""

import logging
import os
from typing import Tuple

from simpler_fine_bert.common.process.initialization import (
    initialize_process,
    initialize_worker,
    cleanup_process
)

logger = logging.getLogger(__name__)

# Re-export initialization functions
initialize_process_resources = initialize_worker
cleanup_process_resources = cleanup_process

__all__ = [
    'initialize_process',
    'initialize_process_resources',
    'cleanup_process_resources'
]
