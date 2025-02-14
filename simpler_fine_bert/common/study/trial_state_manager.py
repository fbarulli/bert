from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import optuna

logger = logging.getLogger(__name__)

class TrialStatus(Enum):
    INITIALIZING = "initializing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class TrialStateManager:
    def __init__(
        self,
        trial: optuna.Trial,
        max_memory_gb: float = 12.0
    ):
        self.trial = trial
        self.trial_number = trial.number
        self.max_memory_gb = max_memory_gb
        self.status = TrialStatus.INITIALIZING
        self.start_time = datetime.now()
        
        # Set initial state
        self.trial.set_user_attr('current_status', self.status.value)
        self.trial.set_user_attr('start_time', self.start_time.isoformat())
        self.trial.set_user_attr('completed', False)
    
    def update_state(self, new_status: TrialStatus, metrics: Optional[Dict[str, Any]] = None) -> None:
        old_status = self.status
        self.status = new_status
        
        # Update trial attributes
        self.trial.set_user_attr('current_status', new_status.value)
        if metrics:
            for key, value in metrics.items():
                self.trial.set_user_attr(key, value)
        
        # Handle completion
        if new_status == TrialStatus.COMPLETED:
            self.trial.set_user_attr('completed', True)
            self.trial.set_user_attr('end_time', datetime.now().isoformat())
            duration = (datetime.now() - self.start_time).total_seconds()
            self.trial.set_user_attr('duration_seconds', duration)
            
        # Handle failure
        elif new_status == TrialStatus.FAILED:
            self.trial.set_user_attr('completed', False)
            self.trial.set_user_attr('end_time', datetime.now().isoformat())
            duration = (datetime.now() - self.start_time).total_seconds()
            self.trial.set_user_attr('duration_seconds', duration)
