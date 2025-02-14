from __future__ import annotations
import sqlite3
import contextlib
import logging
from pathlib import Path
from filelock import FileLock
import json
from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler
import shutil

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages Optuna storage and study persistence."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_path = storage_dir / 'optuna.db'
        self.lock_path = storage_dir / 'optuna.lock'
        self.history_path = storage_dir / 'trial_history.json'
        
        # Create standard directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = storage_dir / 'trials'
        self.trials_dir.mkdir(exist_ok=True)
        self.profiler_dir = storage_dir / 'profiler'
        self.profiler_dir.mkdir(exist_ok=True)
        
        self._initialize_storage()
        
        logger.info(
            f"Initialized storage manager:\n"
            f"- Storage dir: {self.storage_dir}\n"
            f"- Trials dir: {self.trials_dir}\n"
            f"- Profiler dir: {self.profiler_dir}"
        )
    
    def _initialize_storage(self):
        """Initialize SQLite storage with proper settings."""
        with contextlib.closing(sqlite3.connect(self.storage_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.commit()
    
    def create_study(self, study_name: str, sampler: TPESampler) -> optuna.Study:
        """Create or load study with proper locking."""
        storage_url = f"sqlite:///{self.storage_path}?timeout=60"
        
        with FileLock(self.lock_path):
            return optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                sampler=sampler,
                direction='minimize',
                load_if_exists=True
            )
    
    def save_history(self, study: optuna.Study):
        """Save trial history to file."""
        history = {'trials': []}
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history['trials'].append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name
                })
            
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    def get_trial_dir(self, trial_number: int) -> Path:
        """Get directory for trial outputs.
        
        Args:
            trial_number: Trial number
            
        Returns:
            Path to trial directory
        """
        trial_dir = self.trials_dir / f'trial_{trial_number}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir
    
    def get_profiler_dir(self, trial_number: Optional[int] = None) -> Path:
        """Get directory for profiler outputs.
        
        Args:
            trial_number: Optional trial number for trial-specific profiling
            
        Returns:
            Path to profiler directory
        """
        if trial_number is not None:
            profiler_dir = self.get_trial_dir(trial_number) / 'profiler'
        else:
            profiler_dir = self.profiler_dir
            
        profiler_dir.mkdir(parents=True, exist_ok=True)
        return profiler_dir
    
    def cleanup_trial(self, trial_number: int):
        """Clean up trial directory and associated files.
        
        Args:
            trial_number: Trial number to clean up
        """
        try:
            trial_dir = self.trials_dir / f'trial_{trial_number}'
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
                logger.debug(f"Cleaned up trial directory: {trial_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up trial {trial_number}: {e}")
            
    def cleanup_profiler(self, trial_number: Optional[int] = None):
        """Clean up profiler outputs.
        
        Args:
            trial_number: Optional trial number for trial-specific cleanup
        """
        try:
            if trial_number is not None:
                profiler_dir = self.get_trial_dir(trial_number) / 'profiler'
            else:
                profiler_dir = self.profiler_dir
                
            if profiler_dir.exists():
                shutil.rmtree(profiler_dir)
                profiler_dir.mkdir(exist_ok=True)
                logger.debug(f"Cleaned up profiler directory: {profiler_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up profiler: {e}")
            
    def cleanup_all(self):
        """Clean up all storage."""
        try:
            # Clean up trials
            if self.trials_dir.exists():
                shutil.rmtree(self.trials_dir)
                self.trials_dir.mkdir(exist_ok=True)
            
            # Clean up profiler
            if self.profiler_dir.exists():
                shutil.rmtree(self.profiler_dir)
                self.profiler_dir.mkdir(exist_ok=True)
            
            # Clean up database
            if self.storage_path.exists():
                self.storage_path.unlink()
                self._initialize_storage()
                
            # Clean up history
            if self.history_path.exists():
                self.history_path.unlink()
                
            logger.info("Cleaned up all storage")
            
        except Exception as e:
            logger.error(f"Error cleaning up storage: {e}")
            raise
