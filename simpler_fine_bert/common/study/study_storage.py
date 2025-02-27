# simpler_fine_bert/study_storage.py

import logging
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class StudyStorage:
    """Handles database operations and trial history for Optuna studies."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.storage_path = base_dir / 'optuna.db'
        self.history_path = base_dir / 'trial_history.json'
        
        # Initialize database
        self._init_database_schema()
    
    def _init_database_schema(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Create studies table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id INTEGER PRIMARY KEY,
                    study_name TEXT UNIQUE,
                    direction TEXT,
                    system_attrs TEXT,
                    user_attrs TEXT,
                    datetime_start DATETIME,
                    datetime_complete DATETIME
                )
            """)

            # Create trials table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY,
                    study_id INTEGER,
                    number INTEGER UNIQUE,
                    state TEXT,
                    value REAL,
                    datetime_start DATETIME,
                    datetime_complete DATETIME,
                    params TEXT,
                    user_attrs TEXT,
                    system_attrs TEXT,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id)
                )
            """)

            conn.commit()
            conn.close()
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

    def get_storage_url(self) -> str:
        """Get SQLite URL for Optuna storage."""
        return f"sqlite:///{self.storage_path}"

    def save_trial_history(self, trials: list) -> None:
        """Save trial history to JSON file."""
        try:
            history = {'trials': []}
            for trial in trials:
                trial_data = {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value if trial.state.name == 'COMPLETE' else None,
                    'state': trial.state.name,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                    'fail_reason': trial.user_attrs.get('fail_reason', None),
                }
                history['trials'].append(trial_data)

            with open(self.history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"\nSaved trial history to {self.history_path}")
        except Exception as e:
            logger.error(f"Error saving trial history: {str(e)}")

    def log_database_status(self) -> None:
        """Log current database status."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM trials")
            total_trials = cursor.fetchone()[0]

            cursor.execute("SELECT datetime_complete FROM trials WHERE datetime_complete IS NOT NULL ORDER BY datetime_complete DESC LIMIT 1")
            last_modified = cursor.fetchone()

            logger.info(f"\nOptuna Database Status:")
            logger.info(f"Location: {self.storage_path}")
            logger.info(f"Total trials in DB: {total_trials}")
            if last_modified:
                logger.info(f"Last modified: {last_modified[0]}")

            conn.close()
        except Exception as e:
            logger.error(f"Error checking database status: {e}")
