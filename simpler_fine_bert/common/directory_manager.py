from __future__ import annotations

import logging
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class DirectoryManager:
    """Manages directory structure for outputs and caching"""
    
    def __init__(self, base_dir: Path):
        """Initialize directory manager.
        
        Args:
            base_dir: Base directory for all outputs
        """
        if base_dir is None:
            raise ValueError("base_dir cannot be None")
            
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard directories
        self.cache_dir = self.base_dir / 'cache'
        self.mmap_dir = self.base_dir / 'mmap'
        self.cache_dir.mkdir(exist_ok=True)
        self.mmap_dir.mkdir(exist_ok=True)
        
        logger.debug(f"Created directory structure at {self.base_dir}")
    
    def get_db_path(self) -> Path:
        """Get path to optuna database.
        
        Returns:
            Path to optuna.db file
        """
        return self.base_dir / 'optuna.db'
    
    def get_history_path(self) -> Path:
        """Get path to trial history file.
        
        Returns:
            Path to trial_history.json file
        """
        return self.base_dir / 'trial_history.json'
    
    def get_cache_path(self, data_path: Path, prefix: str = '') -> Path:
        """Get cache path for a data file.
        
        Args:
            data_path: Original data file path
            prefix: Optional prefix for cache file
            
        Returns:
            Path to cache file
        """
        # Create hash of data path and modification time
        hasher = hashlib.sha256()
        hasher.update(str(data_path).encode())
        if data_path.exists():
            hasher.update(str(data_path.stat().st_mtime).encode())
        cache_hash = hasher.hexdigest()[:16]
        
        return self.cache_dir / f"{prefix}_{cache_hash}.pt"
    
    def get_mmap_path(self, array_name: str, trial_number: Optional[int] = None) -> Path:
        """Get path for memory-mapped array.
        
        Args:
            array_name: Name of array
            trial_number: Optional trial number
            
        Returns:
            Path to memory-mapped file
        """
        if trial_number is not None:
            return self.mmap_dir / f"trial_{trial_number}_{array_name}.mmap"
        return self.mmap_dir / f"{array_name}.mmap"
    
    def cleanup_cache(self, older_than_days: Optional[float] = None) -> None:
        """Clean up old cache files.
        
        Args:
            older_than_days: Optional number of days, files older than this will be deleted
        """
        try:
            if older_than_days is not None:
                import time
                cutoff = time.time() - (older_than_days * 24 * 60 * 60)
                for path in self.cache_dir.glob('*.pt'):
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        logger.debug(f"Deleted old cache file: {path}")
            else:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                logger.debug("Cleared cache directory")
        except Exception as e:
            logger.warning(f"Error cleaning cache: {e}")
    
    def cleanup_mmap(self, trial_number: Optional[int] = None) -> None:
        """Clean up memory-mapped files.
        
        Args:
            trial_number: Optional trial number, if provided only clean files for this trial
        """
        try:
            if trial_number is not None:
                pattern = f"trial_{trial_number}_*.mmap"
            else:
                pattern = "*.mmap"
                
            for path in self.mmap_dir.glob(pattern):
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting mmap file {path}: {e}")
            
            logger.debug(f"Cleaned up mmap files{f' for trial {trial_number}' if trial_number else ''}")
            
        except Exception as e:
            logger.warning(f"Error cleaning mmap directory: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all temporary files."""
        try:
            self.cleanup_cache()
            self.cleanup_mmap()
            logger.debug("Cleaned up all temporary files")
        except Exception as e:
            logger.warning(f"Error in cleanup: {e}")

# Create singleton instance
directory_manager = DirectoryManager(base_dir=Path(os.getcwd()))

__all__ = ['DirectoryManager', 'directory_manager']
