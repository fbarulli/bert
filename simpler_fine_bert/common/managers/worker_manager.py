# simpler_fine_bert/common/managers/worker_manager.py

import logging
import os
import gc
import pickle
import traceback
import threading
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import optuna
from optuna.trial import Trial
import torch

from simpler_fine_bert.common.managers import get_cuda_manager

# Get manager instance
cuda_manager = get_cuda_manager()
from simpler_fine_bert.common.study.objective_factory import ObjectiveFactory
from simpler_fine_bert.common.managers import (
    get_model_manager,
    get_tokenizer_manager
)

# Get manager instances
model_manager = get_model_manager()
tokenizer_manager = get_tokenizer_manager()

logger = logging.getLogger(__name__)

class WorkerManager:
    """Manages worker processes for parallel optimization."""
    
    def __init__(
        self,
        n_jobs: int = 2,
        study_name: str = "",
        storage_url: str = "",
        max_workers: int = 32,
        health_check_interval: int = 60
    ):
        """Initialize worker manager.
        
        Args:
            n_jobs: Initial number of worker processes (default: 2)
            study_name: Name of the optuna study
            storage_url: URL for optuna storage
            max_workers: Maximum number of workers allowed (default: 8)
        """
        self.n_jobs = min(n_jobs, max_workers)  # Cap at max_workers
        self.max_workers = max_workers
        self.study_name = study_name
        self.storage_url = storage_url
        self.worker_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self._active_workers = {}
        self._worker_groups = {}  # Group workers by task type
        self._resource_limits = {
            'memory_gb': 22.5,  # From config
            'gpu_memory_gb': 10.0,
            'max_cpu_percent': 80.0
        }
        
        # Health check settings
        self.health_check_interval = health_check_interval
        self._last_health_check = time.time()
        self._health_check_thread = None
        
        # Initialize process settings
        from simpler_fine_bert.common.process.multiprocessing_setup import verify_spawn_method
        
        # Start health check thread
        self._start_health_check_thread()

    def _monitor_resources(self) -> Dict[str, float]:
        """Monitor system resources."""
        try:
            import psutil
            import torch
            
            cpu_percent = psutil.cpu_percent()
            memory_gb = psutil.Process().memory_info().rss / (1024 ** 3)
            
            gpu_memory_gb = 0
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                
            return {
                'cpu_percent': cpu_percent,
                'memory_gb': memory_gb,
                'gpu_memory_gb': gpu_memory_gb
            }
        except ImportError:
            logger.warning("psutil not installed - resource monitoring disabled")
            return {}

    def _check_resource_limits(self, resources: Dict[str, float]) -> bool:
        """Check if resource usage is within limits."""
        if not resources:
            return True
            
        return (
            resources.get('memory_gb', 0) < self._resource_limits['memory_gb'] and
            resources.get('gpu_memory_gb', 0) < self._resource_limits['gpu_memory_gb'] and
            resources.get('cpu_percent', 0) < self._resource_limits['max_cpu_percent']
        )

    def _check_worker_health(self) -> None:
        """Check health of workers and restart if needed."""
        for worker_id, process in list(self._active_workers.items()):
            if not process.is_alive():
                logger.warning(f"Worker {worker_id} died - restarting")
                group = next(g for g, workers in self._worker_groups.items() if worker_id in workers)
                
                # Clean up old worker
                self._active_workers.pop(worker_id)
                self._worker_groups[group].pop(worker_id)
                
                # Start new worker
                self.n_jobs = 1
                self.start_workers(group)

    def start_workers(self, group: str = "default", check_resources: bool = True) -> None:
        """Start worker processes for parallel optimization.
        
        Args:
            group: Worker group name for organization (default: "default")
            check_resources: Whether to check system resources before starting workers (default: True)
        """
        # Check system resources
        if check_resources:
            resources = self._monitor_resources()
            if not self._check_resource_limits(resources):
                raise RuntimeError(
                    f"Insufficient resources to start workers:\n"
                    f"CPU: {resources.get('cpu_percent', 0)}%\n"
                    f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                    f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                )

        # Check worker limits
        if len(self._active_workers) + self.n_jobs > self.max_workers:
            raise RuntimeError(
                f"Cannot start {self.n_jobs} workers - would exceed max_workers ({self.max_workers})\n"
                f"Active workers: {len(self._active_workers)}"
            )
            
        logger.info(f"Starting {self.n_jobs} worker processes in group '{group}'")
        
        # Initialize worker group if needed
        if group not in self._worker_groups:
            self._worker_groups[group] = {}
            
        # Calculate starting worker_id to avoid conflicts
        start_id = max(self._active_workers.keys()) + 1 if self._active_workers else 0
        
        for i in range(self.n_jobs):
            worker_id = start_id + i
            
            # Create and start worker process
            process = mp.Process(
                target=self._worker_process,
                args=(worker_id, group),
                daemon=True
            )
            process.start()
            
            self._active_workers[worker_id] = process
            self._worker_groups[group][worker_id] = process
            
            logger.info(f"Started worker {worker_id} in group '{group}' with PID {process.pid}")

    def _worker_process(self, worker_id: int, group: str) -> None:
        """Worker process with enhanced logging and resource management."""
        current_pid = os.getpid()
        logger.info(f"\n=== Worker {worker_id} Starting ===")
        logger.info(f"Process ID: {current_pid}")
        logger.info(f"Parent Process ID: {os.getppid()}")
        
        # Verify spawn method
        verify_spawn_method()
        
        # Create study connection
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage_url
        )
        logger.info(f"Worker {worker_id} initialized and connected to study")
        
        try:
            import threading
            import time
            
            # Start health check thread
            def health_check():
                while True:
                    try:
                        resources = self._monitor_resources()
                        if not self._check_resource_limits(resources):
                            logger.warning(
                                f"Worker {worker_id} exceeding resource limits:\n"
                                f"CPU: {resources.get('cpu_percent', 0)}%\n"
                                f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                                f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                            )
                    except Exception as e:
                        logger.error(f"Health check failed in worker {worker_id}: {e}")
                    time.sleep(60)  # Check every minute
            
            health_thread = threading.Thread(target=health_check, daemon=True)
            health_thread.start()
            
            while True:
                try:
                    trial_data = self.worker_queue.get(timeout=300)  # 5 minute timeout
                    if trial_data is None:  # Exit signal
                        logger.info(f"Worker {worker_id} received exit signal")
                        break
                except Exception as e:
                    # Queue timeout or other error
                    logger.warning(f"Worker {worker_id} queue error: {e}")
                    # Check if we should exit
                    if not self._active_workers.get(worker_id):
                        logger.info(f"Worker {worker_id} no longer active - shutting down")
                        break
                    continue

                try:
                    # Log trial start
                    trial_number = trial_data['trial_number']
                    logger.info(f"\n=== Trial {trial_number} Starting in Worker {worker_id} ===")
                    logger.info(f"Process ID: {current_pid}")
                    
                    # Extract and validate trial data
                    config = trial_data['config']
                    output_path = Path(trial_data['output_path'])
                    
                    # Initialize all process resources
                    from simpler_fine_bert.common.resource.resource_initializer import ResourceInitializer
                    ResourceInitializer.initialize_process()
                    logger.info(f"Process resources initialized for {current_pid}")
                    
                    # Create trial object using worker's study connection
                    trial = optuna.trial.Trial(
                        study,
                        trial_number,
                        trial_data['trial_params']
                    )
                    logger.info("Trial object created")
                    
                    try:
                        # Create fresh ObjectiveFactory instance
                        factory = ObjectiveFactory(config, output_path)
                        logger.info("ObjectiveFactory created")
                        
                        # Run trial
                        logger.info(f"Starting trial {trial_number} execution")
                        result = factory.objective(trial, config, output_path, None)
                        logger.info(f"Trial {trial_number} completed with result: {result}")
                        self.result_queue.put((trial_number, result, None))
                        
                    except Exception as e:
                        logger.error(f"Trial {trial_number} failed: {str(e)}")
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                        self.result_queue.put((trial_number, None, str(e)))
                        
                    finally:
                        logger.info(f"Cleaning up resources for trial {trial_number}")
                        # Clean up model and tokenizer resources
                        model_manager.cleanup_worker(worker_id)
                        tokenizer_manager.cleanup_worker(worker_id)
                        # Clean up remaining resources
                        ResourceInitializer.cleanup_process()
                        logger.info(f"Process resources cleaned up for trial {trial_number}")
                        
                except Exception as e:
                    logger.error(f"Error in trial setup: {str(e)}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    if 'trial_number' in trial_data:
                        self.result_queue.put((trial_data['trial_number'], None, str(e)))
                    
        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        finally:
            logger.info(f"\n=== Worker {worker_id} Shutting Down ===")
            # Clean up model and tokenizer resources
            model_manager.cleanup_worker(worker_id)
            tokenizer_manager.cleanup_worker(worker_id)
            # Clean up remaining resources
            ResourceInitializer.cleanup_process()
            logger.info(f"Worker {worker_id} cleanup complete")

    def _start_health_check_thread(self) -> None:
        """Start background thread for periodic health checks."""
        def health_check_loop():
            while True:
                try:
                    self._check_worker_health()
                    resources = self._monitor_resources()
                    if not self._check_resource_limits(resources):
                        logger.warning(
                            f"System resources exceeding limits:\n"
                            f"CPU: {resources.get('cpu_percent', 0)}%\n"
                            f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                            f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                        )
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                time.sleep(self.health_check_interval)
        
        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="HealthCheckThread"
        )
        self._health_check_thread.start()
        logger.info("Started health check thread")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup_workers()
        except:
            pass

    def cleanup_workers(self, group: Optional[str] = None, timeout: int = 30) -> None:
        """Clean up worker processes.
        
        Args:
            group: Optional group name to clean up specific workers. If None, clean up all workers.
        """
        if group:
            if group not in self._worker_groups:
                logger.warning(f"No workers found in group '{group}'")
                return
                
            logger.info(f"Cleaning up workers in group '{group}'")
            workers_to_cleanup = self._worker_groups[group]
        else:
            logger.info("Cleaning up all worker processes")
            workers_to_cleanup = self._active_workers
            
        for worker_id, process in workers_to_cleanup.items():
            self.worker_queue.put(None)  # Send exit signal
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                
            # Clean up worker references
            self._active_workers.pop(worker_id, None)
            if group:
                self._worker_groups[group].pop(worker_id, None)
                if not self._worker_groups[group]:
                    del self._worker_groups[group]
                    
        if not group:
            self._worker_groups.clear()
            # Stop health check thread
            if self._health_check_thread and self._health_check_thread.is_alive():
                self._health_check_thread = None

    def queue_trial(self, trial_data: Dict[str, Any]) -> None:
        """Queue a trial for execution."""
        try:
            # Verify trial data is picklable
            pickle.dumps(trial_data)
            self.worker_queue.put(trial_data)
            logger.info(f"Queued trial {trial_data['trial_number']}")
        except Exception as e:
            logger.error(f"Failed to queue trial: {e}")
            raise

    def get_result(self) -> Tuple[int, Optional[float], Optional[str]]:
        """Get result from a completed trial."""
        return self.result_queue.get()

    def scale_workers(self, n_jobs: int, group: str = "default", check_resources: bool = True) -> None:
        """Scale number of workers up or down.
        
        Args:
            n_jobs: New number of workers
            group: Worker group to scale
        """
        # Check system resources before scaling up
        if check_resources and n_jobs > len(self._worker_groups.get(group, {})):
            resources = self._monitor_resources()
            if not self._check_resource_limits(resources):
                raise RuntimeError(
                    f"Insufficient resources to scale to {n_jobs} workers:\n"
                    f"CPU: {resources.get('cpu_percent', 0)}%\n"
                    f"Memory: {resources.get('memory_gb', 0):.1f}GB\n"
                    f"GPU Memory: {resources.get('gpu_memory_gb', 0):.1f}GB"
                )

        if n_jobs > self.max_workers:
            raise ValueError(
                f"Cannot scale to {n_jobs} workers - exceeds max_workers ({self.max_workers})\n"
                f"Consider increasing max_workers if more capacity is needed"
            )
            
        current_workers = len(self._worker_groups.get(group, {}))
        
        if n_jobs > current_workers:
            # Scale up
            self.n_jobs = n_jobs - current_workers
            self.start_workers(group)
        elif n_jobs < current_workers:
            # Scale down
            workers_to_remove = current_workers - n_jobs
            workers = list(self._worker_groups[group].items())[-workers_to_remove:]
            
            for worker_id, _ in workers:
                self.worker_queue.put(None)
                self._active_workers.pop(worker_id, None)
                self._worker_groups[group].pop(worker_id, None)
                
        self.n_jobs = n_jobs

# Create singleton instance with default values (will be initialized later)
worker_manager = WorkerManager(n_jobs=2, study_name="", storage_url="")

__all__ = ['WorkerManager', 'worker_manager']
