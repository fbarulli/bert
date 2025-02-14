# simpler_fine_bert/worker_manager.py

import logging
import os
import gc
import pickle
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import optuna
from optuna.trial import Trial

from simpler_fine_bert.cuda_utils import cuda_manager
from simpler_fine_bert.objective_factory import ObjectiveFactory
from simpler_fine_bert.model_manager import model_manager
from simpler_fine_bert.tokenizer_manager import tokenizer_manager

logger = logging.getLogger(__name__)

class WorkerManager:
    """Manages worker processes for parallel optimization."""
    
    def __init__(self, n_jobs: int, study_name: str, storage_url: str):
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage_url = storage_url
        self.worker_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self._active_workers = {}

    def start_workers(self) -> None:
        """Start worker processes."""
        logger.info(f"Starting {self.n_jobs} worker processes")
        for worker_id in range(self.n_jobs):
            process = mp.Process(
                target=self._worker_process,
                args=(worker_id,),
                daemon=True
            )
            process.start()
            self._active_workers[worker_id] = process
            logger.info(f"Started worker {worker_id} with PID {process.pid}")

    def _worker_process(self, worker_id: int) -> None:
        """Worker process with enhanced logging and resource management."""
        current_pid = os.getpid()
        logger.info(f"\n=== Worker {worker_id} Starting ===")
        logger.info(f"Process ID: {current_pid}")
        logger.info(f"Parent Process ID: {os.getppid()}")
        
        # Create a fresh study connection for this worker
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage_url
        )
        logger.info(f"Worker {worker_id} connected to study")
        
        try:
            while True:
                trial_data = self.worker_queue.get()
                if trial_data is None:  # Exit signal
                    logger.info(f"Worker {worker_id} received exit signal")
                    break

                try:
                    # Log trial start
                    trial_number = trial_data['trial_number']
                    logger.info(f"\n=== Trial {trial_number} Starting in Worker {worker_id} ===")
                    logger.info(f"Process ID: {current_pid}")
                    
                    # Extract and validate trial data
                    config = trial_data['config']
                    output_path = Path(trial_data['output_path'])
                    
                    # Initialize all process resources
                    from simpler_fine_bert.resource_initializer import ResourceInitializer
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

    def cleanup_workers(self) -> None:
        """Clean up worker processes."""
        logger.info("Cleaning up worker processes")
        for worker_id, process in self._active_workers.items():
            self.worker_queue.put(None)  # Send exit signal
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        self._active_workers.clear()

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
