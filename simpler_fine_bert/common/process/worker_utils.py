# simpler_fine_bert/worker_utils.py

import logging
import os
import gc
import traceback
import optuna
from optuna.trial import FixedTrial
from pathlib import Path
import multiprocessing as mp
from typing import Dict, Any, Optional, Tuple
import torch

from simpler_fine_bert.cuda_utils import cuda_manager
from simpler_fine_bert.objective_factory import ObjectiveFactory

logger = logging.getLogger(__name__)

def cleanup_process_resources():
    """Clean up all resources for current process."""
    try:
        # Clear CUDA cache and move tensors to CPU
        cuda_manager.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        # Clear any remaining CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"Cleaned up process resources for PID {os.getpid()}")
    except Exception as e:
        logger.error(f"Error during process cleanup: {str(e)}")

def run_worker(
    worker_id: int,
    study_name: str,
    storage_url: str,
    in_queue: mp.Queue,
    out_queue: mp.Queue
) -> None:
    """Standalone worker function that runs trials.
    
    Args:
        worker_id: ID of this worker
        study_name: Name of the Optuna study
        storage_url: URL for study storage
        in_queue: Queue for receiving trial data
        out_queue: Queue for sending results
    """
    # Initialize process
    from simpler_fine_bert.process_init import initialize_process
    current_pid, parent_pid = initialize_process()
    
    logger.info(f"\n=== Worker {worker_id} Starting ===")
    logger.info(f"Worker Process Details:")
    logger.info(f"- Process ID: {current_pid}")
    logger.info(f"- Parent Process ID: {parent_pid}")
    logger.info(f"- Study Name: {study_name}")
    logger.info(f"- Storage URL: {storage_url}")
    
    # Create fresh study connection for this worker
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )
    logger.info(f"Worker {worker_id} connected to study")
    
    try:
        while True:
            # Get trial data
            trial_data = in_queue.get()
            if trial_data is None:  # Exit signal
                logger.info(f"Worker {worker_id} received exit signal")
                break

            try:
                # Log received trial data
                logger.info("\n=== Received Trial Data ===")
                logger.info(f"Trial data keys: {trial_data.keys()}")
                logger.info(f"Trial number: {trial_data['trial_number']}")
                logger.info(f"Trial params: {trial_data['trial_params']}")
                logger.info(f"Config keys: {trial_data['config'].keys()}")
                logger.info(f"Output path: {trial_data['output_path']}")

                # Extract trial data
                trial_number = trial_data['trial_number']
                trial_params = trial_data['trial_params']
                config = trial_data['config']
                output_path = Path(trial_data['output_path'])
                
                # Create FixedTrial with parameters
                logger.info("\n=== Creating FixedTrial ===")
                logger.info(f"Parameters: {trial_params}")
                trial = FixedTrial(trial_params)
                trial_info = {
                    'trial': trial,
                    'number': trial_number
                }
                
                # Initialize CUDA
                logger.info("\n=== Initializing CUDA ===")
                try:
                    cuda_manager.setup(config)
                    logger.info(f"CUDA initialized successfully for process {current_pid}")
                except Exception as e:
                    logger.error(f"CUDA initialization failed: {str(e)}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    raise
                
                try:
                    # Create ObjectiveFactory
                    logger.info("\n=== Creating ObjectiveFactory ===")
                    factory = ObjectiveFactory(config, output_path)
                    
                    # Run trial
                    logger.info(f"\n=== Executing Trial {trial_info['number']} ===")
                    result = factory.objective_method(trial_info['trial'])
                    
                    # Validate result
                    if not isinstance(result, (int, float)):
                        raise ValueError(f"Invalid result type: {type(result)}. Expected int or float.")
                    
                    logger.info(f"Trial {trial_info['number']} completed with result: {result}")
                    out_queue.put((trial_info['number'], result, None))
                    
                except Exception as e:
                    logger.error(f"\n=== Trial {trial_info['number']} Failed ===")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Error message: {str(e)}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    out_queue.put((trial_info['number'], None, f"{type(e).__name__}: {str(e)}"))
                    
                finally:
                    logger.info(f"Cleaning up resources for trial {trial_info['number']}")
                    # Clean up trial resources
                    cleanup_process_resources()
                    
            except Exception as e:
                logger.error(f"Error in trial setup: {str(e)}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                if 'trial_number' in trial_data:
                    out_queue.put((trial_data['trial_number'], None, str(e)))
                cleanup_process_resources()
                    
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
    finally:
        logger.info(f"\n=== Worker {worker_id} Shutting Down ===")
        # Final cleanup
        cleanup_process_resources()
        logger.info(f"Worker {worker_id} cleanup complete")
