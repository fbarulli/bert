from __future__ import annotations

import logging
import time
import json
import threading
from typing import Dict, Any, Optional
import gc
import torch
import os

logger = logging.getLogger(__name__)

# Set WANDB_START_METHOD *before* importing wandb
os.environ["WANDB_START_METHOD"] = "thread"

try:
    import wandb
    WANDB_AVAILABLE = True
    logger.debug("Module level wandb import successful")
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # Make sure wandb is defined even if import fails
    logger.debug("Module level wandb import failed")

class WandbManager:
    def __init__(
        self,
        config: Dict[str, Any],
        study_name: str
    ):
        logger.debug("\n=== WandbManager Init ===")
        logger.debug(f"Process: PID={os.getpid()}, PPID={os.getppid()}")
        logger.debug(f"Thread: {threading.current_thread().name}")
        logger.debug(f"WANDB_AVAILABLE={WANDB_AVAILABLE}")

        self.config = config
        self.study_name = study_name
        self.start_time = None
        self.run = None  # Store the current wandb run

        # Check if wandb is enabled in the config *and* available
        self.enabled = WANDB_AVAILABLE and config.get('output', {}).get('wandb', {}).get('enabled', False)

        logger.debug(f"Enabled: {self.enabled}")
        logger.debug(f"Study name: {study_name}")
        logger.debug(f"Wandb config: {json.dumps(config.get('output', {}).get('wandb', {}), indent=2)}")

        if self.enabled:
            wandb_config = config.get('output', {}).get('wandb', {})
            api_key = wandb_config.get('api_key')
            if not api_key:
                logger.warning("No wandb API key provided, disabling wandb logging")
                self.enabled = False  # Disable if no API key
            else:
                try:
                    wandb.login(key=api_key)
                    logger.info("Successfully logged in to wandb")
                except Exception as e:
                    logger.error(f"Failed to login to wandb: {e}")
                    self.enabled = False # Disable if login fails

        self._log_process_info()

    def _log_process_info(self):
        """Log detailed process and wandb state information"""
        logger.debug("\n=== Process Info ===")
        logger.debug("Process:")
        logger.debug(f"- PID: {os.getpid()}")
        logger.debug(f"- PPID: {os.getppid()}")
        logger.debug(f"- Thread: {threading.current_thread().name}")
        logger.debug("\nWandb State:")
        logger.debug(f"- WANDB_AVAILABLE: {WANDB_AVAILABLE}")
        logger.debug(f"- enabled: {self.enabled}")
        logger.debug(f"- current run: {self.run.id if self.run else None}")
        logger.debug("\nEnvironment Variables:")
        wandb_env_vars = {k: v for k, v in os.environ.items() if 'WANDB' in k}
        logger.debug(json.dumps(wandb_env_vars, indent=2))

    def cleanup_run(self):
        """Helper method to properly cleanup a run"""
        logger.debug("\n=== Cleanup Run ===")
        self._log_process_info()

        if self.enabled and self.run:
            try:
                self.run.finish()
                logger.debug("Run finished successfully")
            except Exception as e:
                logger.warning(f"Error finishing run: {e}")
            finally:
                self.run = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def init_optimization(self) -> None:
        """Initialize a single wandb run for the entire optimization."""
        logger.debug("\n=== Init Optimization ===")
        self._log_process_info()

        if not self.enabled:
            logger.debug("Wandb disabled, skipping initialization")
            return

        # Cleanup any existing run first
        self.cleanup_run()

        try:
            logger.debug("Starting wandb.init")
            wandb_config = self.config.get('output', {})
            group_name = f"optimization_{self.study_name}"
            
            self.run = wandb.init(
                project=wandb_config.get('project', self.study_name),
                group=group_name,  # Group all optimization runs
                name="optimization_main",  # Main run name
                job_type="optimization",
                tags=wandb_config.get('tags', []),
                config=self.config,
                reinit=True  # Allow reinit for parallel runs
            )
            self.start_time = time.time()
            logger.debug(f"Wandb run initialized: {self.run.id if self.run else None}")
            logger.info("Initialized wandb run for optimization")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.cleanup_run()

    def init_trial(self, trial_number: int) -> None:
        """Initialize a new wandb run for this trial."""
        logger.debug(f"\n=== Init Trial {trial_number} ===")
        self._log_process_info()

        if not self.enabled:
            return

        try:
            # Cleanup any existing run first
            self.cleanup_run()
            
            # Create a new run for this trial
            group_name = f"optimization_{self.study_name}"
            wandb_config = self.config.get('output', {})
            
            self.run = wandb.init(
                project=wandb_config.get('project', self.study_name),
                group=group_name,  # Same group as main run
                name=f"trial_{trial_number}",  # Unique name per trial
                job_type="trial",
                tags=wandb_config.get('tags', []),
                config=self.config,
                reinit=True  # Allow reinit for parallel runs
            )
            
            # Log trial start
            self.run.log({
                'trial_number': trial_number,
                'trial_status': 'started',
                'process_id': os.getpid()
            }, step=0)  # Start at step 0 for each trial
            
            logger.info(f"Successfully initialized trial {trial_number} (PID: {os.getpid()})")
        except Exception as e:
            logger.warning(f"Failed to initialize trial: {e}")
            self.cleanup_run()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb. Step must be provided by the trainer."""
        if not self.enabled or self.run is None:
            return

        if step is None:
            logger.warning("Step must be provided for logging metrics")
            return

        logger.debug(f"\n=== Log Metrics ===")
        logger.debug(f"Step: {step}")
        logger.debug(f"Metrics: {json.dumps(metrics, indent=2)}")
        self._log_process_info()

        try:
            self.run.log(metrics, step=step)
            logger.debug("Metrics logged successfully")
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")

    def finish_trial(self, trial_number: int) -> None:
        """Log trial completion in the existing run."""
        logger.debug(f"\n=== Finish Trial {trial_number} ===")
        self._log_process_info()

        if not self.enabled or self.run is None:
            return

        try:
            self.run.log({
                'trial_number': trial_number,
                'trial_status': 'completed',
                'process_id': os.getpid()
            }, step=0)  # Use step 0 for trial completion
            logger.info(f"Successfully logged trial completion (PID: {os.getpid()})")
            
            # Cleanup this trial's run
            self.cleanup_run()
        except Exception as e:
            logger.warning(f"Failed to log trial completion: {e}")

    def finish(self) -> None:
        """Finish the wandb run."""
        logger.debug("\n=== Finish ===")
        self._log_process_info()

        if not self.enabled:
            return

        # Only the main process should finish the run
        if os.getpid() == os.getppid():
            logger.debug("Main process finishing run")
            self.cleanup_run()
            logger.info("Finished wandb optimization run")
        else:
            logger.debug("Child process skipping run finish")

    def flush(self) -> None:
        """Flush wandb logs."""
        logger.debug("\n=== Flush ===")
        self._log_process_info()

        if not self.enabled or self.run is None:
            return

        try:
            self.run.flush()
            logger.debug("Logs flushed successfully")
        except Exception as e:
            logger.warning(f"Failed to flush wandb: {e}")

    def init_final_training(self) -> None:
        """Initialize wandb for final training."""
        logger.debug("\n=== Init Final Training ===")
        self._log_process_info()

        if not self.enabled:
            return

        # Cleanup any existing run first
        self.cleanup_run()

        try:
            logger.debug("Starting wandb.init for final training")
            wandb_config = self.config.get('output', {}).get('wandb', {})
            self.run = wandb.init(
                project=wandb_config.get('project', self.study_name),
                name="final_training",
                job_type="final_training",
                tags=wandb_config.get('tags', []) + ["final"],
                config=self.config,
                reinit=True  # Allow reinit for final training
            )
            self.start_time = time.time()
            logger.debug(f"Wandb run initialized: {self.run.id if self.run else None}")
            logger.info("Initialized wandb for final training")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb for final training: {e}")
            self.cleanup_run()

    def log_progress(self, current: int, total: int, prefix: str = '', step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None:
            return

        try:
            elapsed = time.time() - self.start_time if self.start_time else 0
            progress = current / total
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0

            metrics = {
                f'{prefix}progress': progress * 100,
                f'{prefix}current': current,
                f'{prefix}total': total,
                f'{prefix}elapsed_time': elapsed,
                f'{prefix}remaining_time': remaining,
                f'{prefix}speed': current / elapsed if elapsed > 0 else 0
            }

            if step is not None:
                self.run.log(metrics, step=step)
            else:
                self.run.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log progress to wandb: {e}")

    def log_epoch_progress(self, epoch: int, total_epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None:
            return

        try:
            elapsed = time.time() - self.start_time if self.start_time else 0
            progress = epoch / total_epochs
            remaining = (elapsed / progress) * (1 - progress) if progress > 0 else 0

            metrics = {
                'epoch': epoch,
                'epoch_progress': progress * 100,
                'epochs_remaining': total_epochs - epoch,
                'epoch_countdown': remaining
            }

            # Add train metrics with prefix
            for key, value in train_metrics.items():
                metrics[f'train_{key}'] = value

            # Add val metrics with prefix
            for key, value in val_metrics.items():
                metrics[f'val_{key}'] = value

            if step is not None:
                self.run.log(metrics, step=step)
            else:
                self.run.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log epoch progress to wandb: {e}")
