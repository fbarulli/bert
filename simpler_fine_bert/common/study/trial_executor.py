from __future__ import annotations
import torch
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
import optuna
from optuna.trial import Trial

from simpler_fine_bert.common.resource.resource_factory import resource_factory
from simpler_fine_bert.common.cuda_utils import is_cuda_available

logger = logging.getLogger(__name__)

class TrialExecutor:
    """Handles trial execution and resource management."""
    
    def __init__(self, base_config: Dict[str, Any], n_jobs: int):
        self.base_config = base_config
        self.n_jobs = n_jobs
        self.cuda_available = is_cuda_available()
        
    def run_trials(
        self,
        study: optuna.Study,
        objective_fn: Callable,
        n_trials: int,
        resources: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> List[Tuple[optuna.trial.FrozenTrial, float]]:
        """Run trials with proper process management."""
        ctx = mp.get_context('spawn')
        queues = []
        workers = []
        
        try:
            trial_queue = ctx.Queue()
            result_queue = ctx.Queue()
            queues.extend([trial_queue, result_queue])
            
            # Initialize trial queue
            for i in range(n_trials):
                trial_queue.put(i)
            
            # Start workers
            n_workers = min(self.n_jobs, n_trials)
            workers = self._start_workers(ctx, n_workers, trial_queue, result_queue, study, objective_fn, resources)
            
            # Wait for all workers
            results = self._collect_results(workers, result_queue)
            return results
            
        finally:
            self._cleanup_resources(queues, workers)
    
    def _start_workers(self, ctx, n_workers, trial_queue, result_queue, study, objective_fn, resources):
        """Start worker processes."""
        workers = []
        for worker_id in range(n_workers):
            p = ctx.Process(
                target=self._worker_process,
                args=(worker_id, trial_queue, result_queue, study, objective_fn, resources)
            )
            p.start()
            workers.append(p)
        return workers

    def _worker_process(self, worker_id, trial_queue, result_queue, study, objective_fn, resources):
        """Execute trials in worker process."""
        try:
            while True:
                try:
                    trial_idx = trial_queue.get_nowait()
                except mp.queues.Empty:
                    break
                    
                trial = study.ask()
                try:
                    value = objective_fn(trial, resources)
                    study.tell(trial, value)
                    result_queue.put((trial, value))
                except Exception as e:
                    logger.error(f"Trial {trial_idx} failed: {e}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    
        finally:
            pass

    def _collect_results(self, workers, result_queue):
        """Collect results from all workers."""
        for w in workers:
            w.join()
            
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        return results

    def _cleanup_resources(self, queues, workers):
        """Clean up resources."""
        # Clean up queues
        for q in queues:
            if q is not None:
                try:
                    while not q.empty():
                        _ = q.get_nowait()
                except (mp.queues.Empty, EOFError, BrokenPipeError):
                    pass
                finally:
                    q.close()
                    q.join_thread()
        
        # Clean up workers
        for w in workers:
            if w is not None and w.is_alive():
                w.terminate()
                w.join(timeout=1)
                if w.is_alive():
                    w.kill()
