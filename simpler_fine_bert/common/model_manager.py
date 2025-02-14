from __future__ import annotations
import torch
import logging
from typing import Dict, Any, Optional
from transformers import PreTrainedModel, BertConfig
from simpler_fine_bert.model import EmbeddingBert, ClassificationBert
from transformers.utils import logging as transformers_logging
from transformers.utils.hub import HFValidationError
from simpler_fine_bert.cuda_utils import cuda_manager
import os
import gc
import weakref
import threading
import torch.distributed as dist
import torch.distributed.rpc as rpc

logger = logging.getLogger(__name__)

class ModelManager:
    _local = threading.local()
    
    def __init__(self):
        if not hasattr(self._local, 'initialized'):
            self._local.process_models = {}  # Process-local model cache
            self._local.model_refs = weakref.WeakValueDictionary()  # Track model references
            self._local.pid = os.getpid()
            self._local.initialized = True
            self._local.parameter_server = None
            logger.info(f"ModelManager initialized for process {self._local.pid}")

    def _verify_model_device(self, model: torch.nn.Module, device: torch.device) -> bool:
        """Verify model is on the correct device."""
        try:
            # Check model parameters
            for param in model.parameters():
                if param.device != device:
                    logger.error(f"Parameter {param.shape} on {param.device}, expected {device}")
                    return False
                    
            # Check model buffers
            for buffer in model.buffers():
                if buffer.device != device:
                    logger.error(f"Buffer {buffer.shape} on {buffer.device}, expected {device}")
                    return False
                    
            # Check model state dict
            for key, tensor in model.state_dict().items():
                if tensor.device != device:
                    logger.error(f"State dict tensor {key} on {tensor.device}, expected {device}")
                    return False
                    
            # Check forward hook to verify input tensors
            def check_inputs(module, input):
                if isinstance(input, (tuple, list)):
                    for x in input:
                        if isinstance(x, torch.Tensor) and x.device != device:
                            logger.error(f"Input tensor {x.shape} on {x.device}, expected {device}")
                            return False
                elif isinstance(input, torch.Tensor) and input.device != device:
                    logger.error(f"Input tensor {input.shape} on {input.device}, expected {device}")
                    return False
                return True
                
            handle = model.register_forward_pre_hook(check_inputs)
            handle.remove()
            
            return True
        except Exception as e:
            logger.error(f"Error verifying model device: {str(e)}")
            return False

    def _move_model_to_device(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """Move model to device with verification."""
        try:
            # Move model to device
            model = model.to(device)
            
            # Verify model is on correct device
            if not self._verify_model_device(model, device):
                raise RuntimeError("Failed to move model to correct device")
            
            return model
        except Exception as e:
            logger.error(f"Error moving model to device: {str(e)}")
            raise

    def _optimize_model(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.nn.Module:
        """Apply PyTorch optimizations to model."""
        try:
            # Enable TorchScript JIT compilation if configured
            if config.get('training', {}).get('jit', False):
                logger.info("Applying TorchScript optimization")
                with torch.jit.optimized_execution(True):
                    model = torch.jit.script(model)

            # Enable torch.compile if configured
            if config.get('training', {}).get('compile', False):
                logger.info("Applying torch.compile optimization")
                model = torch.compile(
                    model,
                    mode=config.get('training', {}).get('compile_mode', 'default'),
                    fullgraph=True,
                    dynamic=False
                )

            # Enable static graph optimization if configured
            if config.get('training', {}).get('static_graph', False):
                logger.info("Enabling static graph optimization")
                if hasattr(model, '_set_static_graph'):
                    model._set_static_graph()

            return model
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise

    def _setup_parameter_server(self, config: Dict[str, Any]) -> None:
        """Setup parameter server for model parallelism."""
        try:
            if config.get('training', {}).get('parameter_server', False):
                logger.info("Setting up parameter server")
                # Initialize RPC framework
                if not rpc.is_initialized():
                    rpc.init_rpc(
                        name=f"worker_{self._local.pid}",
                        rank=0,
                        world_size=1
                    )
                # Create parameter server
                self._local.parameter_server = torch.nn.parallel.ParameterServer()
        except Exception as e:
            logger.error(f"Error setting up parameter server: {str(e)}")
            raise

    def get_worker_model(
        self,
        worker_id: int,
        model_name: str,
        model_type: str,  # 'embedding' or 'classification'
        device_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> PreTrainedModel:
        """Create model for worker using process-local resources."""
        # Check if we need to reinitialize for a new process
        current_pid = os.getpid()
        if hasattr(self._local, 'pid') and self._local.pid != current_pid:
            logger.info(f"Reinitializing ModelManager for new process {current_pid}")
            self._local.process_models = {}
            self._local.model_refs = weakref.WeakValueDictionary()
            self._local.pid = current_pid
            self._local.initialized = True
            self._local.parameter_server = None

        logger.debug(f"get_worker_model called from process {current_pid}")
        
        # Initialize CUDA for this process if needed
        if config:
            # Ensure cuda_manager is initialized before setup
            if not hasattr(cuda_manager._local, 'initialized'):
                cuda_manager.initialize()
            cuda_manager.setup(config)
        
        # Get device through cuda_manager
        device = cuda_manager.get_device()
        cache_key = f"{worker_id}_{model_name}"
        
        try:
            # Check if model exists in cache and is valid
            if cache_key in self._local.process_models:
                model = self._local.process_models[cache_key]
                if model is not None and self._verify_model_device(model, device):
                    logger.info(f"Using cached model for worker {worker_id}")
                    return model
                else:
                    # Remove invalid model from cache
                    logger.warning(f"Removing invalid model from cache for worker {worker_id}")
                    del self._local.process_models[cache_key]
                    self._local.model_refs.pop(cache_key, None)
                    gc.collect()
                    cuda_manager.cleanup()

            logger.info(f"Creating new model for worker {worker_id} in process {current_pid}")
            
            # Clear CUDA cache and verify clean state before model creation
            cuda_manager.cleanup()
            cuda_manager.verify_cuda_state()
            
            # Create model with error handling
            try:
                logger.info(f"Creating model with clean CUDA state in process {current_pid}")
                
                # Validate model name before initialization
                if not isinstance(model_name, str) or not model_name.strip():
                    raise ValueError("Invalid model name provided")
                    
                # Suppress transformers warnings during model loading
                transformers_logging.set_verbosity_error()
                
                try:
                    # Create model configuration
                    model_config = BertConfig.from_pretrained(
                        model_name,
                        hidden_dropout_prob=config.get('training', {}).get('hidden_dropout_prob', 0.1),
                        attention_probs_dropout_prob=config.get('training', {}).get('attention_probs_dropout_prob', 0.1)
                    )

                    # Create appropriate model based on type
                    if model_type == 'embedding':
                        model = EmbeddingBert(
                            config=model_config,
                            tie_weights=True
                        )
                    elif model_type == 'classification':
                        model = ClassificationBert(
                            config=model_config,
                            num_labels=config['model']['num_labels']
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                except (OSError, HFValidationError) as e:
                    raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Unexpected error loading model: {str(e)}")
                finally:
                    # Restore normal logging
                    transformers_logging.set_verbosity_warning()
                    
                # Verify model was created successfully
                if not isinstance(model, PreTrainedModel):
                    raise RuntimeError("Model initialization failed to return a valid model instance")
                
                # Move to device
                model = self._move_model_to_device(model, device)
                
                # Apply optimizations
                model = self._optimize_model(model, config)
                
                # Setup parameter server if needed
                self._setup_parameter_server(config)
                
                # Register model with cuda_manager for protection
                cuda_manager.register_model(model)
                
                # Verify model creation
                if model is None:
                    raise ValueError("Model creation failed")
                    
            except Exception as e:
                logger.error(f"Error creating model in process {current_pid}: {str(e)}")
                raise
            
            # Store model in cache and track reference
            self._local.process_models[cache_key] = model
            self._local.model_refs[cache_key] = model
            logger.info(f"Successfully created model for worker {worker_id} on {device} in process {current_pid}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to get worker model in process {current_pid}: {str(e)}")
            # Clean up on error
            if cache_key in self._local.process_models:
                try:
                    # Move model to CPU before deletion
                    self._local.process_models[cache_key].cpu()
                except:
                    pass
                del self._local.process_models[cache_key]
                self._local.model_refs.pop(cache_key, None)
            cuda_manager.cleanup()
            gc.collect()
            raise

    def cleanup_worker(self, worker_id: int):
        """Cleanup worker's model resources for current process."""
        try:
            # Remove models for this worker
            keys_to_remove = [k for k in self._local.process_models.keys() if k.startswith(f"{worker_id}_")]
            for key in keys_to_remove:
                if key in self._local.process_models:
                    try:
                        # Move model to CPU before deletion
                        model = self._local.process_models[key]
                        if model is not None:
                            # Unregister from cuda_manager
                            cuda_manager.unregister_model(model)
                            # Clear gradients
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.detach_()
                                    param.grad.zero_()
                            # Move to CPU
                            model.cpu()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model: {str(e)}")
                    finally:
                        del self._local.process_models[key]
                        self._local.model_refs.pop(key, None)
            
            # Clean up parameter server
            if self._local.parameter_server is not None:
                del self._local.parameter_server
                self._local.parameter_server = None
                if rpc.is_initialized():
                    rpc.shutdown()
            
            # Force garbage collection
            gc.collect()
            
            # Clean up CUDA resources
            cuda_manager.cleanup()
            
            logger.info(f"Cleaned up resources for worker {worker_id} in process {self._local.pid}")
            
        except Exception as e:
            logger.error(f"Error during worker cleanup: {str(e)}")
            # Ensure CUDA is cleaned up even if other cleanup fails
            cuda_manager.cleanup()

model_manager = ModelManager()
