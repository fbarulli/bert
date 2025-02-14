from __future__ import annotations
import logging
import os
import weakref
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from transformers.utils import logging as transformers_logging

from simpler_fine_bert.common.base_manager import BaseManager

logger = logging.getLogger(__name__)

class TokenizerManager(BaseManager):
    """Process-local tokenizer manager."""
    
    def _initialize_process_local(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize process-local attributes."""
        # Call parent's initialization first
        super()._initialize_process_local(config)
        
        logger.info(f"Initializing TokenizerManager for process {os.getpid()}")
        
        # Initialize process-local storage
        self._local.process_tokenizers = {}  # Process-local tokenizer cache
        self._local.tokenizer_refs = weakref.WeakValueDictionary()  # Track tokenizer references

    def get_worker_tokenizer(
        self,
        worker_id: int,
        model_name: str,
        model_type: str = 'embedding',
        config: Optional[Dict[str, Any]] = None
    ) -> PreTrainedTokenizerFast:
        """Create tokenizer for worker using process-local resources."""
        # Ensure manager is initialized
        self.ensure_initialized()
        
        logger.debug(f"get_worker_tokenizer called from process {os.getpid()}")
        
        cache_key = f"{worker_id}_{model_name}"
        
        try:
            # Check if tokenizer exists in cache
            if cache_key in self._local.process_tokenizers:
                tokenizer = self._local.process_tokenizers[cache_key]
                if tokenizer is not None:
                    logger.info(f"Using cached tokenizer for worker {worker_id}")
                    return tokenizer
                else:
                    # Remove invalid tokenizer from cache
                    logger.warning(f"Removing invalid tokenizer from cache for worker {worker_id}")
                    del self._local.process_tokenizers[cache_key]
                    self._local.tokenizer_refs.pop(cache_key, None)

            logger.info(f"Creating new tokenizer for worker {worker_id} in process {os.getpid()}")
            
            # Create tokenizer with error handling
            try:
                logger.info(f"Creating tokenizer in process {os.getpid()}")
                
                # Validate model name before initialization
                if not isinstance(model_name, str) or not model_name.strip():
                    raise ValueError("Invalid model name provided")
                    
                # Suppress transformers warnings during tokenizer loading
                transformers_logging.set_verbosity_error()
                
                try:
                    # Create tokenizer with fast tokenization enabled
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    
                    # Verify we got a fast tokenizer
                    if not isinstance(tokenizer, PreTrainedTokenizerFast):
                        raise RuntimeError(f"Failed to get fast tokenizer for model '{model_name}'")
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to load tokenizer for model '{model_name}': {str(e)}")
                finally:
                    # Restore normal logging
                    transformers_logging.set_verbosity_warning()
                    
                # Verify tokenizer was created successfully
                if tokenizer is None:
                    raise ValueError("Tokenizer creation failed")
                    
            except Exception as e:
                logger.error(f"Error creating tokenizer in process {os.getpid()}: {str(e)}")
                raise
            
            # Store tokenizer in cache and track reference
            self._local.process_tokenizers[cache_key] = tokenizer
            self._local.tokenizer_refs[cache_key] = tokenizer
            logger.info(f"Successfully created tokenizer for worker {worker_id} in process {os.getpid()}")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to get worker tokenizer in process {os.getpid()}: {str(e)}")
            # Clean up on error
            if cache_key in self._local.process_tokenizers:
                del self._local.process_tokenizers[cache_key]
                self._local.tokenizer_refs.pop(cache_key, None)
            raise

    def cleanup_worker(self, worker_id: int):
        """Cleanup worker's tokenizer resources for current process."""
        self.ensure_initialized()
        
        try:
            # Remove tokenizers for this worker
            keys_to_remove = [k for k in self._local.process_tokenizers.keys() if k.startswith(f"{worker_id}_")]
            for key in keys_to_remove:
                if key in self._local.process_tokenizers:
                    del self._local.process_tokenizers[key]
                    self._local.tokenizer_refs.pop(key, None)
            
            logger.info(f"Cleaned up tokenizer resources for worker {worker_id} in process {os.getpid()}")
            
        except Exception as e:
            logger.error(f"Error during worker cleanup: {str(e)}")

tokenizer_manager = TokenizerManager()

__all__ = ['TokenizerManager', 'tokenizer_manager']
