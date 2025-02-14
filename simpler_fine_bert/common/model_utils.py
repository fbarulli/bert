"""Model utilities for BERT fine-tuning."""

import logging
from pathlib import Path
from transformers import BertForMaskedLM

logger = logging.getLogger(__name__)

_model_cache = {}

def get_model_with_cache(model_name_or_path: str) -> BertForMaskedLM:
    """Get or create model with caching."""
    cache_key = str(Path(model_name_or_path).resolve())
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = BertForMaskedLM.from_pretrained(
            model_name_or_path,
            local_files_only=True
        )
        logger.info(f"Created new model for {model_name_or_path}")
    
    return _model_cache[cache_key]
