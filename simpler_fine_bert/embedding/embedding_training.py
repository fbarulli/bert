from __future__ import annotations

import logging
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transformers import BertConfig

from simpler_fine_bert.embedding.embedding_trainer import EmbeddingTrainer
from simpler_fine_bert.common.tokenizer_manager import tokenizer_manager
from simpler_fine_bert.common.dataloader_manager import create_dataloaders
from simpler_fine_bert.common.wandb_manager import WandbManager
from simpler_fine_bert.common.utils import measure_memory, clear_memory
from simpler_fine_bert.embedding.model import EmbeddingBert

logger = logging.getLogger(__name__)

@measure_memory
def train_embeddings(
    config: Dict[str, Any],
    output_dir: Path,
    is_trial: bool = False,
    trial: Optional['optuna.Trial'] = None,
    wandb_manager: Optional[WandbManager] = None,
    job_id: Optional[int] = None
) -> Tuple[float, Dict[str, Any]]:
    """Train a BERT model to learn embeddings through masked token prediction.
    
    Args:
        config: Training configuration
        output_dir: Directory to save outputs
        is_trial: Whether this is an Optuna trial
        trial: Optuna trial object if is_trial=True
        wandb_manager: Optional W&B logging manager
        job_id: Optional job ID for parallel training
        
    Returns:
        Tuple of (best validation loss, metrics dictionary)
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get tokenizer through manager
        tokenizer = tokenizer_manager.get_worker_tokenizer(
            worker_id=job_id if job_id is not None else os.getpid(),
            model_name=config['model']['name'],
            model_type='embedding'
        )
        
        # Create dataloaders with efficient loading
        train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
            data_path=Path(config['data']['csv_path']),
            tokenizer=tokenizer,
            max_length=config['data']['max_length'],
            batch_size=config['training']['batch_size'],
            train_ratio=config['data']['train_ratio'],
            is_embedding=True,
            mask_prob=config['data']['embedding_mask_probability'],
            max_predictions=config['data']['max_predictions'],
            max_span_length=config['data']['max_span_length'],
            num_workers=config['data']['num_workers']
        )
        
        # Get model configuration
        model_config = BertConfig.from_pretrained(
            config['model']['name'],
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,  # Standard BERT base size
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=config['training']['hidden_dropout_prob'],
            attention_probs_dropout_prob=config['training']['attention_probs_dropout_prob'],
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=tokenizer.pad_token_id,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
        )
        
        # Initialize model with proper embedding head
        model = EmbeddingBert(
            config=model_config,
            tie_weights=True  # Important for embedding learning
        )
        
        # Load pre-trained weights if using pre-trained model
        if config['model']['type'] == 'pretrained':
            logger.info(f"Loading pre-trained weights from {config['model']['name']}")
            pretrained_dict = torch.load(
                f"{config['model']['name']}/pytorch_model.bin",
                map_location='cpu'
            )
            model_dict = model.state_dict()
            
            # Filter out embedding head weights that we want to train from scratch
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and not k.startswith('cls.')
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            logger.info(
                f"Loaded pre-trained weights:\n"
                f"- Total parameters: {sum(p.numel() for p in model.parameters())}\n"
                f"- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
            )
        
        # Initialize trainer
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            metrics_dir=output_dir / 'metrics',
            is_trial=is_trial,
            trial=trial,
            wandb_manager=wandb_manager,
            job_id=job_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        try:
            # Train model
            trainer.train(config['training']['num_epochs'])
            
            # Get best metrics
            metrics = {
                'best_embedding_loss': trainer.best_embedding_loss,
                'final_learning_rate': trainer.optimizer.param_groups[0]['lr']
            }
            
            return trainer.best_embedding_loss, metrics
            
        finally:
            clear_memory()
        
    except Exception as e:
        logger.error(
            f"Embedding training failed:\n"
            f"Error: {str(e)}\n"
            f"Config: {config}\n"
            f"Trial: {trial.number if trial else None}"
        )
        if trial and hasattr(trial, 'set_user_attr'):
            trial.set_user_attr('error', str(e))
        raise
    finally:
        clear_memory()

@measure_memory
def validate_embeddings(
    model_path: str,
    data_path: str,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Validate a trained embedding model.
    
    Args:
        model_path: Path to saved model
        data_path: Path to validation data
        config: Model configuration
        output_dir: Optional output directory for metrics
        
    Returns:
        Dictionary of validation metrics
    """
    try:
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get tokenizer
        tokenizer = tokenizer_manager.get_worker_tokenizer(
            worker_id=os.getpid(),
            model_name=config['model']['name'],
            model_type='embedding'
        )
        
        # Create dataloader
        _, val_loader, _, val_dataset = create_dataloaders(
            data_path=Path(data_path),
            tokenizer=tokenizer,
            max_length=config['data']['max_length'],
            batch_size=config['training']['batch_size'],
            train_ratio=config['data']['train_ratio'],
            is_embedding=True,
            mask_prob=config['data']['embedding_mask_probability'],
            max_predictions=config['data']['max_predictions'],
            max_span_length=config['data']['max_span_length'],
            num_workers=config['data']['num_workers']
        )
        
        # Load model configuration
        model_config = BertConfig.from_pretrained(model_path)
        
        # Initialize model with embedding head
        model = EmbeddingBert(
            config=model_config,
            tie_weights=True
        )
        
        # Load trained weights
        model.load_state_dict(
            torch.load(
                f"{model_path}/pytorch_model.bin",
                map_location='cpu'
            )
        )
        
        # Initialize trainer
        trainer = EmbeddingTrainer(
            model=model,
            train_loader=None,
            val_loader=val_loader,
            config=config,
            metrics_dir=output_dir / 'metrics' if output_dir else None,
            is_trial=False,
            val_dataset=val_dataset
        )
        
        try:
            # Run validation
            metrics = trainer.validate()
            
            # Log results
            logger.info(
                f"Validation results:\n"
                f"- Loss: {metrics['loss']:.4f}\n"
                f"- Embedding Loss: {metrics['embedding_loss']:.4f}\n"
                f"- Perplexity: {metrics['perplexity']:.2f}\n"
                f"- Accuracy: {metrics['accuracy']:.4f}\n"
                f"- Top-5 Accuracy: {metrics['top5_accuracy']:.4f}"
            )
            
            return metrics
            
        finally:
            clear_memory()
        
    except Exception as e:
        logger.error(
            f"Embedding validation failed:\n"
            f"Error: {str(e)}\n"
            f"Model path: {model_path}\n"
            f"Data path: {data_path}"
        )
        raise
    finally:
        clear_memory()
