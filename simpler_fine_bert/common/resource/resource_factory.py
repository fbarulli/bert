from __future__ import annotations
import torch 
from typing import Dict, Any, Optional, Type, Callable, TypeVar
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass
from transformers import BertConfig

from simpler_fine_bert.common.cuda_utils import get_cuda_device
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

def get_data_manager():
    """Get data manager instance at runtime to avoid circular imports."""
    from simpler_fine_bert.common.data_manager import data_manager
    return data_manager

def get_embedding_model():
    """Get EmbeddingBert class at runtime to avoid circular imports."""
    from simpler_fine_bert.embedding.model import EmbeddingBert
    return EmbeddingBert

def get_dataset_class(stage: str = 'embedding'):
    """Get appropriate dataset class based on stage."""
    if stage == 'embedding':
        from simpler_fine_bert.embedding.dataset import EmbeddingDataset
        return EmbeddingDataset
    elif stage == 'classification':
        from simpler_fine_bert.classification.dataset import CSVDataset
        return CSVDataset
    else:
        raise ValueError(f"Unknown stage: {stage}")

def create_dataset(config: Dict[str, Any], split: str = 'train', stage: str = 'embedding') -> Dataset:
    """Create a dataset instance.
    
    Args:
        config: Configuration dictionary
        split: Dataset split to create
        stage: Stage of training ('embedding' or 'classification')
        
    Returns:
        Created Dataset instance
    """
    DatasetClass = get_dataset_class(stage)
    return DatasetClass(
        data_path=Path(config['data']['csv_path']),
        tokenizer=get_data_manager().get_tokenizer(config),
        max_length=config['data']['max_length'],
        split=split,
        train_ratio=float(config['data']['train_ratio'])
    )

def create_dataloader(config: Dict[str, Any], dataset: Optional[Dataset] = None, split: str = 'train') -> DataLoader:
    """Create a dataloader instance.
    
    Args:
        config: Configuration dictionary
        dataset: Optional dataset to create loader for
        split: Dataset split if creating new dataset
        
    Returns:
        Created DataLoader instance
    """
    return get_data_manager().create_dataloader(config=config, dataset=dataset, split=split)

T = TypeVar('T')

@dataclass
class ResourceType:
    """Configuration for a registered resource type."""
    name: str
    description: str
    factory: Callable[..., Any]
    validator: Optional[Callable[[Any], bool]] = None

class ResourceFactory:
    """Factory for creating process-local resources."""
    
    def __init__(self):
        self._resource_types: Dict[str, ResourceType] = {
            'dataset': ResourceType(
                name='dataset',
                description='Dataset resources',
                factory=create_dataset,
                validator=lambda x: isinstance(x, Dataset)
            ),
            'dataloader': ResourceType(
                name='dataloader',
                description='DataLoader resources',
                factory=create_dataloader,
                validator=lambda x: isinstance(x, DataLoader)
            ),
            'model': ResourceType(
                name='model',
                description='Model resources',
                factory=lambda config: self.create_model(config),
                validator=lambda x: isinstance(x, torch.nn.Module)
            )
        }

    def create_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Create a model instance based on configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If model type is invalid
            RuntimeError: If model creation fails
        """
        try:
            # Get tokenizer through manager
            tokenizer = get_data_manager().get_tokenizer(config)
            
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
                tie_word_embeddings=True,  # Enable weight tying
                safe_serialization=False  # Disable safe serialization
            )
            
            # Initialize model with pre-trained weights
            EmbeddingBert = get_embedding_model()
            model = EmbeddingBert.from_pretrained(
                config['model']['name'],
                config=model_config,
                tie_weights=None  # Let HuggingFace handle weight tying
            )

            # Freeze all layers first
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze last 3 encoder layers
            for i in range(9, 12):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

            # Unfreeze embedding head
            for param in model.cls.parameters():
                param.requires_grad = True

            # Log parameter counts
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"Initialized model with:\n"
                f"- Total parameters: {total_params:,}\n"
                f"- Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)\n"
                f"- Frozen parameters: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)"
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def register_resource_type(
        self,
        name: str,
        description: str,
        factory: Callable[..., T],
        validator: Optional[Callable[[Any], bool]] = None
    ) -> None:
        """Register a new resource type with the factory.
        
        Args:
            name: Unique identifier for the resource type
            description: Human-readable description
            factory: Function to create resources of this type
            validator: Optional function to validate resources of this type
        
        Raises:
            ValueError: If resource type already exists
        """
        try:
            if name in self._resource_types:
                raise ValueError(f"Resource type '{name}' already registered")
            
            self._resource_types[name] = ResourceType(
                name=name,
                description=description,
                factory=factory,
                validator=validator
            )
            logger.info(f"Registered new resource type: {name}")
            
        except Exception as e:
            logger.error(f"Failed to register resource type '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_resource(
        self,
        resource_type: str,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        """Create a single resource of the specified type.
        
        Args:
            resource_type: Type of resource to create
            config: Configuration for resource creation
            **kwargs: Additional arguments passed to factory function
        
        Returns:
            Created resource instance
            
        Raises:
            ValueError: If resource type not registered
            RuntimeError: If resource creation fails
        """
        try:
            if resource_type not in self._resource_types:
                raise ValueError(f"Unknown resource type: {resource_type}")
                
            resource_config = self._resource_types[resource_type]
            resource = resource_config.factory(config, **kwargs)
            
            if resource_config.validator and not resource_config.validator(resource):
                raise RuntimeError(
                    f"Created resource failed validation for type {resource_type}"
                )
                
            logger.debug(f"Created resource of type {resource_type}")
            return resource
            
        except Exception as e:
            logger.error(f"Failed to create resource of type {resource_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_resources(
        self,
        config: Dict[str, Any],
        device_id: Optional[int] = None,
        cache_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create fresh resources for a process."""
        try:
            # Get process-specific device
            device = get_cuda_device(device_id)
            
            # Create all registered resource types
            resources = {'device': device}
            for resource_type in self._resource_types:
                try:
                    resource = self.create_resource(resource_type, config)
                    if resource is not None:
                        resources[resource_type] = resource
                except Exception as e:
                    logger.warning(
                        f"Failed to create optional resource {resource_type}: {str(e)}"
                    )
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to create resources: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_resource_config(self, resource: Any) -> Dict[str, Any]:
        """Extract configuration that can be used to recreate a resource."""
        if isinstance(resource, DataLoader):
            return {
                'type': 'DataLoader',
                'batch_size': resource.batch_size,
                'num_workers': resource.num_workers,
                'dataset': self.get_resource_config(resource.dataset),
                'shuffle': resource.shuffle
            }
        elif isinstance(resource, Dataset):
            return {
                'type': resource.__class__.__name__,
                'params': {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in resource.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }
            }
        return None

resource_factory = ResourceFactory()
