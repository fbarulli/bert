# Simpler Fine BERT

A simplified BERT finetuning package with embedding learning and classification stages, optimized for single GPU training.

## Features

### Embedding Stage
- Masked Language Modeling (MLM)
- Whole Word Masking
- Resource-efficient training
- Mixed precision (FP16)

### Classification Stage
- Uses MLM-finetuned embeddings
- Hyperparameter optimization
- Resource monitoring
- Gradient checkpointing

### Optimization
- Optuna trials for hyperparameters
- Resource-aware worker scaling
- Memory management
- Early stopping

### Monitoring
- Weights & Biases logging
- Resource tracking
- Process monitoring
- Performance metrics

## Training Stages

The training process consists of three main stages:

### Stage 1: Train Embedding Model
Train the BERT model to learn better embeddings through MLM:
```python
from simpler_fine_bert.embedding.embedding_training import train_embeddings
from simpler_fine_bert.common.config_utils import load_config
from pathlib import Path

config = load_config("config_embedding.yaml")
output_dir = Path("outputs")
loss, metrics = train_embeddings(config, output_dir)
```

### Stage 2: Optimize Classification
Run Optuna trials to find optimal classification hyperparameters:
```python
from simpler_fine_bert.classification.classification_training import run_classification_optimization
from simpler_fine_bert.common.config_utils import load_config

best_params = run_classification_optimization(
    embedding_model_path="outputs/embedding_stage/best_model",  # Path to your best embedding model
    config_path="config_finetune.yaml",  # Classification config
    study_name="classification_study"  # Optional study name
)
```

### Stage 3: Train Final Classification Model
Train the final classification model using the best parameters:
```python
from simpler_fine_bert.classification.classification_training import train_final_model
from simpler_fine_bert.common.config_utils import load_config
from pathlib import Path

train_final_model(
    embedding_model_path="outputs/embedding_stage/best_model",  # Path to your best embedding model
    best_params=best_params,  # Parameters from optimization
    config_path="config_finetune.yaml",  # Classification config
    output_dir=Path("outputs")  # Optional output directory
)
```

## Memory Management

The package includes several memory optimizations:
- Gradient checkpointing
- Mixed precision (FP16)
- Gradient accumulation
- Resource monitoring
- Memory-aware worker scaling

## Outputs

Each training stage produces:
1. Embedding stage:
   - Best MLM model
   - Training metrics
   - Resource logs
   
2. Classification optimization:
   - Best hyperparameters
   - Trial metrics
   - Study statistics
   
3. Final classification:
   - Best model checkpoint
   - Evaluation metrics
   - Performance analysis



## License

This project is licensed under the MIT License - see the LICENSE file for details.
