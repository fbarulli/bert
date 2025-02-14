# Simpler Fine BERT

A simplified BERT finetuning package with MLM and classification stages, optimized for running on Google Colab.

## Features

### MLM Stage
- Whole Word Masking
- Span Masking
- Focal Loss for hard examples
- SimCSE for better embeddings
- R-Drop regularization

### Classification Stage
- Uses MLM-finetuned embeddings
- Layer-wise learning rates
- Enhanced feature extraction
- Gradient checkpointing

### Optimization
- Multi-objective optimization
- Parameter constraints
- Historical performance tracking
- Advanced pruning strategies

### Monitoring
- Comprehensive wandb logging
- Resource tracking
- Gradient statistics
- Advanced visualizations

## Running on Colab

1. Clone and install:
```python
# Clone repository
!git clone https://github.com/yourusername/simpler_fine_bert.git
%cd simpler_fine_bert

# Install dependencies
!pip install -e .
```

2. Create sample data:
```python
# Create a simple dataset
!echo "text,label\nThis is a positive example,positive\nThis is a negative example,negative" > sample.csv
```

3. Run training stages:

### Training Stages

The training process consists of 4 stages - 2 for MLM (Masked Language Modeling) and 2 for fine-tuning:

#### MLM Stage 1: Hyperparameter Optimization
Run Optuna trials to find optimal hyperparameters for MLM:
```bash
python train.py mlm-trials \
    --config config_mlm.yaml \
    --study-name "mlm_optimization" \
    --output-dir "./output/mlm"
```

#### MLM Stage 2: Full Training
Train on full dataset using best hyperparameters from Stage 1:
```bash
python train.py mlm-train \
    --config config_mlm.yaml \
    --best-trial "./output/mlm/best_trial.json" \
    --output-dir "./output/mlm_final"
```

#### Fine-tuning Stage 1: Hyperparameter Optimization
Run Optuna trials to find optimal hyperparameters for fine-tuning:
```bash
python train.py finetune-trials \
    --config config_finetune.yaml \
    --study-name "finetune_optimization" \
    --mlm-model "./output/mlm_final/model" \
    --output-dir "./output/finetune"
```

#### Fine-tuning Stage 2: Full Training
Train on full dataset using best hyperparameters from Stage 3:
```bash
python train.py finetune-train \
    --config config_finetune.yaml \
    --best-trial "./output/finetune/best_trial.json" \
    --mlm-model "./output/mlm_final/model" \
    --output-dir "./output/finetune_final"
```

Each stage outputs:
- Trained model checkpoints
- Training metrics and logs
- Wandb tracking (if enabled)
- Best hyperparameters (for optimization stages)

## Configuration

The `config_mlm.yaml` file is already optimized for Colab:

```yaml
# Model configuration
model_name: 'bert-base-uncased'
max_length: 512
num_labels: 2
hidden_dim: 768
dropout_rate: 0.1

# Training configuration
num_epochs: 10
batch_size: 16  # Reduced for Colab
gradient_accumulation_steps: 2  # For memory efficiency
num_workers: 2  # Reduced for Colab
fp16: true  # Mixed precision training

# MLM configuration
mlm_weight: 1.0
simcse_weight: 0.1
simcse_temperature: 0.05
rdrop_alpha: 1.0

# Data configuration
data:
  train_csv_path: 'sample.csv'  # Local file
  val_csv_path: 'sample.csv'    # Local file
  text_column: 'text'
  label_column: 'label'
```

## Memory Optimization

The package includes several memory optimizations for Colab:
- Gradient checkpointing
- Mixed precision training (fp16)
- Gradient accumulation
- Reduced batch size
- Efficient data loading

## Monitoring

Training progress can be monitored through:
1. Weights & Biases dashboard
2. Local CSV logs
3. TensorBoard metrics
4. Console output

## Outputs

The training process creates:
1. MLM stage outputs:
   - Best MLM model
   - Training logs
   - Metrics
2. Classification stage outputs:
   - Best classification model
   - Evaluation results
   - Performance plots

## Requirements

All dependencies are handled by setup.py:
- PyTorch >= 1.9.0
- Transformers >= 4.5.0
- Optuna >= 3.0.0
- Other ML libraries (numpy, pandas, etc.)

## Troubleshooting

Common Colab issues:
1. Out of memory:
   - Reduce batch_size in config
   - Enable gradient_accumulation
   - Use gradient_checkpointing

2. Runtime disconnection:
   - Enable checkpointing
   - Use wandb for tracking
   - Save frequent backups

## License

This project is licensed under the MIT License - see the LICENSE file for details.
