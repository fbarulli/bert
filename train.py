#!/usr/bin/env python3

import os

os.environ["TRANSFORMERS_TF_ONLY"] = "1"  # Or set to "pt" for PyTorch

from simpler_fine_bert.data_manager import data_manager
import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support

if __name__ == '__main__':
    # Add freeze_support before any multiprocessing
    freeze_support()
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        if "context has already been set" in str(e):
            print("Warning: Multiprocessing context has already been set, skipping...")
        else:
            raise e

import warnings
import logging
import transformers
import os
import argparse
import traceback
import sys
import json
from pathlib import Path
import torch
import optuna
import gc
from torch.utils.data import DataLoader  # Add this import

from simpler_fine_bert import (
    train_mlm,
    validate_mlm,
    load_config,
    setup_logging,
    seed_everything,
    MLMDataset,
    WandbManager,
    OptunaManager,
    BERTTrainer,
    MLMTrainer,
    FinetunedBERT
)
from simpler_fine_bert.classification_training import run_classification_optimization, train_final_model
from simpler_fine_bert.cuda_utils import cuda_manager

logger = logging.getLogger(__name__)

def setup_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Output to console
        ]
    )
    # Disable other loggers that might interfere
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('torch.cuda').setLevel(logging.ERROR)

def init_worker():
    """Initialize worker process."""
    # No need to clear CUDA cache here anymore
    pass

def init_shared_resources(config):
    """Initialize only the essential shared resources (tokenizer and datasets)."""
    # Initialize shared resources using DataManager singleton instance
    data_resources = data_manager.init_shared_resources(config)
    tokenizer = data_resources['tokenizer']
    train_dataset = data_resources['train_dataset']
    val_dataset = data_resources['val_dataset']
    

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'tokenizer': tokenizer,
    }

def main():
    # Setup logging first thing
    setup_logging()
    
    # Initialize CUDA exactly once at the start of the program
    if __name__ == '__main__':
        # cuda_manager._setup_cuda() # Removed to initialize CUDA in worker processes
        pass

    parser = argparse.ArgumentParser(description="Train and fine-tune BERT model in stages")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    mlm_trials_parser = subparsers.add_parser('mlm-trials', help='Run MLM Optuna trials')
    mlm_trials_parser.add_argument(
        "--config",
        type=str,
        default="config_mlm.yaml",
        help="Path to MLM configuration file"
    )
    mlm_trials_parser.add_argument(
        "--study-name",
        type=str,
        help="Optuna study name"
    )
    mlm_trials_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )
    mlm_trials_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )

    mlm_train_parser = subparsers.add_parser('mlm-train', help='Train MLM with best hyperparameters')
    mlm_train_parser.add_argument(
        "--config",
        type=str,
        default="config_mlm.yaml",
        help="Path to MLM configuration file"
    )
    mlm_train_parser.add_argument(
        "--best-trial",
        type=str,
        required=True,
        help="Path to best trial parameters JSON"
    )
    mlm_train_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )

    val_parser = subparsers.add_parser('validate', help='Validate MLM model')
    val_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to MLM model"
    )
    val_parser.add_argument(
        "--config",
        type=str,
        default="config_mlm.yaml",
        help="Path to MLM configuration file"
    )
    val_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for validation metrics"
    )

    classification_opt_parser = subparsers.add_parser('classification-opt', help='Run classification optimization')
    classification_opt_parser.add_argument(
        "--mlm-model-path",
        type=str,
        required=True,
        help="Path to the pre-trained MLM model"
    )
    classification_opt_parser.add_argument(
        "--config",
        type=str,
        default="config_classification.yaml",
        help="Path to the classification configuration file"
    )
    classification_opt_parser.add_argument(
        "--study-name",
        type=str,
        help="Optuna study name for classification"
    )
    classification_opt_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for classification optimization (overrides config)"
    )

    classification_train_parser = subparsers.add_parser('classification-train', help='Train final classification model')
    classification_train_parser.add_argument(
        "--mlm-model-path",
        type=str,
        required=True,
        help="Path to the pre-trained MLM model"
    )
    classification_train_parser.add_argument(
        "--best-trial",
        type=str,
        required=True,
        help="Path to the best trial parameters JSON for classification"
    )
    classification_train_parser.add_argument(
        "--config",
        type=str,
        default="config_classification.yaml",
        help="Path to the classification configuration file"
    )
    classification_train_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for final classification training (overrides config)"
    )

    args = parser.parse_args()

    if args.command == 'mlm-trials':
        # Do one-time initialization before creating workers
        from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
        from simpler_fine_bert.dataset import create_dataloaders
        
        # Make sure we're in the main process
        if __name__ == '__main__':
            freeze_support()
            
            logger.info("Starting MLM Optuna trials...")
            config = load_config(args.config)
            
            # Create a unique study name using timestamp
            from datetime import datetime
            study_name = f"mlm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if args.study_name:
                study_name = args.study_name
            config['study_name'] = study_name

            # Create directories once
            output_dir = Path(args.output_dir if args.output_dir else config['output']['dir'])
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize shared resources before creating objective
            logger.info("Initializing shared resources...")
            shared_resources = init_shared_resources(config)
            logger.info("Shared resources initialized successfully")

            # Get n_jobs directly from config
            n_jobs = int(config['training']['n_jobs'])  # Get n_jobs from config directly
            logger.info(f"Running optimization with {n_jobs} parallel jobs")
            
            # Create Optuna study with parallel jobs
            optuna_manager = OptunaManager(
                study_name=study_name,
                config=config,
                storage_dir=output_dir,
            )

            # Initialize WandB once if needed
            wandb_manager = None
            if config['output']['wandb']['enabled']:
                wandb_manager = WandbManager(config, args.study_name or 'mlm_optimization')
                wandb_manager.init_optimization()

            from simpler_fine_bert.objective_factory import ObjectiveFactory

            # Create objective factory
            objective_factory = ObjectiveFactory(
                optuna_manager=optuna_manager,
                config=config,
                output_dir=output_dir
            )

            try:
                # Use the objective method instead of create_objective
                best_trial = optuna_manager.optimize(objective_factory.objective)

                if best_trial:
                    best_trial_path = output_dir / 'best_trial.json'
                    with open(best_trial_path, 'w') as f:
                        json.dump({
                            'hyperparameters': best_trial.params,
                            'value': best_trial.value,
                            'number': best_trial.number  # Use best_trial instead of undefined trial
                        }, f, indent=2)
                    logger.info(f"Best trial parameters saved to {best_trial_path}")
                else:
                    logger.error("No successful trials completed")

                if wandb_manager:
                    wandb_manager.finish()
            except Exception as e:
                logger.error(f"Fatal error: {e}")
                logger.error(traceback.format_exc())
                sys.exit(1)
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                sys.exit(0)

    elif args.command == 'mlm-train':
        logger.info("Training MLM with best hyperparameters...")
        config = load_config(args.config)
        output_dir = Path(args.output_dir if args.output_dir else config['output']['dir'])

        with open(args.best_trial) as f:
            best_trial = json.load(f)

        if 'hyperparameters' in best_trial:
            config['training'].update(best_trial['hyperparameters'])

            logger.info(f"Using best hyperparameters: {best_trial['hyperparameters']}")
        else:
            logger.warning("No hyperparameters found in best trial file, using default config")

        train_mlm(
            config=config,
            output_dir=output_dir,
            is_trial=False,
            gradient_checkpointing=args.gradient_checkpointing
        )
        logger.info("MLM training completed")

    elif args.command == 'validate':
        config = load_config(args.config)

        metrics = validate_mlm(
            model_path=args.model_path,
            data_path=config['data']['csv_path'],
            config=config,
            output_dir=args.output_dir
        )

        logger.info(f"Validation results: {json.dumps(metrics, indent=2)}")

    elif args.command == 'classification-opt':
        logger.info("Starting classification optimization...")
        config = load_config(args.config)
        output_dir = Path(args.output_dir if args.output_dir else config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        best_params = run_classification_optimization(
            mlm_model_path=args.mlm_model_path,
            config_path=args.config,
            study_name=args.study_name,
        )

        best_params_path = output_dir / 'best_classification_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Best classification parameters saved to {best_params_path}")

    elif args.command == 'classification-train':
        logger.info("Training final classification model...")
        config = load_config(args.config)
        output_dir = Path(args.output_dir if args.output_dir else config['output']['dir'])

        with open(args.best_trial) as f:
            best_params = json.load(f)

        train_final_model(
            mlm_model_path=args.mlm_model_path,
            best_params=best_params,
            config_path=args.config,
            output_dir=output_dir
        )

    else:
        parser.print_help()

try:
    main()
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
except KeyboardInterrupt:
    print("Keyboard interrupt detected. Exiting...")
    sys.exit(0)
