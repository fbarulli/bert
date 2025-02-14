# preprocess.py (same as before)
import logging
from pathlib import Path
from transformers import BertTokenizer
from simpler_fine_bert.dataset import MLMDataset
from simpler_fine_bert.utils import setup_logging, seed_everything
import yaml

def preprocess_data(data_path: str, model_name: str, max_length: int, train_ratio: float = 1.0):
    """Pre-tokenizes data and creates memmap files."""
    setup_logging()
    seed_everything(42) #Seed for reproducibility
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing data from {data_path}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = MLMDataset(
        data_path=Path(data_path),
        tokenizer=tokenizer,
        max_length=max_length,
        split='train',  # Process entire dataset
        train_ratio=train_ratio
    )
    # Memmap files are created during dataset initialization

    logger.info("Preprocessing complete. Memmap files created.")

if __name__ == "__main__":
    # Example Usage (load relevant parts of the config for preprocessing)
    config = yaml.safe_load(open("config.yaml", 'r'))
    data_path = config['data']['csv_path']
    model_name = config['model']['name']
    max_length = config['data']['max_length']
    train_ratio = config['data']['train_ratio']
    preprocess_data(data_path, model_name, max_length, train_ratio)