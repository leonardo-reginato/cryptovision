from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

import tensorflow as tf

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Model Configuration
MODEL_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "image_shape": [224, 224],
    "batch_size": 64,
    "unfreeze_layers": False,
    "dense_layers": 1,
    "neurons": 256,
    "batch_norm": True,
    "dropout": 0.2,
    "l1": 0.0005,
    "l2": 0.0005,
    "learning_rate": 0.0001,
    "epochs": 20,
    "attention_layer":True,
    "pooling_type": "max",
    "lr_patience": 3,
    "early_stopping_patience": 3,
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy", "AUC", "Precision", "Recall"],
}


try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
