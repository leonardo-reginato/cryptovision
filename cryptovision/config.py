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
PARAMS = {
    "val_size":0.15,
    "test_size": 0.15,
    "random_state": 42,
    "stratify_by": "folder_label",
    "data_aug":{
        "flip": 'horizontal',
        "rotation": 0.2,
        "zoom": 0.2,
        "translation": (0.1, 0.1),
        "contrast": 0.2,
        "brightness": 0.2,
    },
    "img_size": (299, 299),
    "batch_size": 64,
    "model":{
      "base_model":'ResNet50V2',
      "base_model_short": 'rn50v2',
      "weights": 'imagenet',
      "trainable": False,
      "dropout": 0.2,
      "unfreeze_layers": False,
      "shared_layer": 512,
      "family_transform": 256,
      "family_attention": 512,
      "genus_transform": 256,
      "genus_residual": 256,
      "genus_attention": 512,
      "species_transform": 256,
      "species_residual": 256,
      "loss": "categorical_crossentropy",
      "metrics": ["accuracy", "AUC", "Precision", "Recall"],
      "epochs": 10,
      'learning_rate': 0.0001,
      "ftun_last_layers": 70,
      "ftun_learning_rate": 0.00001,
      "ftun_epochs": 10,
    },
    "verbose":1,
}


try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
