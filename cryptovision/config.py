from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

import tensorflow as tf

from datetime import datetime

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

PROJ_NAME = "CryptoVision - Training"
PROJ_TASK = "HACPL Model Training"

# Settings
SETUP = {
    # General Setup
    "seed": 42,
    "verbose": 2,
    'batch_size': 64,
    "img_size": (384, 384),
    "sufix": "phorcys_conv_att_v01",
    "arch_type": "hacpl",
    "base_model_nickname": "rn50v2",
    "version": f"v{datetime.now().strftime('%y%m%d%H%M')}",
    
    # Dataset Setup
    "val_size": 0.15,
    "test_size": 0.15,
    "stratify_by": "folder_label",
    
    # Augmentation Setup
    'aug':{
        "flip": "horizontal",
        "rotation": 0.2,
        "zoom": 0.2,
        "translation": (0.1, 0.1),
        "contrast": 0.2,
        "brightness": 0.2,
    },
    
    # Compile Setup
    "learning_rate": 1e-4,
    "loss": {
        "family": "categorical_focal_crossentropy",
        "genus": "categorical_focal_crossentropy",
        "species": "categorical_focal_crossentropy",
    },
    "metrics": {
        "family": ["accuracy", "AUC", "Precision", "Recall"],
        "genus": ["accuracy", "AUC", "Precision", "Recall"],
        "species": ["accuracy", "AUC", "Precision", "Recall"],
    },
    "loss_weights": {
        "family": 1.0,
        "genus": 1.5,
        "species": 2.0,
    },
    
    # Mode Training Setup
    "epochs": 10,
    "monitor": "val_loss",
    "early_stopping": 10,
    "restore_best_weights": True,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "lr_min": 1e-6,
    
    # Fine-tuning Setup
    "ftun_last_layers": 90,
    "ftun_learning_rate": 1e-5,
    "ftun_epochs": 15,  
}


# Model Architecture Settings
PROTEON = {
    "input_shape": SETUP['img_size'] + (3,),
    "nick_name": "proteon",
    "dropout": 0.3,
    "shared_layer": 512,
    "family_hidden": 512,
    "genus_hidden": 512,
    "species_hidden": 512,
    "attention_neurons": 512,
}


PHORCYS = {
    "input_shape": SETUP['img_size'] + (3,),
    "nick_name": "phorcys_conv",
    "dropout": 0.3,
    "shared_layer": 512,
    "genus_hidden": 256,
    "species_hidden": 128,
    "attention": True
}


try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
