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

PROJ_NAME = "CryptoVision - Training"
PROJ_TASK = "HACPL Model Training"

# Parameters
PARAMS = {
    "run_sufix": "Proteon 299",            # Suffix for run name
    "img_size": (299, 299),                # Image dimensions (height, width)
    "batch_size": 64,                      # Batch size for training and validation
    "val_size": 0.15,                      # Validation dataset proportion
    "test_size": 0.15,                     # Test dataset proportion
    "random_state": 42,                    # Seed for dataset splitting
    "stratify_by": "folder_label",         # Column to stratify dataset splits

    # Data Augmentation parameters
    "data_aug": {
        "flip": "horizontal",              # Options: 'horizontal', 'vertical', 'horizontal_and_vertical'
        "rotation": 0.2,                   # Max rotation angle in radians
        "zoom": 0.2,                       # Zoom range as a float
        "translation": (0.1, 0.1),         # Horizontal and vertical translation factors
        "contrast": 0.2,                   # Contrast adjustment factor
        "brightness": 0.2,                 # Brightness adjustment factor
    },

    # Model configuration
    "model": {
        "base_model": "ResNet50V2",        # Base model to use for feature extraction
        "base_model_short": "rn50v2",      # Short name for model logging and saving
        "weights": "imagenet",             # Pre-trained weights (e.g., 'imagenet')
        "trainable": False,                # Initial trainability of the base model
        "dropout": 0.3,                    # Dropout rate for regularization
        "shared_layer": 512,               # Neurons in the shared dense layer
        "family_hidden": 512,              # Neurons in the family-level dense layer
        "genus_hidden": 512,               # Neurons in the genus-level dense layer
        "species_hidden": 512,             # Neurons in the species-level dense layer
        "attention_neurons": 512,          # Neurons in the attention layer
        "early_stopping_patience": 10,     # Patience for early stopping
        "lr_factor": 0.5,                  # Learning rate factor
        "lr_patience": 5,                  # Patience for learning rate reduction
        "lr_min": 1e-6,                    # Minimum learning rate

        # Training parameters
        "learning_rate": 1e-4,             # Initial learning rate
        "epochs": 20,                      # Number of epochs for initial training
        "ftun_last_layers": 70,            # Number of last layers to unfreeze in fine-tuning
        "ftun_learning_rate": 1e-5,        # Learning rate for fine-tuning
        "ftun_epochs": 10,                 # Number of epochs for fine-tuning
        "loss_weights": {"family": 1.0, "genus": 0.8, "species": 0.6},  # Loss weighting for each output
    },

    # Evaluation metrics
    "metrics": ["accuracy", "AUC", "Precision", "Recall"],  # Metrics for each taxonomic level

    # Logging and verbosity
    "verbose": 0,                          # Verbosity mode for training output
}


try:
    from tqdm import tqdm
    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
