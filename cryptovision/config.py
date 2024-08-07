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
    "pre_trained_model": tf.keras.applications.efficientnet_v2.EfficientNetV2B2,
    "img_preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
    "target": "species",
    "min_images_trashold": 50,
    "test_size": 0.2,
    "random_state": 42,
    "image_shape": (224, 224),
    "batch_size": 32,
    "unfreeze_layers": 10,
    "num_layer": 1,
    "neurons": [512],
    "batch_norm": True,
    "dropout": 0.2,
    "l1": 0.01,
    "l2": 0.001,
    "learning_rate": 0.0001,
    "epochs": 100,
    "lr_patience": 3,
    "early_stopping_patience": 5,
}

IMG_GEN_PARAMS = dict(
    rotation_range=40,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.35,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    preprocessing_function=MODEL_PARAMS["img_preprocess"],
    validation_split=0.20,
)


try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
