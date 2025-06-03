import datetime
import os
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
import yaml
from loguru import logger

from cryptovision import utils
from cryptovision.dataset import load_dataset
from cryptovision.models import CryptoVisionModels as cv_models
from cryptovision.train import train_with_wandb

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Enable mixed precision for Apple Silicon (if needed)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_and_update_settings(arch: str) -> Dict[str, Any]:
    """
    Load and update settings for a specific architecture.

    Args:
        arch: Architecture name to use

    Returns:
        Updated settings dictionary
    """
    try:
        with open("cryptovision/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("settings.yaml not found in cryptovision directory")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in settings.yaml: {e}")

    # Update architecture-specific settings
    settings["architecture"] = arch
    settings["tags"].append(arch)

    # Set version if auto
    if settings.get("version") == "auto":
        settings["version"] = datetime.datetime.now().strftime("%y%m.%d.%H%M")

    return settings


def prepare_data(settings: Dict[str, Any]) -> Tuple[Dict[str, tf.data.Dataset], Any]:
    """
    Prepare and preprocess the dataset for training.

    Args:
        settings: Configuration dictionary

    Returns:
        Tuple containing:
            - Dictionary of TensorFlow datasets (train, val, test)
            - Test dataframe for reference
    """
    # Load and split dataset
    train_df, val_df, test_df = load_dataset(
        src_path=settings["data_path"],
        min_samples=settings["samples_threshold"],
        return_split=True,
        stratify_by="species",
        test_size=settings["test_size"],
        val_size=settings["validation_size"],
        random_state=settings["seed"],
    )

    # Update image paths
    def update_path(df):
        df["image_path"] = df["image_path"].apply(
            lambda x: x.replace(
                "/Volumes/T7_shield/CryptoVision/Data/Sources", settings["data_path"]
            )
        )
        return df

    train_df, val_df, test_df = map(update_path, [train_df, val_df, test_df])

    # Create TensorFlow datasets
    image_size = (settings["image_size"], settings["image_size"])
    tf_data = {
        split: utils.tensorflow_dataset(
            df,
            settings["batch_size"],
            image_size,
            shuffle=False,
        )
        for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]
    }

    return tf_data, test_df


def create_model(
    config: Dict[str, Any], output_neurons: Tuple[int, int, int]
) -> tf.keras.Model:
    """
    Create and compile the model with specified configuration.

    Args:
        config: Model configuration dictionary
        output_neurons: Tuple of (family, genus, species) output sizes

    Returns:
        Compiled Keras model
    """
    model = cv_models.basic(
        imagenet_name=config["pretrain"],
        augmentation=cv_models.augmentation_layer(image_size=config["image_size"]),
        input_shape=(config["image_size"], config["image_size"], 3),
        shared_dropout=config["shared_dropout"],
        feat_dropout=config["features_dropout"],
        shared_layer_neurons=config["shared_layer_neurons"],
        pooling_type=config["pooling_type"],
        architecture=config["architecture"],
        output_neurons=output_neurons,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss={key: config["loss"] for key in ["family", "genus", "species"]},
        metrics={key: config["metrics"] for key in ["family", "genus", "species"]},
        loss_weights={
            "family": config["loss_weights"][0],
            "genus": config["loss_weights"][1],
            "species": config["loss_weights"][2],
        },
    )

    return model


def main():
    """Main function to run architecture comparison experiments."""
    logger.info("Starting architecture comparison...")

    architectures = {
        "std": "/Users/leonardo/Documents/Projects/cryptovision/models/CVisionClassifier/2504.09.1835/final.weights.h5",
        "att": "/Users/leonardo/Documents/Projects/cryptovision/models/CVisionClassifier/2504.09.1953/final.weights.h5",
        "gated": "/Users/leonardo/Documents/Projects/cryptovision/models/CVisionClassifier/2504.14.1928/final.weights.h5",
        "concat": "/Users/leonardo/Documents/Projects/cryptovision/models/CVisionClassifier/2504.09.2233/final.weights.h5",
    }

    for arch, weights_path in architectures.items():
        logger.success(f"🔧 Running experiment for architecture: {arch.upper()}")

        # Load and update settings
        config = load_and_update_settings(arch)

        # Prepare data
        tf_data, test_df = prepare_data(config)

        # Create and compile model
        output_neurons = (
            test_df["family"].nunique(),
            test_df["genus"].nunique(),
            test_df["species"].nunique(),
        )
        model = create_model(config, output_neurons)

        # Train model
        train_with_wandb(
            project_name=config["project_name"],
            experiment_name=config["experiment_name"],
            tags=config["tags"],
            config=config,
            model=model,
            datasets=tf_data,
            save=True,
        )

    logger.success("✅ All architecture runs completed.")


if __name__ == "__main__":
    main()
