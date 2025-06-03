import datetime
import os
import random
import warnings
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from loguru import logger

import cryptovision.dataset as dataset
from cryptovision import utils
from cryptovision.models import CryptoVisionModels as cv_models
from cryptovision.train import train_with_wandb


def load_training_settings(settings_file_path: str) -> dict:
    with open(settings_file_path, "r") as f:
        settings = yaml.safe_load(f)
    if settings.get("version") == "auto":
        settings["version"] = datetime.datetime.now().strftime("%y%m.%d.%H%M")
    return settings


def rename_image_path(df, src_path):
    df["image_path"] = df["image_path"].apply(
        lambda x: x.replace("/Volumes/T7_shield/CryptoVision/Data/Sources", src_path)
    )
    return df


def main():
    logger.info("Starting grid search pipeline...")

    # Load training settings from YAML
    settings = load_training_settings("cryptovision/settings.yaml")

    if settings.get("suppress_warnings", False):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        warnings.filterwarnings("ignore")

    # Setup grid search parameters – either from the settings or defaults
    grid_params = settings.get("grid_search", {})
    arch_list = grid_params.get("architecture", ["gated", "concat", "std", "att"])
    loss_list = grid_params.get("loss_type", ["cfc", "tfcl"])
    seed_list = grid_params.get("seed", [42, 1, 17])

    combinations = list(product(arch_list, loss_list, seed_list))

    grid_csv_path = "grid_search_results.csv"
    if not os.path.exists(grid_csv_path):
        df_params = pd.DataFrame(
            combinations,
            columns=[
                "architecture",
                "loss_type",
                "seed",
            ],
        )
        df_params["status"] = "pending"
        df_params.to_csv(grid_csv_path, index=False)
    else:
        df_params = pd.read_csv(grid_csv_path)

    total_combinations = len(df_params)
    for idx, row in df_params.iterrows():
        if row["status"] == "done":
            continue
        logger.info(
            f"Grid search combination {idx + 1}/{total_combinations}: {row.to_dict()}"
        )
        try:
            # Update settings with the current grid search parameters
            settings["architecture"] = row["architecture"]
            settings["loss_type"] = row["loss_type"]
            settings["seed"] = row["seed"]
            settings["version"] = datetime.datetime.now().strftime("%y%m.%d.%H%M")

            SEED = settings["seed"]
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            # Update tags with grid search information
            settings["tags"] = [
                f"{settings['architecture'].upper()}",
                f"{settings['loss_type'].upper()}",
                f"SEED{settings['seed']}",
                "GridSearch",
            ]

            # Load dataset
            data = {}
            tf_data = {}
            data["train"], data["val"], data["test"] = dataset.load_dataset(
                src_path=settings["data_path"],
                min_samples=settings["samples_threshold"],
                return_split=True,
                stratify_by="species",
                test_size=settings["test_size"],
                val_size=settings["validation_size"],
                random_state=settings["seed"],
            )

            data["train"] = rename_image_path(data["train"], settings["data_path"])
            data["val"] = rename_image_path(data["val"], settings["data_path"])
            data["test"] = rename_image_path(data["test"], settings["data_path"])

            tf_data["train"] = utils.tensorflow_dataset(
                data["train"],
                batch_size=settings["batch_size"],
                image_size=(settings["image_size"], settings["image_size"]),
                shuffle=False,
            )
            tf_data["val"] = utils.tensorflow_dataset(
                data["val"],
                batch_size=settings["batch_size"],
                image_size=(settings["image_size"], settings["image_size"]),
                shuffle=False,
            )
            tf_data["test"] = utils.tensorflow_dataset(
                data["test"],
                batch_size=settings["batch_size"],
                image_size=(settings["image_size"], settings["image_size"]),
                shuffle=False,
            )

            # Build the model using the updated grid parameters
            model = cv_models.basic(
                imagenet_name=settings["pretrain"],
                augmentation=cv_models.augmentation_layer(
                    image_size=settings["image_size"], seed=settings["seed"]
                ),
                input_shape=(settings["image_size"], settings["image_size"], 3),
                shared_dropout=settings["shared_dropout"],
                feat_dropout=settings["features_dropout"],
                shared_layer_neurons=settings["shared_layer_neurons"],
                pooling_type=settings["pooling_type"],
                architecture=settings["architecture"],
                se_block=settings["se_block"],
                output_neurons=(
                    data["test"]["family"].nunique(),
                    data["test"]["genus"].nunique(),
                    data["test"]["species"].nunique(),
                ),
            )

            # Set Loss function
            parent_genus, parent_species = utils.make_parent_lists(
                data["train"]["family"].tolist(),
                data["train"]["genus"].tolist(),
                data["train"]["species"].tolist(),
            )

            if settings["loss_type"] == "cfc":
                family_loss, genus_loss, species_loss = utils.loss_factory(
                    loss_type="cfc"
                )
                logger.warning("Loss Function selected -> CFC")

            elif settings["loss_type"] == "tfcl":
                family_loss, genus_loss, species_loss = utils.loss_factory(
                    loss_type="tfcl",
                    parent_genus=parent_genus,
                    parent_species=parent_species,
                    alpha=0.1,
                    beta=0.1,
                    gamma=2.0,
                    smoothing=0.1,
                    from_logits=False,
                )
                logger.warning("Loss Function selected -> TFCL")

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=settings["lr"]),
                loss={
                    "family": family_loss,  # top level: no consistency penalty
                    "genus": genus_loss,  # includes soft penalty
                    "species": species_loss,  # includes soft penalty
                },
                metrics={
                    key: settings["metrics"] for key in ["family", "genus", "species"]
                },
                loss_weights={
                    "family": settings["loss_weights"][0],
                    "genus": settings["loss_weights"][1],
                    "species": settings["loss_weights"][2],
                },
            )

            # Initial training phase
            model = train_with_wandb(
                project_name=settings["project_name"],
                experiment_name=settings.get("experiment_name"),
                tags=settings["tags"],
                config=settings,
                model=model,
                datasets=tf_data,
                save=True,
                parent_genus=parent_genus,
                parent_species=parent_species,
            )

            df_params.loc[idx, "status"] = "done"
            df_params.to_csv(grid_csv_path, index=False)

        except Exception as e:
            logger.error(f"Error in combination {idx}: {e}")
            df_params.loc[idx, "status"] = "error"
            df_params.to_csv(grid_csv_path, index=False)
            continue

    logger.success("Grid search pipeline finished.")


if __name__ == "__main__":
    # Settings
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Optionally limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
