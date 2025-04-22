import datetime
import json
import os
import random
import warnings

import numpy as np
import tensorflow as tf
import yaml
from loguru import logger

import wandb
from cryptovision import tools
from wandb.integration.keras import WandbMetricsLogger

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Enable mixed precision for Apple Silicon (if needed)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Optionally limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TaxonomyAlignmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, parent_genus, parent_species):
        """
        val_ds:          tf.data.Dataset yielding (x_batch, (y_fam, y_gen, y_spe))
        parent_genus:    list[int]  mapping genus_idx -> family_idx
        parent_species:  list[int]  mapping species_idx -> genus_idx
        """
        super().__init__()
        self.val_ds = val_ds
        self.parent_genus = np.array(parent_genus, dtype=int)
        self.parent_species = np.array(parent_species, dtype=int)

    def on_epoch_end(self, epoch, logs=None):
        # 1) get predictions on the entire val set
        fam_pred, gen_pred, spe_pred = self.model.predict(self.val_ds, verbose=0)

        # 2) hard‐decisions
        pf = np.argmax(fam_pred, axis=1)
        pg = np.argmax(gen_pred, axis=1)
        ps = np.argmax(spe_pred, axis=1)

        # 3) check tree consistency
        genus_ok = self.parent_genus[pg] == pf
        species_ok = self.parent_species[ps] == pg
        alignment = np.mean(genus_ok & species_ok)

        # 4) log to Keras and to wandb
        if logs is not None:
            logs["epoch/taxo_alignment"] = alignment
        wandb.log({"epoch/taxo_alignment": alignment})
        print(f" — epoch/taxo_alignment: {alignment:.4f}")


def load_settings(settings_file_path: str) -> dict:
    """Load YAML settings and set version if 'auto'."""
    with open(settings_file_path, "r") as f:
        settings = yaml.safe_load(f)
    if settings.get("version") == "auto":
        settings["version"] = datetime.datetime.now().strftime("%y%m.%d.%H%M")
    return settings


def train_with_wandb(
    project_name: str,
    experiment_name: str,
    tags: list,
    config: dict,
    model: tf.keras.Model,
    datasets: dict,
    parent_genus: list[int],
    parent_species: list[int],
    save: bool = True,
):
    # Prepare wandb initialization arguments
    wandb_init_args = {
        "project": project_name,
        "config": config,
        "tags": tags,
    }

    if experiment_name:
        wandb_init_args["name"] = experiment_name

    # Log into wandb
    wandb.login()
    with wandb.init(**wandb_init_args) as run:
        if experiment_name is None:
            experiment_name = run.name
        logger.info(f"Wandb run started: {run.name} | ID: {run.id}")
        logger.info("Model summary before training:")
        logger.info(model.summary(show_trainable=True))

        output_dir = os.path.join("models", project_name, f"{config['version']}")
        os.makedirs(output_dir, exist_ok=True)

        # Optional save the untrained model
        if save:
            model.save_weights(os.path.join(output_dir, "untrained.weights.h5"))

        # Define callbacks
        wandb_cb = WandbMetricsLogger()
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=config["early_stop"]["monitor"],
            patience=config["early_stop"]["patience"],
            restore_best_weights=config["early_stop"]["best_weights"],
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config["reduce_lr"]["monitor"],
            factor=config["reduce_lr"]["factor"],
            patience=config["reduce_lr"]["patience"],
            min_lr=config["reduce_lr"]["min"],
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "checkpoint.weights.h5"),
            monitor=config["checkpoint"]["monitor"],
            save_best_only=config["checkpoint"]["save_best_only"],
            mode=config["checkpoint"]["mode"],
            save_weights_only=config["checkpoint"]["weights_only"],
            verbose=0,
        )

        taxo_cb = TaxonomyAlignmentCallback(
            val_ds=datasets["val"],
            parent_genus=parent_genus,
            parent_species=parent_species,
        )

        history = model.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=config["epochs"],
            callbacks=[
                wandb_cb,
                early_stop,
                reduce_lr,
                checkpoint,
                taxo_cb,
                tools.TQDMProgressBar(),
            ],
            verbose=0,
        )

        # Save training history
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(history.history, f)

        # Save the final model weights
        if save:
            model.save_weights(os.path.join(output_dir, "final.weights.h5"))

        # Save settings
        with open(os.path.join(output_dir, "settings.json"), "w") as f:
            json.dump(config, f)

        # Evaluate on the test set and log the results
        test_results = model.evaluate(datasets["test"], verbose=0, return_dict=True)
        for metric, value in test_results.items():
            wandb.log({f"test/{metric}": value})
            logger.info(f"test - {metric}: {value:.3f}")

        # compute test‐alignment exactly the same way
        fam_p, gen_p, spe_p = model.predict(datasets["test"], verbose=0)
        pf = np.argmax(fam_p, axis=1)
        pg = np.argmax(gen_p, axis=1)
        ps = np.argmax(spe_p, axis=1)
        test_align = np.mean(
            (np.array(parent_genus)[pg] == pf) & (np.array(parent_species)[ps] == pg)
        )
        wandb.log({"test/taxo_alignment": float(test_align)})
        logger.info(f"test - taxo_alignment: {test_align:.4f}")

        logger.success("wandb training pipeline completed successfully.")

    wandb.finish()
    return model


def main():
    logger.info("Starting training pipeline...")

    # Load settings from YAML
    settings = load_settings("cryptovision/settings.yaml")
    if settings.get("suppress_warnings", False):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        warnings.filterwarnings("ignore")

    data = {}
    tf_data = {}

    # Load dataset
    from cryptovision.dataset import load_dataset

    data["train"], data["val"], data["test"] = load_dataset(
        src_path=settings["data_path"],
        min_samples=settings["samples_threshold"],
        return_split=True,
        stratify_by="species",
        test_size=settings["test_size"],
        val_size=settings["validation_size"],
        random_state=SEED,
    )

    def rename_image_path(df, src_path):
        df["image_path"] = df["image_path"].apply(
            lambda x: x.replace(
                "/Volumes/T7_shield/CryptoVision/Data/Sources",
                src_path,
            )
        )
        return df

    data["train"] = rename_image_path(data["train"], settings["data_path"])
    data["val"] = rename_image_path(data["val"], settings["data_path"])
    data["test"] = rename_image_path(data["test"], settings["data_path"])

    tf_data["train"] = tools.tensorflow_dataset(
        data["train"],
        batch_size=settings["batch_size"],
        image_size=(settings["image_size"], settings["image_size"]),
        shuffle=False,
    )
    tf_data["val"] = tools.tensorflow_dataset(
        data["val"],
        batch_size=settings["batch_size"],
        image_size=(settings["image_size"], settings["image_size"]),
        shuffle=False,
    )
    tf_data["test"] = tools.tensorflow_dataset(
        data["test"],
        batch_size=settings["batch_size"],
        image_size=(settings["image_size"], settings["image_size"]),
        shuffle=False,
    )

    # Deep Learning Model's Setup
    from cryptovision.models import CryptoVisionModels as cv_models

    model = cv_models.basic(
        imagenet_name=settings["pretrain"],
        augmentation=cv_models.augmentation_layer(image_size=settings["image_size"]),
        input_shape=(settings["image_size"], settings["image_size"], 3),
        shared_dropout=settings["shared_dropout"],
        feat_dropout=settings["features_dropout"],
        shared_layer_neurons=settings["shared_layer_neurons"],
        se_block=settings["se_block"],
        pooling_type="max",
        architecture=settings["architecture"],
        output_neurons=(
            data["test"]["family"].nunique(),
            data["test"]["genus"].nunique(),
            data["test"]["species"].nunique(),
        ),
    )

    parent_genus, parent_species = tools.make_parent_lists(
        data["train"]["family"].tolist(),
        data["train"]["genus"].tolist(),
        data["train"]["species"].tolist(),
    )

    # Get loss functions from factory
    family_loss, genus_loss, species_loss = tools.loss_factory(
        loss_type=settings["loss_type"],
        parent_genus=parent_genus,
        parent_species=parent_species,
        alpha=0.1,
        beta=0.1,
        gamma=2.0,
        smoothing=0.1,
        from_logits=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["lr"]),
        loss={
            "family": family_loss,  # top level: no consistency penalty
            "genus": genus_loss,  # includes soft penalty
            "species": species_loss,  # includes soft penalty
        },
        metrics={key: settings["metrics"] for key in ["family", "genus", "species"]},
        loss_weights={
            "family": settings["loss_weights"][0],
            "genus": settings["loss_weights"][1],
            "species": settings["loss_weights"][2],
        },
    )

    train_with_wandb(
        project_name=settings["project_name"],
        experiment_name=settings["experiment_name"],
        tags=settings["tags"],
        config=settings,
        datasets=tf_data,
        model=model,
        save=True,
        parent_genus=parent_genus,
        parent_species=parent_species,
    )

    logger.success("Training Script finished.")


if __name__ == "__main__":
    main()
