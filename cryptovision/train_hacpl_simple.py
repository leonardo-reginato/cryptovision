import wandb
import typer
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from cryptovision.config import PARAMS, PROCESSED_DATA_DIR, PROJ_NAME
from cryptovision.tools import (
    image_directory_to_pandas,
    split_image_dataframe,
    tf_dataset_from_pandas,
)
from cryptovision.ai_architecture import (
    proteon_model, augmentation_layer, simple_hacpl_model)

# Initialize Typer app and set mixed precision for TensorFlow
app = typer.Typer()
wandb.require("core")
tf.keras.mixed_precision.set_global_policy("mixed_float16")


run_sufix = PARAMS["run_sufix"]

# Matrics Logger with Wandb
def log_metrics(model, target_dataset, prefix,):
    
    logger.info(f"Evaluating {prefix} metrics...")
    
    # Evaluate the model
    results = model.evaluate(target_dataset, return_dict=True, verbose=0)
    
    # Log metrics to WandB
    for metric_name, metric_value in results.items():
        wandb.log({f"{prefix}_{metric_name}": metric_value})
        
        # Log metrics to console
        logger.info(f"{prefix} {metric_name}: {metric_value:.3f}")
    
    return True

@app.command()
def main(
    dataset_dir: Path = PROCESSED_DATA_DIR / "cv_images_dataset"
):
    # Initialize Weights and Biases run
    with wandb.init(
        project=PROJ_NAME,
        name=f"{PARAMS['model']['base_model_short']} - {run_sufix}",
        config={**PARAMS},
    ) as run:
        
        # Dataset Setup
        image_df = image_directory_to_pandas(dataset_dir)
        train_df, val_df, test_df = split_image_dataframe(
            image_df,
            test_size=PARAMS["test_size"],
            val_size=PARAMS["val_size"],
            random_state=PARAMS["random_state"],
            stratify_by=PARAMS["stratify_by"],
        )

        logger.info(
            f"Train: {len(train_df)} ({len(train_df)/len(image_df) * 100:.2f} %), "
            f"Val: {len(val_df)} ({len(val_df)/len(image_df) * 100:.2f} %), "
            f"Test: {len(test_df)} ({len(test_df)/len(image_df) * 100:.2f} %)"
        )

        # Dataset Preparation
        train_ds, family_labels, genus_labels, species_labels = tf_dataset_from_pandas(
            train_df, PARAMS["batch_size"], PARAMS["img_size"]
        )
        val_ds, _, _, _ = tf_dataset_from_pandas(val_df, PARAMS["batch_size"], PARAMS["img_size"])
        test_ds, _, _, _ = tf_dataset_from_pandas(test_df, PARAMS["batch_size"], PARAMS["img_size"])

        # Define Data Augmentation
        data_augmentation = augmentation_layer(
            flip=PARAMS["data_aug"]["flip"],
            rotation=PARAMS["data_aug"]["rotation"],
            zoom=PARAMS["data_aug"]["zoom"],
            translation=(0.1, 0.1),
            contrast=PARAMS["data_aug"]["contrast"],
            brightness=PARAMS["data_aug"]["brightness"],
        )

        # Model Creation
        model = simple_hacpl_model(
            n_families=len(family_labels), 
            n_genera=len(genus_labels), 
            n_species=len(species_labels), 
            input_shape=PARAMS['img_size'] + (3,), 
            base_weights="imagenet", 
            base_trainable=False, 
            augmentation_layer=data_augmentation,
            shared_layer_neurons=PARAMS['model']['shared_layer'],
            shared_layer_dropout=PARAMS['model']['dropout'],
            genus_hidden_neurons=PARAMS['model']['genus_hidden'],
            specie_hidden_neurons=PARAMS['model']['species_hidden'],
        )

        # Model Compilation
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=PARAMS["model"]["learning_rate"]),
            loss={
                'family': 'categorical_focal_crossentropy',
                'genus': 'categorical_focal_crossentropy',
                'species': 'categorical_focal_crossentropy',
            },
            metrics={
                "family": PARAMS["metrics"],
                "genus": PARAMS["metrics"],
                "species": PARAMS["metrics"],
            },
            loss_weights=PARAMS["model"]["loss_weights"],
        )

        # Log Metrics before Training
        log_metrics(model, test_ds, "pre-train")

        # Training Phase
        wandb_logger = WandbMetricsLogger()
        
        # Define Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PARAMS["model"]["early_stopping_patience"],
            restore_best_weights=True,
        )
        
        # Define Reduce Learning Rate on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=PARAMS["model"]["lr_factor"],
            patience=PARAMS["model"]["lr_patience"],
            min_lr=PARAMS["model"]["lr_min"],
        )
        
        # Define Model Checkpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"/Users/leonardo/Documents/Projects/cryptovision/models/hapcl_model.keras",
            monitor="val_loss",
            save_best_only=True,
        )
        
        history = model.fit(
            train_ds,
            epochs=PARAMS["model"]["epochs"],
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, model_checkpoint],
            verbose=PARAMS["verbose"],
        )

        # Log Metrics after Training
        log_metrics(model, test_ds, "trained")

        # Fine-tuning Phase
        base_model = model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-PARAMS["model"]["ftun_last_layers"]]:
            layer.trainable = False

        logger.info(f"Unfreezing the last {PARAMS['model']['ftun_last_layers']} layers")
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=PARAMS["model"]["ftun_learning_rate"]),
            loss={
                'family': 'categorical_focal_crossentropy',
                'genus': 'categorical_focal_crossentropy',
                'species': 'categorical_focal_crossentropy',
            },
            metrics={
                "family": PARAMS["metrics"],
                "genus": PARAMS["metrics"],
                "species": PARAMS["metrics"],
            },
            loss_weights=PARAMS["model"]["loss_weights"],
        )

        total_epochs = PARAMS["model"]["epochs"] + PARAMS["model"]["ftun_epochs"]
        
        history_fine = model.fit(
            train_ds,
            epochs=total_epochs,
            initial_epoch=len(history.epoch),
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, model_checkpoint],
            verbose=PARAMS["verbose"],
        )

        # Log Metrics after Fine-tuning
        log_metrics(model, test_ds, "fine-tuned")

        # Model Saving
        today = datetime.now().strftime("%y%m%d%H%M")
        model_path_name = (
            f"/Users/leonardo/Documents/Projects/cryptovision/models/hacpl_{PARAMS['model']['base_model_short']}_{PARAMS['img_size'][0]}_{run_sufix}_{today}.keras"
        )
        model.save(model_path_name)

        wandb.log_artifact(
            model_path_name,
            name=f"hacpl_{PARAMS['model']['base_model_short']}_{PARAMS['img_size'][0]}_{run_sufix}_{today}",
            type="model",
        )

        logger.success(f"Model {PARAMS['model']['base_model']} trained and logged to wandb.")

if __name__ == "__main__":
    app()
