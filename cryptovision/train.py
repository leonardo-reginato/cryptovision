import wandb
import typer
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from cryptovision.config import (
    PROCESSED_DATA_DIR, PROJ_NAME, SETTINGS, 
    AUG_SETTINGS, MODEL_SETTINGS)
from cryptovision.tools import (
    image_directory_to_pandas,
    split_image_dataframe,
    tf_dataset_from_pandas,
)
from cryptovision.ai_architecture import (
    proteon_model, augmentation_layer, simple_hacpl_model)

# Initialize Typer app and set mixed precision for TensorFlow
app = typer.Typer()
tf.keras.mixed_precision.set_global_policy("mixed_float16")

run_sufix = SETTINGS['run_sufix']
settings = SETTINGS
model_settings = MODEL_SETTINGS
aug_settings = AUG_SETTINGS

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
    dataset_dir: Path = PROCESSED_DATA_DIR / "cv_images_dataset",
):
    # Initialize Weights and Biases run
    with wandb.init(
        project=PROJ_NAME,
        name = f"{model_settings['base_model_short']} - {run_sufix}",
        config={**model_settings},
    ) as run:
        
        # Dataset Setup
        image_df = image_directory_to_pandas(dataset_dir)
        train_df, val_df, test_df = split_image_dataframe(
            image_df,
            test_size=settings["test_size"],
            val_size=settings["val_size"],
            random_state=settings["random_state"],
            stratify_by=settings["stratify_by"],
        )

        logger.info(
            f"Train: {len(train_df)} ({len(train_df)/len(image_df) * 100:.2f} %), "
            f"Val: {len(val_df)} ({len(val_df)/len(image_df) * 100:.2f} %), "
            f"Test: {len(test_df)} ({len(test_df)/len(image_df) * 100:.2f} %)"
        )

        # Dataset Preparation
        train_ds, family_labels, genus_labels, species_labels = tf_dataset_from_pandas(
            train_df, settings["batch_size"], settings["img_size"]
        )
        val_ds, _, _, _ = tf_dataset_from_pandas(val_df, settings["batch_size"], settings["img_size"])
        test_ds, _, _, _ = tf_dataset_from_pandas(test_df, settings["batch_size"], settings["img_size"])

        # Define Data Augmentation
        data_augmentation = augmentation_layer(
            flip=aug_settings["flip"],
            rotation=aug_settings["rotation"],
            zoom=aug_settings["zoom"],
            translation=(0.1, 0.1),
            contrast=aug_settings["contrast"],
            brightness=aug_settings["brightness"],
        )

        # Model Creation
        model = proteon_model(
            input_shape=settings["img_size"] + (3,),
            n_families=len(family_labels),
            n_genera=len(genus_labels),
            n_species=len(species_labels),
            base_weights=model_settings["weights"],
            base_trainable=model_settings["trainable"],
            se_ratio=16,
            shared_layer_neurons=model_settings["shared_layer"],
            shared_layer_dropout=model_settings["dropout"],
            family_transform_neurons=model_settings["family_hidden"],
            genus_transform_neurons=model_settings["genus_hidden"],
            species_transform_neurons=model_settings["species_hidden"],
            attention_neurons=model_settings["attention_neurons"],
            augmentation_layer=data_augmentation
        )

        # Model Compilation
        model.compile(                                                                                                                                  
            optimizer=tf.keras.optimizers.Adam(learning_rate=model_settings["learning_rate"]),
            loss={
                'family': 'categorical_focal_crossentropy',
                'genus': 'categorical_focal_crossentropy',
                'species': 'categorical_focal_crossentropy',
            },
            metrics={
                "family": model_settings["metrics"],
                "genus": model_settings["metrics"],
                "species": model_settings["metrics"],
            },
            loss_weights=model_settings["loss_weights"],
        )

        # Log Metrics before Training
        log_metrics(model, test_ds, "pre-train")

        # Training Phase
        wandb_logger = WandbMetricsLogger()
        
        # Define Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=model_settings["early_stopping_patience"],
            restore_best_weights=True,
        )
        
        # Define Reduce Learning Rate on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=model_settings["lr_factor"],
            patience=model_settings["lr_patience"],
            min_lr=model_settings["lr_min"],
        )
        
        # Define Model Checkpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"/Users/leonardo/Documents/Projects/cryptovision/models/hapcl_model.keras",
            monitor="val_loss",
            save_best_only=True,
        )
        
        history = model.fit(
            train_ds,
            epochs=model_settings["epochs"],
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, model_checkpoint],
            verbose=settings["verbose"],
        )

        # Log Metrics after Training
        log_metrics(model, test_ds, "trained")

        # Fine-tuning Phase
        base_model = model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-model_settings["ftun_last_layers"]]:
            layer.trainable = False

        logger.info(f"Unfreezing the last {model_settings['ftun_last_layers']} layers")
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=model_settings["ftun_learning_rate"]),
            loss={
                'family': 'categorical_focal_crossentropy',
                'genus': 'categorical_focal_crossentropy',
                'species': 'categorical_focal_crossentropy',
            },
            metrics={
                "family": model_settings["metrics"],
                "genus": model_settings["metrics"],
                "species": model_settings["metrics"],
            },
            loss_weights=model_settings["loss_weights"],
        )

        total_epochs = model_settings["epochs"] + model_settings["ftun_epochs"]
        
        history_fine = model.fit(
            train_ds,
            epochs=total_epochs,
            initial_epoch=len(history.epoch),
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, model_checkpoint],
            verbose=settings["verbose"],
        )

        # Log Metrics after Fine-tuning
        log_metrics(model, test_ds, "fine-tuned")

        # Model Saving
        today = datetime.now().strftime("%y%m%d%H%M")
        model_path_name = (
            f"/Users/leonardo/Documents/Projects/cryptovision/models/hacpl_{model_settings['base_model_short']}_{settings['img_size'][0]}_{run_sufix}_{today}.keras"
        )
        model.save(model_path_name)

        wandb.log_artifact(
            model_path_name,
            name=f"hacpl_{model_settings['base_model_short']}_{settings['img_size'][0]}_{run_sufix}_{today}",
            type="model",
        )

        logger.success(f"Model {model_settings['base_model']} trained and logged to wandb.")

if __name__ == "__main__":
    app()
