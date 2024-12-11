import wandb
import typer
import tensorflow as tf
from pathlib import Path
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from cryptovision.config import (
    PROCESSED_DATA_DIR, PROJ_NAME, SETUP, PROTEON, PHORCYS)
from cryptovision.tools import (
    image_directory_to_pandas,
    split_image_dataframe,
    tf_dataset_from_pandas,
)
from cryptovision.ai_architecture import (
    proteon, augmentation_layer, phorcys, phorcys_conv, stable_phorcys_conv)

# Initialize Typer app and set mixed precision for TensorFlow
app = typer.Typer()
wandb.require("core")
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model_name = (
            f"{SETUP['sufix']}_"
            f"{SETUP['arch_type']}_"
            f"{SETUP['base_model_nickname']}_"
            f"{SETUP['version']}"
        )


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
        name = f"{SETUP['sufix']}",
        config={**SETUP,**PHORCYS,},
    ) as run:
        
        # Dataset Setup
        image_df = image_directory_to_pandas(dataset_dir)
        train_df, val_df, test_df = split_image_dataframe(
            image_df,
            test_size=SETUP["test_size"],
            val_size=SETUP["val_size"],
            random_state=SETUP["seed"],
            stratify_by=SETUP["stratify_by"],
        )

        logger.info(
            f"Train: {len(train_df)} ({len(train_df)/len(image_df) * 100:.2f} %), "
            f"Val: {len(val_df)} ({len(val_df)/len(image_df) * 100:.2f} %), "
            f"Test: {len(test_df)} ({len(test_df)/len(image_df) * 100:.2f} %)"
        )

        # Dataset Preparation
        train_ds, family_labels, genus_labels, species_labels = tf_dataset_from_pandas(
            train_df, SETUP["batch_size"], SETUP["img_size"]
        )
        val_ds, _, _, _ = tf_dataset_from_pandas(val_df, SETUP["batch_size"], SETUP["img_size"])
        test_ds, _, _, _ = tf_dataset_from_pandas(test_df, SETUP["batch_size"], SETUP["img_size"])

        # Define Data Augmentation
        data_augmentation = augmentation_layer(
            flip=SETUP['aug']["flip"],
            rotation=SETUP['aug']["rotation"],
            zoom=SETUP['aug']["zoom"],
            translation=(0.1, 0.1),
            contrast=SETUP['aug']["contrast"],
            brightness=SETUP['aug']["brightness"],
        )

        # Model Creation
        model = stable_phorcys_conv(
            input_shape=PHORCYS["input_shape"],
            n_families=len(family_labels),
            n_genera=len(genus_labels),
            n_species=len(species_labels),
            augmentation_layer=data_augmentation,
            shared_layer_neurons=PHORCYS["shared_layer"],
            shared_layer_dropout=PHORCYS["dropout"],
            genus_hidden_neurons=PHORCYS["genus_hidden"],
            species_hidden_neurons=PHORCYS["species_hidden"],
            #attention=PHORCYS["attention"],
        )

        # Model Summary
        model.summary()
        
        # Model Compilation
        model.compile(                                                                                                                                  
            optimizer=tf.keras.optimizers.Adam(learning_rate=SETUP["learning_rate"]),
            loss=SETUP['loss'],
            metrics=SETUP['metrics'],
            loss_weights=SETUP["loss_weights"],
        )

        # Log Metrics before Training
        log_metrics(model, test_ds, "pre-train")

        # Training Phase
        wandb_logger = WandbMetricsLogger()
        
        # Define Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=SETUP["monitor"],
            patience=SETUP["early_stopping"],
            restore_best_weights=SETUP['restore_best_weights'],
        )
        
        # Define Reduce Learning Rate on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=SETUP["monitor"],
            factor=SETUP["lr_factor"],
            patience=SETUP["lr_patience"],
            min_lr=SETUP["lr_min"],
        )
        
        # Save the best model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/{model_name}.keras",  # Path to save the model
            monitor="val_loss",  # Metric to monitor for improvement
            save_best_only=True,  # Save only the best model
            mode="min",  # Minimize the monitored metric
            verbose=0  # Print messages when saving a model
        )
        
        history = model.fit(
            train_ds,
            epochs=SETUP["epochs"],
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint],
            verbose=SETUP["verbose"],
        )

        # Log Metrics after Training
        log_metrics(model, test_ds, "trained")

        # Fine-tuning Phase
        base_model = model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-SETUP["ftun_last_layers"]]:
            layer.trainable = False

        logger.info(f"Unfreezing the last {SETUP['ftun_last_layers']} layers")
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=SETUP["ftun_learning_rate"]),
            loss=SETUP['loss'],
            metrics=SETUP['metrics'],
            loss_weights=SETUP["loss_weights"],
        )

        total_epochs = SETUP["epochs"] + SETUP["ftun_epochs"]
        
        history_fine = model.fit(
            train_ds,
            epochs=total_epochs,
            initial_epoch=len(history.epoch),
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint],
            verbose=SETUP["verbose"],
        )

        # Log Metrics after Fine-tuning
        log_metrics(model, test_ds, "fine-tuned")

        # Model Saving
        model.save(f"models/{model_name}.keras")

        wandb.log_artifact(
            f"models/{model_name}.keras",
            name=model_name,
            type="model",
        )

        logger.success(f"Model {model_name} trained and logged to wandb.")

if __name__ == "__main__":
    app()
