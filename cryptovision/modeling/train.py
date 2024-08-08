from pathlib import Path

import tensorflow as tf
import typer
from loguru import logger
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from wandb.integration.keras import WandbMetricsLogger
import wandb
from cryptovision.config import MODEL_PARAMS, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
wandb.require("core")


@app.command()
def main(
    train_dir: Path = PROCESSED_DATA_DIR / "train",
    test_dir: Path = PROCESSED_DATA_DIR / "test",
    model_path: Path = MODELS_DIR / "model.h5",
):
    """Main function to train the model."""
    # Initialize Wandb
    wandb.init(project="CryptoVision 2.0", config=MODEL_PARAMS)

    # Load Train & Test Datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training",
        interpolation="bilinear",
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation",
        interpolation="bilinear",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
    )

    class_names = train_ds.class_names
    logger.success("Train and Test Datasets Loaded")

    # Autotune
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation and Preprocessing
    preprocess_input = MODEL_PARAMS["img_preprocess"]
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomCrop(224, 224),
        ]
    )

    # Create Model
    logger.info("Creating Model...")
    base_model = MODEL_PARAMS["pre_trained_model"](
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=MODEL_PARAMS["image_shape"] + (3,),
    )

    if MODEL_PARAMS.get("unfreeze_layers"):
        for layer in base_model.layers[-MODEL_PARAMS["unfreeze_layers"] :]:
            layer.trainable = True
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=MODEL_PARAMS["image_shape"] + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=True)
    x = tf.keras.layers.Dropout(MODEL_PARAMS["dropout"])(x)

    for i in range(MODEL_PARAMS["dense_layers"]):
        x = tf.keras.layers.Dense(
            MODEL_PARAMS["neurons"][i],
            activation="relu",
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
        )(x)
        if MODEL_PARAMS["batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(MODEL_PARAMS["dropout"])(x)

    output = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    logger.info(model.summary())

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_PARAMS["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"],
    )

    logger.success("Model Created")
    logger.info("Starting model training...")

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=MODEL_PARAMS["lr_patience"],
        min_lr=1e-6,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=MODEL_PARAMS["early_stopping_patience"],
        restore_best_weights=True,
    )
    wandb_logger = WandbMetricsLogger()

    # Train the model
    history = model.fit(
        train_ds,
        epochs=MODEL_PARAMS["epochs"],
        validation_data=valid_ds,
        callbacks=[early_stop, reduce_lr, wandb_logger],
    )

    logger.success("Model Training Completed")

    # Save Model
    logger.info("Saving Model...")
    model.save(model_path)
    logger.success("Model Saved")

    # Finish Wandb
    wandb.finish()


if __name__ == "__main__":
    app()
