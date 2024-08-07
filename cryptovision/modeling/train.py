from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tqdm import tqdm

import wandb
from wandb.integration.keras import WandbMetricsLogger
from cryptovision.config import (
    IMG_GEN_PARAMS,
    MODEL_PARAMS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()
wandb.require("core")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    train_dataset_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_dataset_path: Path = PROCESSED_DATA_DIR / "test.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # Initialize Wandb
    logger.info("Starting Modeling Training...")
    wandb.init(project="CryptoVision-Species-DL", config=MODEL_PARAMS)

    # Load Train & Test Datasets
    train_df = pd.read_csv(train_dataset_path)
    test_df = pd.read_csv(test_dataset_path)

    logger.success("Train and Test Datasets Loaded")
    logger.info(f"Train Dataset Shape: {train_df.shape}")
    logger.info(f"Test Dataset Shape: {test_df.shape}")

    # Set tensorflow Image Data Generator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(**IMG_GEN_PARAMS)

    train_datagen = image_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col=MODEL_PARAMS["target"],
        target_size=MODEL_PARAMS["image_shape"],
        color_mode="rgb",
        batch_size=MODEL_PARAMS["batch_size"],
        class_mode="categorical",
        shuffle=True,
        subset="training",
    )

    valid_datagen = image_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col=MODEL_PARAMS["target"],
        target_size=MODEL_PARAMS["image_shape"],
        color_mode="rgb",
        batch_size=MODEL_PARAMS["batch_size"],
        class_mode="categorical",
        shuffle=True,
        subset="validation",
    )

    test_datagen = image_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col="path",
        y_col=MODEL_PARAMS["target"],
        target_size=MODEL_PARAMS["image_shape"],
        color_mode="rgb",
        batch_size=MODEL_PARAMS["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )
    
    logger.success("Image Data Generators Initialized")

    # Class Weights Calculation
    train_df_copy = train_df.copy()
    train_df_copy["class_indices"] = train_df_copy[MODEL_PARAMS["target"]].map(
        train_datagen.class_indices
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(0, len(train_datagen.class_indices), 1),
        y=train_df_copy["class_indices"].tolist(),
    )

    class_weights = dict(enumerate(class_weights))

    logger.success("Class Weights Calculated")

    # Create Model
    logger.info("Creating Model...")

    pre_trained_model = MODEL_PARAMS["pre_trained_model"](
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=MODEL_PARAMS["image_shape"] + (3,),
    )

    for layer in pre_trained_model.layers[-MODEL_PARAMS["unfreeze_layers"] :]:
        layer.trainable = True

    input_model = pre_trained_model.input
    x = pre_trained_model.output

    for i in range(MODEL_PARAMS["num_layer"]):
        x = tf.keras.layers.Dense(
            MODEL_PARAMS["neurons"][i],
            activation="relu",
            kernel_regularizer=l1_l2(l1=MODEL_PARAMS["l1"], l2=MODEL_PARAMS["l2"]),
        )(x)
        if MODEL_PARAMS["batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(MODEL_PARAMS["dropout"])(x)

    output = tf.keras.layers.Dense(
        len(train_datagen.class_indices), activation="softmax"
    )(x)
    model = tf.keras.models.Model(inputs=input_model, outputs=output)

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=MODEL_PARAMS["learning_rate"])

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=['accuracy','AUC', 'Precision', 'Recall'],
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

    # Train the model
    history = model.fit(
        train_datagen,
        epochs=MODEL_PARAMS["epochs"],
        validation_data=valid_datagen,
        callbacks=[
            early_stop,
            reduce_lr,
            WandbMetricsLogger(),
        ],
        class_weight=class_weights,
    )

    logger.success("Model Training Completed")

    # Save Model
    logger.info("Saving Model...")
    model.save(model_path)
    logger.success("Model Saved")


if __name__ == "__main__":
    app()
