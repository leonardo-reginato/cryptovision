from pathlib import Path
import itertools
import json
import tensorflow as tf
import typer
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.layers import Layer

import wandb
from cryptovision.config import MODEL_PARAMS, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()
wandb.require("core")

pre_trained_models = {
    #"MOBV2": {
    #    "model": tf.keras.applications.MobileNetV2,
    #    "prep": tf.keras.applications.mobilenet_v2.preprocess_input,
    #},
    #"RES50V2": {
    #    "model": tf.keras.applications.ResNet50V2,
    #    "prep": tf.keras.applications.resnet_v2.preprocess_input,
    #},
    #"XCEP": {
    #    "model": tf.keras.applications.Xception,
    #    "prep": tf.keras.applications.xception.preprocess_input,
    #},
    #"INCEPV3": {
    #    "model": tf.keras.applications.InceptionV3,
    #    "prep": tf.keras.applications.inception_v3.preprocess_input,
    #},
    #"EFFV2B0": {
    #    "model": tf.keras.applications.EfficientNetV2B0,
    #    "prep": tf.keras.applications.efficientnet_v2.preprocess_input,
    #},
    "ViT": {
        "model": tf.keras.applications.VisionTransformer,
        "prep": tf.keras.applications.vit.preprocess_input,
    },
}


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(input_shape[-1], activation="sigmoid")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Global Average Pooling
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        # Dense layer to learn attention weights
        attention = self.dense(avg_pool)
        # Reshape attention weights to match the input dimensions
        attention = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(attention)
        # Multiply the attention weights with the original inputs
        output = tf.keras.layers.Multiply()([inputs, attention])
        return output

@app.command()
def main(
    train_dir: Path = PROCESSED_DATA_DIR / "train",
    valid_dir: Path = PROCESSED_DATA_DIR / "valid",
    test_dir: Path = PROCESSED_DATA_DIR / "test",
):
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"],
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"],
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"],
    )

    class_names = train_ds.class_names
    
    # Autotune
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Data Augmentation Function
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomCrop(224, 224),
            tf.keras.layers.GaussianNoise(0.1),
        ]
    )
    
    for model_name in pre_trained_models.keys():
        
        MODEL_PARAMS['base_model'] = model_name
        
        wandb.init(
            project="PreTrained Models Comparison",
            config = MODEL_PARAMS
        )
        
        config = wandb.config
        
        pre_trained_model = pre_trained_models[model_name]["model"]
        preprocess_input = pre_trained_models[model_name]["prep"]

        # Model Architecture
        base_model = pre_trained_model(
            include_top=False,
            weights="imagenet",
            input_shape=config.image_shape + [3,],
        )
        
        base_model.trainable = False

        inputs = tf.keras.Input(shape=config.image_shape + [3,])
        x = data_augmentation(inputs)  # Data Augmentation Layer
        x = preprocess_input(x)  # PreTrain model image preprocess
        x = base_model(x, training=False)  # Add base model
        if config.attention_layer:
            # Apply the custom attention layer here
            x = AttentionLayer()(x)
            
        #x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        x = tf.keras.layers.Dropout(config.dropout)(x)  # Add drop out layer

        x = tf.keras.layers.Dense(
            config.neurons,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=config.l1, l2=config.l2),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(config.dropout)(x)

        output = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate
            ),
            loss=config.loss,
            metrics=config.metrics
        )
        
        # Define callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config.lr_patience,
            min_lr=1e-6,
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
        )
        
        # Train the model
        history = model.fit(
            train_ds,
            epochs=MODEL_PARAMS["epochs"],
            validation_data=valid_ds,
            callbacks=[
                early_stop,
                reduce_lr,
                WandbMetricsLogger(log_freq=5),
                #WandbModelCheckpoint(MODELS_DIR / f"{config.base}"),
            ],
        )
        
        wandb.finish()
        
        
if __name__ == "__main__":
    app()