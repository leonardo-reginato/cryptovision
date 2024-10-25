from pathlib import Path
import tensorflow as tf
import typer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from wandb.integration.keras import WandbMetricsLogger
import wandb
from cryptovision.config import MODEL_PARAMS, PROCESSED_DATA_DIR
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the Typer app
app = typer.Typer()
wandb.require("core")

# Pre-trained models and preprocessors
pre_trained_models = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
    "Xception": tf.keras.applications.Xception,
    "InceptionV3": tf.keras.applications.InceptionV3,
}

preprocess_inputs = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
    "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
    "Xception": tf.keras.applications.xception.preprocess_input,
    "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
}


# Function to load datasets with caching and prefetching
def load_datasets(train_dir, valid_dir, test_dir):
    # Load training dataset and collect class_names
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"], image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"]
    )
    class_names = train_ds.class_names  # Collect class names before caching, etc.
    
    # Apply caching, shuffling, and prefetching after collecting class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    # Load validation and test datasets
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir, labels="inferred", label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"], image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"]
    ).cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"], image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"]
    ).cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, valid_ds, test_ds, class_names  # Return class_names



# Function to build the model
def build_model(pre_trained_model, preprocess_input, input_shape, num_classes):
    base_model = pre_trained_model(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation()(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(
        512, activation="relu", kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Function to define data augmentation
def data_augmentation():
    return tf.keras.Sequential(
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


# Function to log metrics and save reports
def log_results(history, model, valid_ds, class_names):
    # Example of logging classification report and confusion matrix
    y_true = tf.concat([y for x, y in valid_ds], axis=0)
    y_pred = model.predict(valid_ds)
    y_pred_labels = tf.argmax(y_pred, axis=1)

    report = classification_report(
        tf.argmax(y_true, axis=1), y_pred_labels, target_names=class_names
    )
    cm = confusion_matrix(tf.argmax(y_true, axis=1), y_pred_labels)

    # Log classification report and confusion matrix
    logger.info("Classification Report:\n" + report)
    wandb.log({"classification_report": report})

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


# Main function to run the training
@app.command()
def main(
    train_dir: Path = PROCESSED_DATA_DIR / "train",
    valid_dir: Path = PROCESSED_DATA_DIR / "valid",
    test_dir: Path = PROCESSED_DATA_DIR / "test",
):
    train_ds, valid_ds, test_ds, class_names = load_datasets(train_dir, valid_dir, test_dir)

    for model_name, pre_trained_model in pre_trained_models.items():
        wandb.init(
            project="PreTrained Models Comparison V2",
            config={
                "pre_trained_model": model_name,
                "dense_layers": 1,
                "neurons": 256,
                "dropout": 0.2,
                "batch_norm": True,
            },
        )

        preprocess_input = preprocess_inputs[model_name]
        model = build_model(
            pre_trained_model=pre_trained_model,
            preprocess_input=preprocess_input,
            input_shape=MODEL_PARAMS["image_shape"] + (3,),
            num_classes=len(class_names),
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=MODEL_PARAMS["learning_rate"]
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy", "AUC", "Precision", "Recall"],
        )

        # Callbacks
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

        # Log results
        log_results(history, model, valid_ds, class_names)

        wandb.finish()


if __name__ == "__main__":
    app()
