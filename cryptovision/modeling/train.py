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
from wandb.integration.keras import WandbMetricsLogger
import wandb
from cryptovision.config import MODEL_PARAMS, MODELS_DIR, PROCESSED_DATA_DIR


app = typer.Typer()
wandb.require("core")

pre_trained_models = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "ResNet152V2": tf.keras.applications.ResNet152V2,
    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
    "EfficientNetV2B2": tf.keras.applications.EfficientNetV2B2,
    "EfficientNetV2S": tf.keras.applications.EfficientNetV2S,
}

preprocess_inputs = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
    "ResNet152V2": tf.keras.applications.resnet_v2.preprocess_input,
    "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2B2": tf.keras.applications.efficientnet_v2.preprocess_input,
    "EfficientNetV2S": tf.keras.applications.efficientnet_v2.preprocess_input,
}

dense_layers_options = [1, 2, 3]
neurons_options = [64, 128, 256, 512, 1024, 2048]
dropout_options = [0.2]
batch_norm_options = [True]
unfreeze_layers_options = [None, 5, 10, 20, 30]

@app.command()
def main(
    train_dir: Path = PROCESSED_DATA_DIR / "train",
    valid_dir: Path = PROCESSED_DATA_DIR / "valid",
    test_dir: Path = PROCESSED_DATA_DIR / "test",
):
    """Main function to train the model."""

    # Load Train & Test Datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        shuffle=True,
        seed=42,
        #validation_split=0.2,
        #subset="training",
        interpolation="bilinear",
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        shuffle=True,
        seed=42,
        #validation_split=0.2,
        #subset="validation",
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

    # Iterate over all combinations of hyperparameters
    for model_name, dense_layers, dropout, batch_norm, unfreeze_layers in itertools.product(
        pre_trained_models.keys(),
        dense_layers_options,
        dropout_options,
        batch_norm_options,
        unfreeze_layers_options,
    ):
        pre_trained_model = pre_trained_models[model_name]
        preprocess_input = preprocess_inputs[model_name]

        neurons_list = list(
            itertools.combinations_with_replacement(neurons_options, dense_layers)
        )
        for neurons in neurons_list:
            wandb.init(
                project="CryptoVision 2.0",
                config={
                    "pre_trained_model": model_name,
                    "dense_layers": dense_layers,
                    "neurons": neurons,
                    "dropout": dropout,
                    "batch_norm": batch_norm,
                },
            )

            # Data Augmentation and Preprocessing
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
            logger.info(
                f"Creating Model with {model_name}, {dense_layers} dense layers, dropout {dropout}, batch_norm {batch_norm}"
            )
            base_model = pre_trained_model(
                include_top=False,
                weights="imagenet",
                pooling="avg",
                input_shape=MODEL_PARAMS["image_shape"] + (3,),
            )

            if unfreeze_layers is not None:
                base_model.trainable = True
                for layer in base_model.layers[:unfreeze_layers]:
                    layer.trainable = False
            else:
                base_model.trainable = False

            inputs = tf.keras.Input(shape=MODEL_PARAMS["image_shape"] + (3,))
            x = data_augmentation(inputs)
            x = preprocess_input(x)
            x = base_model(x, training=True)
            x = tf.keras.layers.Dropout(dropout)(x)

            for neuron_count in neurons:
                x = tf.keras.layers.Dense(
                    neuron_count,
                    activation="relu",
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                )(x)
                if batch_norm:
                    x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(dropout)(x)

            output = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
            model = tf.keras.models.Model(inputs=inputs, outputs=output)

            logger.info(model.summary())

            # Compile Model
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=MODEL_PARAMS["learning_rate"]
            )
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
                # callbacks=[early_stop, reduce_lr, wandb_logger],
                callbacks=[
                    early_stop,
                    reduce_lr,
                    wandb_logger,
                ],
            )

            logger.success("Model Training Completed")

            # Save Model
            model_name = f"{model_name}_layers{dense_layers}_neurons{neurons}_dropout{dropout}_batchnorm{batch_norm}.keras"
            model.save(MODELS_DIR / model_name)
            logger.success(f"Model {model_name} Saved")
            

            # Evaluate Model
            logger.info("Evaluating Model...")
            evaluation_results = model.evaluate(test_ds, verbose=0)
            test_loss = evaluation_results[0]
            test_accuracy = evaluation_results[1]
            logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            # Get predictions and true labels
            y_pred = tf.argmax(model.predict(test_ds, verbose=0), axis=1).numpy()
            y_true = tf.concat([y for x, y in test_ds], axis=0)
            y_true = tf.argmax(y_true, axis=1).numpy()

            # Confusion Matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            conf_matrix_path = MODELS_DIR / f"{model_name}_confusion_matrix.png"
            plt.savefig(conf_matrix_path)
            plt.close()
            logger.info(f"Confusion matrix saved to {conf_matrix_path}")

            # Classification Report
            class_report = classification_report(y_true, y_pred, target_names=class_names)
            class_report_path = MODELS_DIR / f"{model_name}_classification_report.txt"
            with open(class_report_path, "w") as f:
                f.write(class_report)
            logger.info(f"Classification report saved to {class_report_path}")

            # Save Class Names
            class_names_path = MODELS_DIR / f"{model_name}_class_names.json"
            with open(class_names_path, "w") as f:
                json.dump(class_names, f)
            logger.info(f"Class names saved to {class_names_path}")


            # Finish Wandb
            wandb.finish()


if __name__ == "__main__":
    app()
