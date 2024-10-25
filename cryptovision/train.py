import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pandas import DataFrame
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2

import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.require("core")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def random_color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image


def create_image_dataframe(image_path, split_flag=True, column_name=["label"]):
    label = []
    path = []

    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            if filename.startswith("."):
                continue  # Ignore files starting with a dot
            if os.path.splitext(filename)[1] in (".jpeg", ".png", ".jpg"):
                if dirname.split()[-1] != "GT":
                    label.append(os.path.split(dirname)[1])
                    path.append(os.path.join(dirname, filename))

    df_og = pd.DataFrame(columns=["path", "label"])
    df_og["path"] = path
    df_og["label"] = label
    df_og["label"] = df_og["label"].astype("category")

    if split_flag:
        # Split the 'label' column into 'family', 'genus', and 'species' columns
        df_og[["family", "genus", "species"]] = df_og["label"].str.split(
            "_", expand=True
        )

        # Combine genus and species to form the scientific species name
        df_og["species"] = df_og["genus"] + "_" + df_og["species"]

        return df_og[["path"] + column_name]

    else:
        df_og.rename(columns={"label": column_name[0]}, inplace=True)
        return df_og


def create_transfer_learning_model(
    pre_trained_model,
    input_shape,
    num_layers,
    neurons_per_layer,
    unfreeze_layers,
    use_batchnorm=True,
    use_dropout=True,
    dropout_value=0.5,
    l1_value=0.01,
    l2_value=0.001,
    learning_rate=0.0001,
    num_classes=10,  # Adjust this to the number of your classes
):
    logging.info("Creating transfer learning model...")

    # Load the pre-trained model
    pre_trained = pre_trained_model(
        include_top=False, pooling="avg", input_shape=input_shape + (3,)
    )

    # Unfreeze some layers for fine-tuning
    for layer in pre_trained.layers[-unfreeze_layers:]:
        layer.trainable = True

    # Build the model with L1 and L2 regularization
    inp_model = pre_trained.input
    x = pre_trained.output
    
     # Add a convolutional layer
    #x = Conv2D(filters=1024, kernel_size=(3,3), activation='relu')(x)
    #x = GlobalAveragePooling2D()(x)

    for i in range(num_layers):
        x = Dense(
            neurons_per_layer[i],
            activation="relu",
            kernel_regularizer=l1_l2(l1=l1_value, l2=l2_value),
        )(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        if use_dropout:
            x = Dropout(dropout_value)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp_model, outputs=output)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    logging.info("Model created and compiled.")
    #logging.info(model.summary())
    return model



def train_model(
    model,
    X_gen_train,
    X_gen_valid,
    class_weight,
    epochs=200,
    early_stop_patience=5,
    lr_patience=5,
):
    logging.info("Starting model training...")

    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=lr_patience, min_lr=1e-6
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=early_stop_patience, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_gen_train,
        epochs=epochs,
        validation_data=X_gen_valid,
        callbacks=[
            early_stop,
            reduce_lr,
            WandbMetricsLogger(),
        ],
        class_weight=class_weight,
    )

    logging.info("Model training completed.")
    return history


def plot_prediction_vs_actual(
    predicted_dataset: DataFrame,
    num_images: int = 4,
    save_path: str = None,
    target: str = None,
):
    df_pred_sample = predicted_dataset.sample(num_images).reset_index(drop=True)

    fig, ax = plt.subplots(
        int(num_images / 5), int(num_images / 5), figsize=(32, 20)
    )  # Create subplots

    ax = ax.flatten()

    for i in range(num_images):
        image_pred = df_pred_sample.loc[i, "pred"]
        actual_label = df_pred_sample.loc[i, target]

        ax[i].imshow(plt.imread(df_pred_sample.loc[i, "path"]))

        if actual_label == image_pred:
            color = "green"
        else:
            color = "red"

        ax[i].set_title(
            f"pred: {df_pred_sample['pred'][i]}\n{target}: {df_pred_sample[target][i]}",
            color=color,
        )

        # Turn off x-axis and y-axis values
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    # Remove any empty subplots
    for j in range(num_images, len(ax)):
        fig.delaxes(ax[j])

    if save_path:
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Prediction vs Actual plot saved at {save_path}")
    else:
        plt.show()


def check_images(df, image_col="path"):
    invalid_images = []
    for image_path in df[image_col]:
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that it is, in fact, an image
        except (UnidentifiedImageError, IOError, SyntaxError) as _:
            logging.warning(f"Invalid image: {image_path}")
            invalid_images.append(image_path)
    return invalid_images


def main(**params):
    wandb.init(project="CryptoVision-Species-DL", config=params, sync_tensorboard=True)
    logging.info("Starting main process...")

    # Set Image from directory dataframe
    df = pd.concat(
        [
            create_image_dataframe(params["image_dir"], True, [params["model_target"]]),
            create_image_dataframe(
                params["fisbase_dir"], True, [params["model_target"]]
            ),
        ]
    ).reset_index(drop=True)

    logging.info("Image DataFrame created.")

    # Check for invalid images
    invalid_images = check_images(df)
    if invalid_images:
        logging.error(
            f"Found {len(invalid_images)} invalid images. Please check the logs for more details."
        )
        return

    # Set Train, Validation & Test dataframes
    label_count = df[params["model_target"]].value_counts()
    valid_labels = label_count[label_count >= params["min_images_per_class"]].index
    filtered_df = df[df[params["model_target"]].isin(valid_labels)]

    # Split the filtered DataFrame into training and testing sets
    X_train, X_test = train_test_split(
        filtered_df,
        test_size=params["split_test_size"],
        stratify=filtered_df[params["model_target"]],
        random_state=params["random_state"],
    )

    # X_train = create_image_dataframe(params["train_set"], True, [params["model_target"]])
    # X_test = create_image_dataframe(params["test_set"], True, [params["model_target"]])

    logging.info(f"X_train shape: {len(X_train)}")
    logging.info(f"X_test shape: {len(X_test)}")
    # X_train.to_csv("train_data.csv")
    # X_test.to_csv("test_data.csv")
    logging.info("Train and Test dataframes created.")

    # Set image generator
    img_generator = ImageDataGenerator(**params["img_gen_params"])

    X_gen_train = img_generator.flow_from_dataframe(
        dataframe=X_train,
        x_col="path",
        y_col=params["model_target"],
        target_size=params["image_shape"],
        color_mode="rgb",
        batch_size=params["batch_size"],
        class_mode="categorical",
        shuffle=True,
        subset="training",
    )

    X_gen_valid = img_generator.flow_from_dataframe(
        dataframe=X_train,
        x_col="path",
        y_col=params["model_target"],
        target_size=params["image_shape"],
        color_mode="rgb",
        batch_size=params["batch_size"],
        class_mode="categorical",
        shuffle=True,
        subset="validation",
    )

    X_gen_test = img_generator.flow_from_dataframe(
        dataframe=X_test,
        x_col="path",
        y_col=params["model_target"],
        target_size=params["image_shape"],
        color_mode="rgb",
        batch_size=params["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )

    logging.info("Image generators created.")

    # Class weight setup
    X_train_cp = X_train.copy()
    X_train_cp["class_num"] = X_train_cp[params["model_target"]].map(
        X_gen_train.class_indices
    )

    # Compute class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(0, len(X_gen_train.class_indices), 1),
        y=X_train_cp["class_num"].tolist(),
    )

    # Convert class weights to dictionary
    class_weight = dict(enumerate(class_weights))

    logging.info("Class weights computed.")

    # Deep Learning Model Create
    model = create_transfer_learning_model(
        pre_trained_model=params["pretrain_model"],
        input_shape=params["image_shape"],
        unfreeze_layers=params["unfreeze_layers"],
        num_layers=params["num_layers"],
        neurons_per_layer=params["neurons_per_layer"],
        use_batchnorm=params["batchnorm_flag"],
        use_dropout=params["dropout_flag"],
        dropout_value=params["dropout_rate"],
        l1_value=params["l1"],
        l2_value=params["l2"],
        learning_rate=params["learning_rate"],
        num_classes=len(X_gen_train.class_indices),
    )

    # Train the model
    results = train_model(
        model=model,
        X_gen_train=X_gen_train,
        X_gen_valid=X_gen_valid,
        class_weight=class_weight,
        epochs=params["train_epochs"],
        early_stop_patience=params["early_stop_patience"],
        lr_patience=params["lr_patience"],
    )

    logging.info("Model training finished.")

    # Model evaluation
    pred = model.predict(X_gen_test)
    pred = np.argmax(pred, axis=1)

    pred_df = X_test.copy()
    labels = {v: k for k, v in X_gen_test.class_indices.items()}
    pred_df["pred"] = pred
    pred_df["pred"] = pred_df["pred"].apply(lambda x: labels[x])

    accuracy = accuracy_score(pred_df[params["model_target"]], pred_df["pred"])
    logging.info(f"Accuracy Score: {accuracy}")

    # Save training history plot
    results_df = pd.DataFrame(results.history)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
    ax = ax.flatten()

    # Plotting accuracy
    ax[0].plot(results_df["accuracy"], label="Training Accuracy")
    ax[0].plot(results_df["val_accuracy"], label="Validation Accuracy")
    ax[0].set_title("Accuracy")
    ax[0].legend()

    # Plotting loss
    ax[1].plot(results_df["loss"], label="Training Loss")
    ax[1].plot(results_df["val_loss"], label="Validation Loss")
    ax[1].set_title("Loss")
    ax[1].legend()

    # Save the training history plot
    model_dir = f"BV_{params['model_target']}_{params['model_name']}_S{int(accuracy_score(pred_df[params['model_target']], pred_df['pred']) * 1000)}_{datetime.now().strftime('%Y%m%d%H%M')}"
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, "training_history.png"))
    plt.close()

    logging.info("Training history plot saved.")

    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        confusion_matrix(pred_df[params["model_target"]], pred_df["pred"]),
        annot=True,
        fmt="d",
    )
    plt.xticks(
        ticks=np.arange(len(labels)), labels=labels.values(), rotation=45, ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(labels)), labels=labels.values(), rotation=0, va="center"
    )
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    logging.info("Confusion matrix plot saved.")

    class_report = classification_report(
        pred_df[params["model_target"]], pred_df["pred"]
    )
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(class_report)

    logging.info("Classification report saved.")

    plot_prediction_vs_actual(
        pred_df, 25, os.path.join(model_dir, "pred_vs_actual"), params["model_target"]
    )

    # Save the model and class indices
    model.save(os.path.join(model_dir, "model.h5"))
    model.save(os.path.join(model_dir, "model.keras"))

    # Save class indices to a JSON file
    class_indices_inv = {v: k for k, v in X_gen_train.class_indices.items()}
    with open(os.path.join(model_dir, "class_indices.json"), "w") as f:
        json.dump(class_indices_inv, f)

    logging.info(f"Model and class indices saved in {model_dir}")


if __name__ == "__main__":
    PRETRAIN_MODELS = {
        "MOBV2": {
            "model": tf.keras.applications.mobilenet_v2.MobileNetV2,
            "img_preprocessing": tf.keras.applications.mobilenet_v2.preprocess_input,
        },
        "RES50V2": {
            "model": tf.keras.applications.resnet_v2.ResNet50V2,
            "img_preprocessing": tf.keras.applications.resnet_v2.preprocess_input,
        },
        "EFFV2B0": {
            "model": tf.keras.applications.efficientnet_v2.EfficientNetV2B0,
            "img_preprocessing": tf.keras.applications.efficientnet_v2.preprocess_input,
        },
        "EFFV2B2": {
            "model": tf.keras.applications.efficientnet_v2.EfficientNetV2B2,
            "img_preprocessing": tf.keras.applications.efficientnet_v2.preprocess_input,
        },
        "EFFV2S": {
            "model": tf.keras.applications.efficientnet_v2.EfficientNetV2S,
            "img_preprocessing": tf.keras.applications.efficientnet_v2.preprocess_input,
        },
        "EFFV2M": {
            "model": tf.keras.applications.efficientnet_v2.EfficientNetV2M,
            "img_preprocessing": tf.keras.applications.efficientnet_v2.preprocess_input,
        },
    }

    SELECTED_MODEL = "EFFV2M"

    def custom_preprocessing(image):
        image = PRETRAIN_MODELS[SELECTED_MODEL]["img_preprocessing"](image)
        image = random_color_jitter(image)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image

    IMG_GEN_PARAMS = dict(
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.35,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
        preprocessing_function=custom_preprocessing,
        validation_split=0.20,
    )

    params = {
        "image_dir": "/Users/leonardo/Library/CloudStorage/GoogleDrive-leonardofonseca.r@gmail.com/My Drive/04_projects/CryptoVision/Data/sjb/species",
        "fisbase_dir": "/Users/leonardo/Library/CloudStorage/GoogleDrive-leonardofonseca.r@gmail.com/My Drive/04_projects/CryptoVision/Data/web_scrapping/species/train",
        "model_target": "label",
        "min_images_per_class": 50,
        "split_test_size": 0.2,
        "random_state": 42,
        "image_shape": (224, 224),
        "batch_size": 32,
        "img_gen_params": IMG_GEN_PARAMS,
        "pretrain_model": PRETRAIN_MODELS[SELECTED_MODEL]["model"],
        "unfreeze_layers": 10,
        "num_layers": 1,
        "neurons_per_layer": [512],
        "batchnorm_flag": True,
        "dropout_flag": True,
        "dropout_rate": 0.5,
        "l1": 0.01,
        "l2": 0.001,
        "learning_rate": 0.0001,
        "train_epochs": 200,
        "lr_patience": 3,
        "early_stop_patience": 5,
        "model_name": SELECTED_MODEL,
    }

    main(**params)
