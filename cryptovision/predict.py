import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


class CryptoVisionAI:
    def __init__(
        self, model_path, family_names, genus_names, species_names, safe_mode=True
    ):
        """
        Wrap a trained model (or a path to a saved one) for prediction and saliency maps.
        """
        if type(model_path) is str:
            self.model = tf.keras.models.load_model(model_path, safe_mode=safe_mode)
        else:
            self.model = model_path

        self.family_names = family_names
        self.genus_names = genus_names
        self.species_names = species_names
        self.input_size = self.model.input_shape[1:3]
        self._image = None
        self._image_array = None

    @property
    def family_model(self):
        return tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("family").output
        )

    @property
    def genus_model(self):
        return tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("genus").output
        )

    @property
    def species_model(self):
        return tf.keras.Model(
            inputs=self.model.input, outputs=self.model.get_layer("species").output
        )

    @property
    def image(self):
        return self._image

    @property
    def image_array(self):
        return self._image_array

    @property
    def confidence(self):
        try:
            family_pred = (
                self.preds[0].max() if self.preds[0] is not None else "Unknown"
            )
            genus_pred = self.preds[1].max() if self.preds[1] is not None else "Unknown"
            species_pred = (
                self.preds[2].max() if self.preds[2] is not None else "Unknown"
            )
            return (family_pred, genus_pred, species_pred)
        except IndexError:
            return ("Unknown", "Unknown", "Unknown")

    def load_image(self, image_path):
        """
        Load and preprocess an image for prediction.
        """
        self._image = tf.keras.utils.load_img(image_path, target_size=self.input_size)
        img_array = tf.keras.utils.img_to_array(self._image)
        self._image_array = np.expand_dims(img_array, axis=0)
        return self._image_array

    def decoder(self, preds):
        try:
            family_pred = (
                self.family_names[np.argmax(preds[0])]
                if preds[0] is not None
                else "Unknown"
            )
            genus_pred = (
                self.genus_names[np.argmax(preds[1])]
                if preds[1] is not None
                else "Unknown"
            )
            species_pred = (
                self.species_names[np.argmax(preds[2])]
                if preds[2] is not None
                else "Unknown"
            )
            return (family_pred, genus_pred, species_pred)
        except IndexError:
            return ("Unknown", "Unknown", "Unknown")

    def predict(self, input_data, return_raw=False, top_k=1):
        """
        Predict family, genus, and species for a given input.

        Args:
            input_data (str, PIL.Image, np.ndarray or tf.data.Dataset): image path, image, array or dataset.
            return_raw (bool): If True, return raw predictions.
            top_k (int): Number of top predictions to return for each level.
        Returns:
            dict or list: Decoded predictions or raw predictions for the input data.
        """

        # Predict from Input Path
        if isinstance(input_data, str):
            # Validate path
            if os.path.exists(input_data):
                img = self.load_image(input_data)
            else:
                raise FileNotFoundError(
                    f"The provided path does not exist: {input_data}"
                )
            self.preds = self.model.predict(img, verbose=0)
            return self.preds if return_raw else self.decoder(self.preds)

        # Predict from PIL Image
        elif isinstance(input_data, Image.Image):
            img = tf.keras.utils.img_to_array(input_data)
            img = np.expand_dims(img, axis=0)
            self.preds = self.model.predict(img, verbose=0)
            return self.preds if return_raw else self.decoder(self.preds)

        elif isinstance(input_data, np.ndarray):
            img = input_data
            self.preds = self.model.predict(img, verbose=0)
            return self.preds if return_raw else self.decoder(self.preds)

        elif isinstance(input_data, tf.data.Dataset):
            raw_predictions = []
            predictions = []
            # Iterate over the dataset batches.
            # (Assuming each element in the dataset is a tuple: (img_batch, labels))
            for img_batch, _ in input_data:
                preds = self.model.predict(img_batch, verbose=0)
                # 'preds' is expected to be a list of three arrays (for family, genus, species)
                for sample_preds in zip(*preds):  # Iterate per sample in the batch.
                    raw_predictions.append(sample_preds)
                    # For each output, determine the index with maximum probability
                    family_idx = np.argmax(sample_preds[0])
                    genus_idx = np.argmax(sample_preds[1])
                    species_idx = np.argmax(sample_preds[2])

                    # Extract the confidence (probability) for the predicted class
                    family_conf = sample_preds[0][family_idx]
                    genus_conf = sample_preds[1][genus_idx]
                    species_conf = sample_preds[2][species_idx]

                    # Decode the labels from the predicted indices
                    family_label = self.family_names[family_idx]
                    genus_label = self.genus_names[genus_idx]
                    species_label = self.species_names[species_idx]

                    predictions.append(
                        {
                            "family": {
                                "label": family_label,
                                "confidence": float(family_conf),
                            },
                            "genus": {
                                "label": genus_label,
                                "confidence": float(genus_conf),
                            },
                            "species": {
                                "label": species_label,
                                "confidence": float(species_conf),
                            },
                        }
                    )
            return raw_predictions if return_raw else predictions

        else:
            raise TypeError(
                "Unsupported input type. Supported types: str (path), PIL.Image, np.ndarray, tf.data.Dataset"
            )

    def generate_saliency_map(self, level, smooth_samples=20, smooth_noise=0.2):
        """
        Generate a saliency map for a specific prediction level.

        Args:
            level (str): One of ['family', 'genus', 'species'].
            smooth_samples (int): Number of smoothing samples.
            smooth_noise (float): Noise for smoothing.
        Returns:
            np.ndarray: Saliency map.
        """
        if self.image_array is None:
            raise ValueError("No image loaded. Use predict or load an image first.")

        # Select model outputs based on level
        if level == "family":
            model = self.family_model
        elif level == "genus":
            model = self.genus_model
        elif level == "species":
            model = self.species_model
        else:
            raise ValueError("Level must be one of ['family', 'genus', 'species']")

        # Predict class and get the predicted index
        preds = self.model.predict(self.image_array, verbose=0)
        class_index = np.argmax(preds[["family", "genus", "species"].index(level)])

        # Generate saliency map
        score = CategoricalScore([class_index])
        saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=False)
        saliency_map = saliency(
            score,
            self.image_array,
            smooth_samples=smooth_samples,
            smooth_noise=smooth_noise,
        )
        return saliency_map

    def plot_saliency_overlay(self, saliency_map, figure_size=(15, 8)):
        """
        Plot the saliency map over the original image.

        Args:
            saliency_map (np.ndarray): Saliency map to overlay.
            figure_size (tuple): Size of the matplotlib figure.
        """
        if self.image is None or self.image_array is None:
            raise ValueError("No image loaded. Use predict or load an image first.")

        plt.figure(figsize=figure_size)
        plt.title("Saliency Map Overlay")
        plt.imshow(self.image)
        plt.imshow(saliency_map[0], cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.show()
