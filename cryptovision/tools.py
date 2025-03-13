import os
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import imagehash
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageStat, ExifTags
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from skimage.segmentation import mark_boundaries
from lime.lime_image import LimeImageExplainer
from tensorflow.keras.callbacks import Callback     # type: ignore

from loguru import logger
from colorama import Fore, Style


class TQDMProgressBar(Callback):
    
    def __init__(self):
        super(TQDMProgressBar, self).__init__()
        self.epoch_bar = None  # Ensure clean initialization
    
    def on_train_begin(self, logs=None):
        # Close any lingering progress bars from previous runs
        if self.epoch_bar:
            self.epoch_bar.close()
        self.epoch_bar = None
    
    def on_epoch_begin(self, epoch, logs=None):
        # Close any existing progress bar to avoid overlaps
        if self.epoch_bar:
            self.epoch_bar.close()
            
        # Create a new progress bar for each epoch
        self.epoch_bar = tqdm(total=self.params['steps'], 
                              desc=f"Epoch {epoch+1}/{self.params['epochs']}", 
                              unit="batch", 
                              dynamic_ncols=True)
    
    def on_batch_end(self, batch, logs=None):
        if self.epoch_bar:
            self.epoch_bar.update(1)
        
        # Reduce updates to avoid clutter
        if batch % 1 == 0 or batch == self.params['steps'] - 1:
            self.epoch_bar.set_postfix({
                'loss': f"{logs.get('loss', 0):.4f}",
                'fam_acc': f"{logs.get('family_accuracy', 0):.4f}",
                'gen_acc': f"{logs.get('genus_accuracy', 0):.4f}",
                'spe_acc': f"{logs.get('species_accuracy', 0):.4f}",
            })
    
    def colorize_accuracy(self, value):
        if value < 0.75:
            return Fore.RED + f"{value:.4f}" + Style.RESET_ALL
        elif 0.75 <= value < 0.85:
            return Fore.YELLOW + f"{value:.4f}" + Style.RESET_ALL
        elif 0.85 <= value < 0.92:
            return Fore.GREEN + f"{value:.4f}" + Style.RESET_ALL
        else:
            return Fore.MAGENTA + f"{value:.4f}" + Style.RESET_ALL
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None  # Reset for the next epoch
        
        val_family_acc = logs.get('val_family_accuracy', 0)
        val_genus_acc = logs.get('val_genus_accuracy', 0)
        val_species_acc = logs.get('val_species_accuracy', 0)

        val_family_acc_colored = self.colorize_accuracy(val_family_acc)
        val_genus_acc_colored = self.colorize_accuracy(val_genus_acc)
        val_species_acc_colored = self.colorize_accuracy(val_species_acc)

        summary_message = (
            f"Epoch {epoch+1} - Val Loss: {logs.get('val_loss', 0):.4f}, "
            f"Val Family Acc: {val_family_acc_colored}, "
            f"Val Genus Acc: {val_genus_acc_colored}, "
            f"Val Species Acc: {val_species_acc_colored}"
        )
        logger.info(summary_message)
    
    def on_train_end(self, logs=None):
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None


def image_dir_pandas(path: Path, source: str=None):
    """
    Loads image paths and their taxonomic labels into a Pandas DataFrame.
    
    Parameters:
    - image_dir (str): Root directory containing image subfolders.
    - source (str, optional): Source identifier to include in the DataFrame.
    
    Returns:
    - pd.DataFrame: DataFrame containing image paths and taxonomic labels.
    """
    paths, labels = [], []
    
    for root, _, files in os.walk(path):
        folder_label = os.path.basename(root)
        if folder_label == "GT" or folder_label.startswith("."):
            continue
        
        for file in files:
            if file.lower().endswith(('.jpeg', '.png', '.jpg')) and not file.startswith('.'):
                paths.append(os.path.join(root, file))
                labels.append(folder_label)
    
    df = pd.DataFrame({'image_path': paths, 'folder_label': labels})
    df['folder_label'] = df['folder_label'].astype("category")
    
    try:
        df[['family', 'genus', 'species']] = df['folder_label'].str.split("_", expand=True)
        df['species'] = df['genus'] + " " + df['species']
    except ValueError:
        raise ValueError("Ensure the folder structure follows 'family_genus_species' format.")
    
    if source:
        df['source'] = source
    
    return df


def split_dataframe(df, test_size=0.2, val_size=0.1, random_state=42, stratify_by='folder_label'):
    """
    Split a pandas DataFrame into train, validation, and test sets,
    stratified by the 'folder_name' column.

    Args:
        df (pd.DataFrame): The DataFrame containing image paths and labels.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        tuple: Three pandas DataFrames for train, validation, and test sets.
    """
    
    # First, split into train+validation and test sets
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_by],
        random_state=random_state
    )
    
    # Calculate the adjusted validation size relative to the remaining train+val data
    val_relative_size = val_size / (1 - test_size)
    
    # Split the train+validation set into train and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df[stratify_by],
        random_state=random_state
    )
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    return train_df, val_df, test_df


def load_image(image_path, image_size: tuple[int, int] = None):
    
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    if image_size:
        img = tf.image.resize(img, image_size)
    
    return img


def standard_directory(input_path, prefix, quality=95):
    """
    Converts images to JPEG format (if necessary) and renames them according to the given pattern.

    Args:
        input_path (str): Path to the main directory containing subfolders.
        prefix (str): A prefix to add at the beginning of each renamed image.
        quality (int, optional): JPEG quality for saving images. Defaults to 95.

    Example:
        If a subfolder is named "Family_Genus_species", the images inside it will be renamed to:
        "{prefix}_Genus_species_0001.jpeg", "{prefix}_Genus_species_0002.jpeg", etc.
    """
    input_dir = Path(input_path)

    # Iterate over subfolders in the input directory
    for subfolder in input_dir.iterdir():
        if not subfolder.is_dir():
            continue

        parts = subfolder.name.split('_')
        if len(parts) != 3:
            print(f"Skipping folder '{subfolder.name}': not in 'Family_Genus_species' format.")
            continue

        _, genus, species = parts

        # Get and sort all image files with supported extensions
        supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted(
            [img for img in subfolder.iterdir() if img.is_file() and img.suffix.lower() in supported_exts]
        )

        for idx, image_file in enumerate(image_files, start=1):
            # Format the counter with leading zeros (e.g., 0001, 0002, ...)
            new_image_name = f"{prefix}_{genus}_{species}_{idx:04}.jpeg"
            new_image_path = subfolder / new_image_name

            try:
                with Image.open(image_file) as img:
                    # Convert image to RGB if needed (e.g., for PNG images with transparency)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(new_image_path, "JPEG", quality=quality)

                # Remove the original file if it differs from the new path
                if image_file != new_image_path:
                    image_file.unlink()

                print(f"Converted and renamed: {image_file} -> {new_image_path}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    print("Renaming and conversion completed.")
    return True


def process_image_and_labels(image_path, family, genus, species, family_labels, genus_labels, species_labels, image_size=(224,224)):
    """
    Process an image and its corresponding labels for training.

    Parameters:
    ----------
    image_path : str
        The path to the image file.
    family : str
        The family label of the image.
    genus : str
        The genus label of the image.
    species : str
        The species label of the image.
    family_labels : tf.Tensor
        Tensor of unique family labels.
    genus_labels : tf.Tensor
        Tensor of unique genus labels.
    species_labels : tf.Tensor
        Tensor of unique species labels.

    Returns:
    -------
    img : tf.Tensor
        The processed image tensor.
    labels : dict
        A dictionary containing one-hot encoded labels for family, genus, and species.
    """
    # Load the raw data from the file as a string
    img = load_image(image_path, image_size)

    # Convert family, genus, and species to indices
    family_label = tf.argmax(tf.equal(family_labels, family))
    genus_label = tf.argmax(tf.equal(genus_labels, genus))
    species_label = tf.argmax(tf.equal(species_labels, species))

    # Convert to one-hot encoded format
    family_label = tf.one_hot(family_label, len(family_labels))
    genus_label = tf.one_hot(genus_label, len(genus_labels))
    species_label = tf.one_hot(species_label, len(species_labels))

    # Return the image and a dictionary of labels with matching keys
    return img, {
        "family": family_label,
        "genus": genus_label,
        "species": species_label
    }


def tensorflow_dataset(df, batch_size: int=32, image_size: tuple[int, int]=(224, 224), shuffle: bool=True) -> tf.data.Dataset:
    """
    Build a TensorFlow dataset from a DataFrame containing image paths and taxonomic labels.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing the following columns:
        - 'path': The path to the image.
        - 'Family': The family label of the image.
        - 'Genus': The genus label of the image.
        - 'Species': The species label of the image.
    batch_size : int, optional
        Batch size for training. Default is 32.

    Returns:
    -------
    image_label_ds : tf.data.Dataset
        A TensorFlow dataset with images and one-hot encoded labels.
    family_labels : list
        A sorted list of unique family labels.
    genus_labels : list
        A sorted list of unique genus labels.
    species_labels : list
        A sorted list of unique species labels.
    """
    
    # Extract the unique family, genus, and species from the dataframe
    family_labels = sorted(df['family'].unique())
    genus_labels = sorted(df['genus'].unique())
    species_labels = sorted(df['species'].unique())

    # Convert family, genus, and species labels to TensorFlow tensors
    family_labels = tf.constant(family_labels)
    genus_labels = tf.constant(genus_labels)
    species_labels = tf.constant(species_labels)

    # Create a TensorFlow dataset from the dataframe's paths and labels
    path_ds = tf.data.Dataset.from_tensor_slices(
        (df['image_path'].values, df['family'].values, df['genus'].values, df['species'].values)
    )

    # Map the processing function to the dataset
    image_dataset = path_ds.map(
        lambda path, family, genus, species: process_image_and_labels(
            path, family, genus, species, family_labels, genus_labels, species_labels, image_size=image_size
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Shuffle, batch, and prefetch the dataset
    image_dataset = image_dataset\
        .batch(batch_size)\
        .cache()\
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        
    if shuffle:
        image_dataset = image_dataset.shuffle(buffer_size=len(df))

    return image_dataset


def predict_image(image_path, model, family_labels, genus_labels, species_labels, image_size=(224,224) ,top_k=3):
    """
    Predict the top-k family, genus, and species from an image using a trained model,
    and display the image with predictions.

    Args:
    - image_path (str): Path to the image file.
    - model (tf.keras.Model): The trained model.
    - family_labels (list): List of family labels.
    - genus_labels (list): List of genus labels.
    - species_labels (list): List of species labels.
    - top_k (int): Number of top predictions to return.

    Returns:
    - top_k_family: List of tuples (family, confidence) for top k family predictions.
    - top_k_genus: List of tuples (genus, confidence) for top k genus predictions.
    - top_k_species: List of tuples (species, confidence) for top k species predictions.
    """

    # Load and preprocess the image
    img = load_image(image_path, image_size)
    img = tf.expand_dims(img, 0)  # Add batch dimension

    # Predict family, genus, and species
    family_pred, genus_preds, species_preds = model.predict(img)
    
    # Get top-k predictions for family
    top_k_family_indices = np.argsort(family_pred[0])[-top_k:][::-1]
    top_k_family = [(family_labels[i], family_pred[0][i]) for i in top_k_family_indices]

    # Get top-k predictions for genus
    top_k_genus_indices = np.argsort(genus_preds[0])[-top_k:][::-1]
    top_k_genus = [(genus_labels[i], genus_preds[0][i]) for i in top_k_genus_indices]

    # Get top-k predictions for species
    top_k_species_indices = np.argsort(species_preds[0])[-top_k:][::-1]
    top_k_species = [(species_labels[i], species_preds[0][i]) for i in top_k_species_indices]

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(tf.image.resize(img[0], image_size) / 255.0)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    # Print top-k predictions for each level
    print("Top 3 Family Predictions:")
    for family, confidence in top_k_family:
        print(f"{family}: {confidence:.4f}")

    print("\nTop 3 Genus Predictions:")
    for genus, confidence in top_k_genus:
        print(f"{genus}: {confidence:.4f}")

    print("\nTop 3 Species Predictions:")
    for species, confidence in top_k_species:
        print(f"{species}: {confidence:.4f}")

    return top_k_family, top_k_genus, top_k_species


def plot_training_history(history, history_fine, fine_tune_at):
    """
    Plot the training history of accuracy and loss for each output.
    
    Args:
    - history (History): History object from the initial training.
    - history_fine (History): History object from the fine-tuning phase.
    - fine_tune_at (int): Epoch at which fine-tuning began.
    """
    # Combine initial training history and fine-tuning history
    accuracy_keys = ['family_accuracy', 'genus_accuracy', 'species_accuracy']
    val_accuracy_keys = ['val_family_accuracy', 'val_genus_accuracy', 'val_species_accuracy']
    loss_keys = ['family_loss', 'genus_loss', 'species_loss']
    val_loss_keys = ['val_family_loss', 'val_genus_loss', 'val_species_loss']

    # Combine the data from the initial training and fine-tuning phases
    combined_history = {}
    for key in accuracy_keys + val_accuracy_keys + loss_keys + val_loss_keys:
        combined_history[key] = history.history.get(key, []) + history_fine.history.get(key, [])

    total_epochs = len(combined_history[accuracy_keys[0]])  # Total number of epochs including fine-tuning
    
    # Create subplots for accuracy and loss
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot accuracy for each output
    for idx, key in enumerate(accuracy_keys):
        axs[0, idx].plot(combined_history[key], label='Training Accuracy')
        axs[0, idx].plot(combined_history[val_accuracy_keys[idx]], label='Validation Accuracy')
        axs[0, idx].axvline(x=fine_tune_at, color='r', linestyle='--', label='Fine-Tuning Start')
        axs[0, idx].set_title(f'{key.replace("_accuracy", "").capitalize()} Accuracy')
        axs[0, idx].set_xlabel('Epochs')
        axs[0, idx].set_ylabel('Accuracy')
        axs[0, idx].legend()
        axs[0, idx].grid(True)
    
    # Plot loss for each output
    for idx, key in enumerate(loss_keys):
        axs[1, idx].plot(combined_history[key], label='Training Loss')
        axs[1, idx].plot(combined_history[val_loss_keys[idx]], label='Validation Loss')
        axs[1, idx].axvline(x=fine_tune_at, color='r', linestyle='--', label='Fine-Tuning Start')
        axs[1, idx].set_title(f'{key.replace("_loss", "").capitalize()} Loss')
        axs[1, idx].set_xlabel('Epochs')
        axs[1, idx].set_ylabel('Loss')
        axs[1, idx].legend()
        axs[1, idx].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    

def get_taxonomic_mappings_from_folders(data_dir):
    """
    Extract family, genus, and species mappings from the dataset folder structure.

    Args:
    - data_dir (str): Path to the training dataset directory.

    Returns:
    - family_labels (list): List of unique family names.
    - genus_labels (list): List of unique genus names.
    - species_labels (list): List of unique species names.
    - genus_to_family (dict): Mapping of genus to family.
    - species_to_genus (dict): Mapping of species to genus.
    """
    family_labels = set()
    genus_labels = set()
    species_labels = set()
    genus_to_family = {}
    species_to_genus = {}

    for folder_name in os.listdir(data_dir):
        parts = folder_name.split('_')
        if len(parts) == 3:
            family, genus, species = parts
            family_labels.add(family)
            genus_labels.add(genus)
            species_full = f"{genus}_{species}"
            species_labels.add(species_full)
            genus_to_family[genus] = family
            species_to_genus[species_full] = genus

    return sorted(list(family_labels)), sorted(list(genus_labels)), sorted(list(species_labels)), genus_to_family, species_to_genus

def get_taxonomic_mappings_from_dataframe(df):
    """
    Extract taxonomic mappings from a DataFrame that contains per-sample image labels.

    Args:
        df (pd.DataFrame): DataFrame with columns ['species', 'genus', 'family', 'folder_label']

    Returns:
        family_labels (list): Sorted unique list of family names
        genus_labels (list): Sorted unique list of genus names
        species_labels (list): Sorted unique list of species names (genus + species)
        genus_to_family_map (dict): Mapping from genus index to family index
        species_to_genus_map (dict): Mapping from species index to genus index
    """

    # Deduplicate based on folder_label (one per class is enough for structure)
    df_unique = df[['species', 'genus', 'family']].drop_duplicates()

    # Sorted label vocabularies
    family_labels = sorted(df_unique['family'].unique())
    genus_labels = sorted(df_unique['genus'].unique())
    species_labels = sorted(df_unique['species'].unique())  # already "Genus species" string

    # Map from genus index → family index
    genus_to_family_map = {
        genus_labels.index(row['genus']): family_labels.index(row['family'])
        for _, row in df_unique[['genus', 'family']].drop_duplicates().iterrows()
    }

    # Map from species index → genus index
    species_to_genus_map = {
        species_labels.index(row['species']): genus_labels.index(row['genus'])
        for _, row in df_unique[['species', 'genus']].drop_duplicates().iterrows()
    }

    return family_labels, genus_labels, species_labels, genus_to_family_map, species_to_genus_map

def analyze_taxonomic_misclassifications(
    model, dataset, genus_labels, species_labels, genus_to_family, species_to_genus):
    """
    Analyze genus and species misclassifications, tracking if genus respects family,
    and if species respects genus and family levels.

    Args:
    - model (tf.keras.Model): The trained model.
    - dataset (tf.data.Dataset): Validation or test dataset.
    - genus_labels, species_labels (list): List of class labels for genus and species.
    - genus_to_family (dict): Mapping of genus to family.
    - species_to_genus (dict): Mapping of species to genus.

    Returns:
    - results (dict): Misclassification percentages for logging.
    """
    genus_respect_family, genus_mistakes = 0, 0
    species_respect_genus, species_respect_family, species_mistakes = 0, 0, 0

    for images, labels in dataset:
        _, genus_logits, species_logits = model(images, training=False)

        true_genus_indices = tf.argmax(labels['genus'], axis=1).numpy()
        true_species_indices = tf.argmax(labels['species'], axis=1).numpy()
        pred_genus_indices = np.argmax(genus_logits, axis=1)
        pred_species_indices = np.argmax(species_logits, axis=1)

        # Genus misclassification respecting family
        for true_idx, pred_idx in zip(true_genus_indices, pred_genus_indices):
            if true_idx != pred_idx:
                genus_mistakes += 1
                if genus_to_family[genus_labels[true_idx]] == genus_to_family.get(genus_labels[pred_idx], None):
                    genus_respect_family += 1

        # Species misclassification respecting genus and family
        for true_idx, pred_idx in zip(true_species_indices, pred_species_indices):
            if true_idx != pred_idx:
                species_mistakes += 1
                true_genus = species_to_genus[species_labels[true_idx]]
                pred_genus = species_to_genus.get(species_labels[pred_idx], None)

                if true_genus == pred_genus:
                    species_respect_genus += 1
                if genus_to_family[true_genus] == genus_to_family.get(pred_genus, None):
                    species_respect_family += 1

    results = {
        "genus_respect_family": (genus_respect_family / genus_mistakes * 100) if genus_mistakes > 0 else 0,
        "species_respect_genus": (species_respect_genus / species_mistakes * 100) if species_mistakes > 0 else 0,
        "species_respect_family": (species_respect_family / species_mistakes * 100) if species_mistakes > 0 else 0
    }

    return results


class CryptoVisionAI:
    def __init__(self, model_path, family_names, genus_names, species_names):
        """
        Initialize the CryptoVisionAI class.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.family_names = family_names
        self.genus_names = genus_names
        self.species_names = species_names
        self.input_size = self.model.input_shape[1:3]
        self._image = None
        self._image_array = None

    @property
    def family_model(self):
        return tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('family').output)

    @property
    def genus_model(self):
        return tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('genus').output)

    @property
    def species_model(self):
        return tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('species').output)

    @property
    def image(self):
        return self._image

    @property
    def image_array(self):
        return self._image_array

    @property
    def confidence(self):
        try:
            family_pred = self.preds[0].max() if self.preds[0] is not None else "Unknown"
            genus_pred = self.preds[1].max() if self.preds[1] is not None else "Unknown"
            species_pred = self.preds[2].max() if self.preds[2] is not None else "Unknown"
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
            family_pred = self.family_names[np.argmax(preds[0])] if preds[0] is not None else "Unknown"
            genus_pred = self.genus_names[np.argmax(preds[1])] if preds[1] is not None else "Unknown"
            species_pred = self.species_names[np.argmax(preds[2])] if preds[2] is not None else "Unknown"
            return (family_pred, genus_pred, species_pred)
        except IndexError:
            return ("Unknown", "Unknown", "Unknown")

    def predict(self, input_data, return_raw=False, top_k=1):
        """
        Predict family, genus, and species for a given input.
        
        Args:
            input_data (str, np.array, tf.data.Dataset, or pd.DataFrame): Input image path, numpy array, TensorFlow dataset, or pandas dataframe.
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
                raise FileNotFoundError(f"The provided path does not exist: {input_data}")
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
                    
                    predictions.append({
                        "family": {"label": family_label, "confidence": float(family_conf)},
                        "genus": {"label": genus_label, "confidence": float(genus_conf)},
                        "species": {"label": species_label, "confidence": float(species_conf)}
                    })
            return raw_predictions if return_raw else predictions
        
        else:
            raise TypeError("Unsupported input type. Supported types: str (path), np.ndarray, tf.data.Dataset, pd.DataFrame")

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
        if level == 'family':
            model = self.family_model
        elif level == 'genus':
            model = self.genus_model
        elif level == 'species':
            model = self.species_model
        else:
            raise ValueError("Level must be one of ['family', 'genus', 'species']")
        
        # Predict class and get the predicted index
        preds = self.model.predict(self.image_array, verbose=0)
        class_index = np.argmax(preds[['family', 'genus', 'species'].index(level)])
        
        # Generate saliency map
        score = CategoricalScore([class_index])
        saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=False)
        saliency_map = saliency(score, self.image_array, smooth_samples=smooth_samples, smooth_noise=smooth_noise)
        return (saliency_map)

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
        #plt.subplot(1, 2, 1)
        #plt.title("Original Image")
        #plt.imshow(self.image)
        #plt.axis('off')

        #plt.subplot(1, 2, 2)
        plt.title("Saliency Map Overlay")
        plt.imshow(self.image)
        plt.imshow(saliency_map[0], cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.show()

    def generate_lime_explanation(self, top_labels=3, num_samples=1000):
        """
        Generate LIME explanation for the given image.
        
        Args:
            image_path (str): Path to the input image.
            top_labels (int): Number of top labels to explain.
            num_samples (int): Number of samples to generate for LIME.
        Returns:
            explanation (lime.explanation): LIME explanation result.
        """
        if self.image is None or self.image_array is None:
            raise ValueError("No image loaded. Use predict or load an image first.")
        
        # Load and preprocess the image
        image = np.squeeze(self._image_array)
        
        # Define prediction function for LIME
        def predict_function(images):
            preds = self.model.predict(images, verbose=0)
            return preds[2]
        
        explainer = LimeImageExplainer()
        explanation = explainer.explain_instance(
            image, 
            predict_function, 
            top_labels=top_labels, 
            num_samples=num_samples
        )
        return explanation

    def plot_lime_results(self, explanation, positive_only=False, negative_only=True, hide_rest=True, num_features=5, figure_size=(10, 5)):
        """
        Plot the LIME explanation results.
        
        Args:
            explanation (lime.explanation): LIME explanation result.
            label (int): The label index to explain.
            positive_only (bool): Show only positive contributions.
            hide_rest (bool): Hide regions not contributing to the label.
            num_features (int): Number of features to display.
            figure_size (tuple): Size of the matplotlib figure.
        """
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=positive_only,
            negative_only=negative_only,
            hide_rest=hide_rest,
            num_features=num_features
        )
        
        plt.figure(figsize=figure_size)
        plt.imshow(self.image)
        plt.imshow(mark_boundaries(temp, mask), cmap='jet', alpha=0.5)
        plt.title("LIME Explanation")
        plt.axis('off')
        plt.show()


class INaturalistScraper:
    def __init__(self, download_dir="images"):
        """
        Initialize the iNaturalist scraper with a directory to save images.
        
        Parameters:
        - download_dir (str): Directory to save downloaded images.
        """
        self.download_dir = download_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
    
    def fetch_observations(self, taxon_name, rank, per_page=30, page=1):
        """
        Fetch observations for a given taxon name and rank from iNaturalist API.
        
        Args:
            taxon_name (str): Name of the taxon (family, genus, or species).
            rank (str): The rank of the taxon ('family', 'genus', 'species').
            per_page (int): Number of observations to fetch per page.
            page (int): Page number to fetch.
        
        Returns:
            dict: JSON response from the API containing observations.
        """
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_name": taxon_name,
            "rank": rank,
            "per_page": per_page,
            "page": page,
            "photos": True
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data from iNaturalist: {response.status_code}")
            return None

    def download_image(self, url, save_path):
        """
        Download an image from a URL and save it to the specified path.
        
        Args:
            url (str): URL of the image to download.
            save_path (str): Local path to save the image.
        """
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Image downloaded: {save_path}")
            else:
                print(f"Failed to download image from {url} (Status code: {response.status_code})")
        except Exception as e:
            print(f"Error downloading image: {e}")
    
    def download_taxon_images(self, taxon_name, rank, max_images=10):
        """
        Download images of a specific taxon (family, genus, or species) from iNaturalist.
        
        Args:
            taxon_name (str): Name of the taxon.
            rank (str): Rank of the taxon ('family', 'genus', 'species').
            max_images (int): Maximum number of images to download.
        """
        downloaded_count, page = 0, 1
        while downloaded_count < max_images:
            observations = self.fetch_observations(taxon_name, rank, page=page)
            if not observations or not observations.get('results'):
                print("No more observations found.")
                break
            
            for result in observations['results']:
                if downloaded_count >= max_images:
                    break
                for photo in result.get('photos', []):
                    image_url = photo.get('url')
                    if image_url:
                        original_url = image_url.replace("square", "original")
                        image_id = photo.get('id')
                        extension = original_url.split('.')[-1]
                        save_path = os.path.join(self.download_dir, f"{taxon_name}_{image_id}.{extension}")
                        self.download_image(original_url, save_path)
                        downloaded_count += 1
            page += 1
        print(f"Downloaded {downloaded_count} images for taxon '{taxon_name}' with rank '{rank}'.")


class ImageAttributes:
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)  # Ensure it's a Path object
        self.img = Image.open(self.image_path)
        
        # Basic Attributes
        self.hash = str(imagehash.phash(self.img))  # Perceptual hash
        self.width, self.height = self.img.size  # Dimensions
        self.aspect_ratio = round(self.width / self.height, 2)  # Aspect ratio
        self.format = self.img.format  # Image format (JPG, PNG, etc.)
        self.mode = self.img.mode  # Color mode (RGB, Grayscale, etc.)
        self.file_size = self.image_path.stat().st_size  # File size in bytes

        # Image Quality & Color Analysis
        self.mean_brightness = self.calculate_mean_brightness()
        self.contrast = self.calculate_contrast()
        self.entropy = self.img.entropy()  # Image entropy (higher = more detail)
        self.blur_score = self.calculate_blur()  # Laplacian blur detection
        self.color_histogram = self.calculate_color_histogram()
        self.dominant_color = self.calculate_dominant_color()

        # Metadata Extraction
        self.metadata = self.extract_exif_metadata()

    def calculate_mean_brightness(self):
        """Calculate the mean pixel brightness."""
        stat = ImageStat.Stat(self.img)
        return round(sum(stat.mean) / len(stat.mean), 2)

    def calculate_contrast(self):
        """Calculate the contrast of the image using pixel standard deviation."""
        try:
            stat = ImageStat.Stat(self.img)
            variance = [max(v, 0) for v in stat.var]  # Ensure non-negative values
            return round(sum(math.sqrt(v) for v in variance) / len(variance), 2)
        except Exception as e:
            print(f"Error calculating contrast for {self.image_path}: {e}")
            return None  # Return None if contrast calculation fails

    def calculate_blur(self):
        """Estimate blur using variance of Laplacian."""
        img_cv = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        return round(cv2.Laplacian(img_cv, cv2.CV_64F).var(), 2) if img_cv is not None else None

    def calculate_color_histogram(self):
        """Compute a simple RGB histogram to detect grayscale images."""
        hist = self.img.histogram()
        return {
            "R": sum(hist[0:256]), 
            "G": sum(hist[256:512]), 
            "B": sum(hist[512:768])
        }

    def calculate_dominant_color(self):
        """Compute the dominant color in the image."""
        # Ensure image is in RGB mode
        img = self.img.convert("RGB")
        
        # Convert image to NumPy array
        img_array = np.array(img)
        
        # Ensure it has 3 channels (RGB)
        if len(img_array.shape) == 2:  # Grayscale image, convert to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Reshape to a list of pixels and compute the mean color
        pixels = img_array.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        
        return tuple(map(int, dominant_color))  # Convert to (R, G, B) format

    def extract_exif_metadata(self):
        """Extracts EXIF metadata if available."""
        try:
            exif_data = self.img._getexif()
            if exif_data:
                return {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        except AttributeError:
            return {}
        return {}

    def to_dict(self, min_size):
        """
        Convert image attributes to a dictionary.
        Adds `flag_small` to indicate if the image is smaller than `min_size`.
        """
        return {
            "image_path": str(self.image_path),
            "hash": self.hash,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "format": self.format,
            "mode": self.mode,
            "file_size": self.file_size,
            "brightness": self.mean_brightness,
            "contrast": self.contrast,
            "entropy": self.entropy,
            "blur_score": self.blur_score,
            "dominant_color": self.dominant_color,
            "flag_small": self.width < min_size and self.height < min_size
        }

    def __repr__(self):
        return (
            f"Image: {self.image_path}, Format: {self.format}, Mode: {self.mode}, "
            f"Size: {self.width}x{self.height}, Aspect Ratio: {self.aspect_ratio}, "
            f"Brightness: {self.mean_brightness}, Contrast: {self.contrast}, "
            f"Entropy: {self.entropy}, Blur Score: {self.blur_score}, "
            f"Dominant Color: {self.dominant_color}"
        )


def build_catalog(path: Path, source: str, min_size: int, num_workers: int = None):
    """
    Build dataset by extracting image attributes in parallel.
    """
    
    # Get initial DataFrame
    df = image_dir_pandas(path, source)
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda img: ImageAttributes(img).to_dict(min_size), df["image_path"]))

    # Remove failed results (None values)
    results = [res for res in results if res is not None]

    # Convert results to DataFrame
    df_attributes = pd.DataFrame.from_records(results)

    # Merge attributes back with the original metadata
    df = df.merge(df_attributes, on="image_path", how="left")
    
    # Flag duplicates
    df['duplicates'] = df.duplicated(subset='hash', keep='first')
    
    # Round entropy to 5 digits
    df['entropy'] = df['entropy'].round(5)

    return df

