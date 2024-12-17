import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

def image_directory_to_pandas(image_path):
    """
    Create a pandas DataFrame with image paths and taxonomic labels extracted from a directory structure.

    Parameters:
    ----------
    image_path : str
        The root directory containing subfolders with images.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing image paths and label information. Columns include:
        - 'path': The full path to the image.
        - 'folder_label': The folder name, representing the original label (format: 'family_genus_species').
        - 'family': Extracted family name from the folder label.
        - 'genus': Extracted genus name from the folder label.
        - 'species': Combination of genus and species names (e.g., 'genus species').

    Raises:
    ------
    ValueError:
        If the folder label format does not match the expected 'family_genus_species' format.
    """
    labels = []
    paths = []

    # Walk through the directory and collect image paths and labels
    for root_dir, _, filenames in os.walk(image_path):
        for filename in filenames:
            # Ignore hidden files and non-image files
            if filename.startswith('.') or os.path.splitext(filename)[1].lower() not in {".jpeg", ".png", ".jpg"}:
                continue

            # Extract the folder name as the label, ignoring 'GT' directories
            folder_label = os.path.basename(root_dir)
            if folder_label != "GT":
                labels.append(folder_label)
                paths.append(os.path.join(root_dir, filename))

    # Create DataFrame with paths and folder labels
    df = pd.DataFrame({'image_path': paths, 'folder_label': labels})
    df['folder_label'] = df['folder_label'].astype("category")

    # Split the folder_label into 'family', 'genus', and 'species'
    try:
        df[['family', 'genus', 'species']] = df['folder_label'].str.split("_", expand=True)
        df['species'] = df['genus'] + " " + df['species']
    except ValueError as e:
        raise ValueError(
            "Error splitting folder labels. Ensure that your folder structure follows 'family_genus_species' format."
        ) from e

    # Return the dataframe with specified columns
    return df[['image_path', 'folder_label', 'family', 'genus', 'species']]


def split_image_dataframe(df, test_size=0.2, val_size=0.1, random_state=42, stratify_by='folder_label'):
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
    
    return train_df, val_df, test_df


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
    img = tf.io.read_file(image_path)
    # Decode the image
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    img = tf.image.resize(img, image_size)

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


def tf_dataset_from_pandas(df, batch_size=32, image_size=(224,224), shuffle=True):
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
    image_label_ds = path_ds.map(
        lambda path, family, genus, species: process_image_and_labels(
            path, family, genus, species, family_labels, genus_labels, species_labels, image_size=image_size
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Shuffle, batch, and prefetch the dataset
    image_label_ds = image_label_ds\
        .batch(batch_size)\
        .cache()\
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        
    if shuffle:
        image_label_ds = image_label_ds.shuffle(buffer_size=len(df))

    return (
        image_label_ds,
        family_labels.numpy().tolist(),
        genus_labels.numpy().tolist(),
        species_labels.numpy().tolist(),
    )


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
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
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
        Initializes the CryptoVisionAI class.
        
        Parameters:
            model (tf.keras.Model): The trained model for prediction.
            family_names (list): List of family label names.
            genus_names (list): List of genus label names.
            species_names (list): List of species label names.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.family_names = family_names
        self.genus_names = genus_names
        self.species_names = species_names
        self.target_size = self.model.input_shape[1:3]
    
    @property  
    def image(self):
        return tf.keras.preprocessing.image.load_img(self.image_path, target_size=self.target_size)    
    
    @property
    def family_model(self):
        if not hasattr(self, '_family_model'):
            self._family_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('family').output
            )
        return self._family_model
    
    @property
    def genus_model(self):
        if not hasattr(self, '_genus_model'):
            self._genus_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('genus').output
            )
        return self._genus_model
    
    @property
    def species_model(self):
        if not hasattr(self, '_species_model'):
            self._species_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('species').output
            )
        return self._species_model
    
    @property
    def image_array(self):
        try:
            image_array = tf.keras.preprocessing.image.img_to_array(self.image)
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            print(f"Error processing image at {self.image_path}: {e}")
            return None

    def decoder(self, preds):
        try:
            family_pred = self.family_names[np.argmax(preds[0])] if preds[0] is not None else "Unknown"
            genus_pred = self.genus_names[np.argmax(preds[1])] if preds[1] is not None else "Unknown"
            species_pred = self.species_names[np.argmax(preds[2])] if preds[2] is not None else "Unknown"
            return [family_pred, genus_pred, species_pred]
        except IndexError:
            return ["Unknown", "Unknown", "Unknown"]
    
    def complete_prediction(self, preds):
        """
        Predicts labels for a single image.

        Parameters:
            image (np.array): Preprocessed image.
            return_probabilities (bool): Whether to return probabilities.

        Returns:
            list: Predicted labels.
        """
        
        results = {
            'family': {},
            'genus': {},
            'species': {},
        }
        
        family_pred = np.argmax(preds[0])
        genus_pred = np.argmax(preds[1])
        species_pred = np.argmax(preds[2])
        
        family_label, genus_label, species_label = self.decoder(preds)
        
        # Return the predicted names
        results['family']['name'] = family_label
        results['genus']['name'] = genus_label
        results['species']['name'] = species_label
        
        # Return the predicted labels
        results['family']['label'] = family_pred
        results['genus']['label'] = genus_pred
        results['species']['label'] = species_pred
        
        # Return the predicted probabilities
        results['family']['prob'] = preds[0].max()
        results['genus']['prob'] = preds[1].max()
        results['species']['prob'] = preds[2].max()
        
        return results

    def predict_from_path(self, image_path, prediction_type='labels'):
        """
        Predicts labels from a list of image paths.
        
        Parameters:
            image_paths (list): List of image paths.
            batch_size (int): Size of batches for prediction.
        
        Returns:
            list: Predictions for all images.
        """
        results = []
        self.image_path = image_path
        preds = self.model.predict(self.image_array, verbose=0)
        if prediction_type == 'complete':
            return self.complete_prediction(preds)
        elif prediction_type == 'raw':
            return preds
        elif prediction_type == 'labels':
            return self.decoder(preds)

    def predict_from_dataframe(self, dataframe, reference_column, batch_size=64):
        """
        Predicts labels from a pandas DataFrame with image paths in batches, with progress tracking.
        
        Parameters:
            dataframe (pd.DataFrame): DataFrame containing image paths and metadata.
            reference_column (str): Column containing the image paths.
            batch_size (int): Number of images to process per batch.
        
        Returns:
            list: List of predictions for all images.
        """
        predictions = []
        num_batches = (len(dataframe) + batch_size - 1) // batch_size  # Calculate the total number of batches

        with tqdm(total=num_batches, desc="Predicting Batches") as pbar:
            for i in range(0, len(dataframe), batch_size):
                batch_df = dataframe.iloc[i:i + batch_size]
                batch_images = np.vstack([self.image_path_to_array(row[reference_column]) for _, row in batch_df.iterrows()])
                batch_preds = self.model.predict(batch_images, verbose=0)
                decoded_batch = [
                    self.decoder([batch_preds[0][i], batch_preds[1][i], batch_preds[2][i]]) 
                    for i in range(len(batch_preds[0]))
                ]
                predictions.extend(decoded_batch)
                pbar.update(1)
        
        return predictions

    def predict_from_tf_dataset(self, tf_dataset):
        """
        Predicts labels from a TensorFlow dataset.

        Parameters:
            tf_dataset (tf.data.Dataset): Dataset containing images and labels.

        Returns:
            list: Decoded predictions for all images in the dataset.
        """
        predictions = []
        for batch in tqdm(tf_dataset, desc="Predicting with Dataset"):
            img_batch, _ = batch  # Assuming labels are the second element
            preds = self.model.predict(img_batch, verbose=0)
            
            # Decode predictions for the batch
            decoded_batch = [
                self.decoder([preds[0][i], preds[1][i], preds[2][i]]) 
                for i in range(len(preds[0]))
            ]
            predictions.extend(decoded_batch)
            
        return predictions
    
    def saliency_map(self, level, image_path, figure_size = (15, 8), image_show = True):
        
        self.image_path = image_path
        preds = self.model.predict(self.image_array, verbose=0)
        preds = self.complete_prediction(preds)
        class_index = preds[level]['label']
        
        if level == 'family':
            model = self.family_model
        elif level == 'genus':
            model = self.genus_model
        elif level == 'species':
            model = self.species_model
            
        score = CategoricalScore([class_index])
        saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=False)
        saliency_map = saliency(score, self.image_array, smooth_samples=20, smooth_noise=0.2)
        saliency_map = normalize(saliency_map)
        
        if image_show:
            # Plot results
            plt.figure(figsize=figure_size)
            plt.suptitle(f"Predicted Class: {preds[level]['name']} - {preds[level]['prob']*100:.2f}%", fontsize=14)
            
            # Original Image
            plt.subplot(1, 3, 1)
            plt.title("Original")
            plt.imshow(self.image)
            plt.axis('off')
            
            # Saliency Map
            plt.subplot(1, 3, 2)
            plt.title("Saliency Map")
            plt.imshow(saliency_map[0], cmap='jet')
            plt.axis('off')
            
            # Saliency overlay
            plt.subplot(1, 3, 3)
            plt.title("Saliency Overlay")
            plt.imshow(self.image)
            plt.imshow(saliency_map[0], cmap='jet', alpha=0.5)  # Overlay saliency map with transparency
            plt.axis('off')
        
        return saliency_map

