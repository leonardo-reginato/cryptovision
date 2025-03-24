import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import copy
import warnings
import keras
import skimage.io
import skimage.segmentation
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import shap

from cryptovision import tools
import cryptovision.dataset as dataset

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Global input parameters
IMG_SIZE = (320, 320)
SRC_PATH = '/Users/leonardo/Library/CloudStorage/GoogleDrive-leonardofonseca.r@gmail.com/My Drive/04_projects/cryptovision/data/processed/Sources'
FTUN_MODEL_PATH = '/Users/leonardo/Documents/Projects/cryptovision/models/CVisionClassifier/CVision_GS_L_1_2503.18.1022/model_trained.keras'
IMAGE_PATH = '/Users/leonardo/Library/CloudStorage/GoogleDrive-leonardofonseca.r@gmail.com/My Drive/04_projects/cryptovision/data/processed/Sources/SJB/Species/v241226/images/Gobiidae_Eviota_melasma/lab_Eviota_melasma_0009.jpeg'


def load_dataset(src_path, min_samples=90):
    """Load dataset from the source path and update image paths."""
    df = dataset.main(src_path=src_path, min_samples=min_samples)
    # Update image paths to reflect local storage
    df['image_path'] = df['image_path'].apply(
        lambda x: x.replace('/Volumes/SANDISK/CryptoVision/Sources', src_path)
    )
    return df


def split_data(df, seed):
    """Split dataframe into training, validation, and test sets."""
    train_df, val_df, test_df = tools.split_dataframe(
        df,
        test_size=0.15,
        val_size=0.15,
        stratify_by='species',
        random_state=seed
    )
    return train_df, val_df, test_df


def create_dataset_names(df):
    """Create sorted lists for family, genus, and species names."""
    names = {
        'family': sorted(df['family'].unique()),
        'genus': sorted(df['genus'].unique()),
        'species': sorted(df['species'].unique()),
    }
    return names


def display_superpixel_boundaries(image, superpixels):
    """Display the image with superpixel boundaries marked."""
    # Convert PIL Image to numpy array if needed
    image_array = np.array(image)
    # Normalize image (assumes values in 0-255 range)
    image_float = image_array.astype(np.float64) / 255.0
    # Create an image with marked boundaries
    boundary_image = skimage.segmentation.mark_boundaries(image_float / 2 + 0.5, superpixels)
    plt.figure(figsize=(8, 8))
    plt.imshow(boundary_image)
    plt.title('Superpixel Boundaries')
    plt.axis('off')
    plt.show()


def perturb_image(img, perturbation, segments):
    """Apply a perturbation to an image based on superpixel segmentation.

    Args:
        img: Input image (numpy array).
        perturbation: Binary array indicating which superpixels to keep.
        segments: Superpixel segmentation mask.

    Returns:
        Perturbed image.
    """
    # Identify active superpixels
    active_pixels = np.where(perturbation == 1)[0]
    # Create a mask based on active superpixels
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    # Copy image to avoid modifying the original
    perturbed_image = copy.deepcopy(img)
    # Apply mask to each channel
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image


def get_gradcam_heatmap(model, image, class_index, last_conv_layer_name=None):
    """Generate a Grad-CAM heatmap for a given image and class index using the provided model."""
    
    model = model.layer[2]
    
    import tensorflow as tf
    # If last_conv_layer_name is not provided, find the last conv layer automatically
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer_name = layer.name
                break
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def main(src_path, model_path, image_path, img_size=(320, 320), seed=42, min_samples=90, num_segments=300, num_perturb=4000, kernel_width=0.25, num_top_features=20):
    # Load dataset and update image paths
    df = load_dataset(src_path, min_samples=min_samples)
    train_df, val_df, test_df = split_data(df, seed)
    names = create_dataset_names(df)

    # (Optional) Create test dataset for model evaluation
    test_ds = tools.tensorflow_dataset(
        test_df,
        batch_size=32,
        image_size=img_size,
        shuffle=False,
    )

    # Initialize CryptoVisionAI with the given model and dataset names
    ai = tools.CryptoVisionAI(
        model_path=model_path,
        family_names=names['family'],
        genus_names=names['genus'],
        species_names=names['species'],
        safe_mode=False,
    )

    # Predict using the model (image_path can be updated to a valid image if needed)
    preds = ai.predict(image_path, return_raw=True)
    # Extract top 5 predicted classes
    top_pred_classes = preds[2][0].argsort()[-5:][::-1]

    # Check model alignment: print predicted class and confidence
    predicted_probs = np.array(preds[2][0])
    predicted_class = top_pred_classes[0] if predicted_probs.shape[0] > 1 else 0
    confidence = predicted_probs[predicted_class]
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

    # Prepare background dataset (take first 50 from test_ds)
    background_images = next(iter(test_ds.cache().take(1).repeat()))[0][:50].numpy()

    # Normalize background if needed
    background_images = background_images.astype(np.float32) / 255.0

    # Get the image from the AI model (assumed to be a PIL Image)
    Xi = ai.image
    Xi = np.array(Xi)

    # Prepare test image
    test_image = np.expand_dims(Xi.astype(np.float32) / 255.0, axis=0)

    # Choose output index to explain (species)
    explanation_level = 'species'
    output_levels = {'family': 0, 'genus': 1, 'species': 2}
    class_index = top_pred_classes[0]

    # Initialize SHAP DeepExplainer
    from tensorflow.keras.models import Model
    from shap.explainers import GradientExplainer
    species_output = ai.model.outputs[2]
    species_model = Model(inputs=ai.model.inputs, outputs=species_output)
    explainer = GradientExplainer(species_model, background_images)
    shap_values = explainer.shap_values(test_image)

    # Extract SHAP values for the explanation level
    shap_map = shap_values[0][0].sum(axis=-1)

    # Threshold the saliency map to isolate the fish body
    saliency_map = ai.generate_saliency_map('species', smooth_samples=20, smooth_noise=0.2)
    if saliency_map.ndim == 3 and saliency_map.shape[0] == 1:
        saliency_map = saliency_map[0]
    elif saliency_map.ndim == 4:
        saliency_map = saliency_map[0, 0]
    saliency_threshold_mask = saliency_map >= np.percentile(saliency_map, 80)

    # Apply masked SLIC segmentation
    masked_image = Xi.copy()
    masked_image[~saliency_threshold_mask] = 0
    # Apply Felzenszwalb segmentation on the masked image
    superpixels = skimage.segmentation.felzenszwalb(masked_image, scale=100, sigma=0.8, min_size=50)
    num_superpixels = np.unique(superpixels).shape[0]
    
    # Remove superpixels that lie entirely outside the saliency mask
    retained_superpixels = []
    for i in range(num_superpixels):
        if np.any((superpixels == i) & saliency_threshold_mask):
            retained_superpixels.append(i)
    retained_superpixels = np.array(retained_superpixels)
    
    # Re-map superpixels to only retained ones
    superpixel_map = {old_idx: new_idx for new_idx, old_idx in enumerate(retained_superpixels)}
    mask_remap = np.isin(superpixels, retained_superpixels)
    new_superpixels = np.full(superpixels.shape, -1)
    for old_idx in retained_superpixels:
        new_superpixels[superpixels == old_idx] = superpixel_map[old_idx]
    superpixels = new_superpixels
    num_superpixels = len(retained_superpixels)
    
    if num_superpixels == 0:
        raise ValueError("No valid superpixels remained after saliency filtering.")

    # Optionally display the masked input image for visual confirmation
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_image.astype(np.uint8))
    plt.title("Masked Image (Saliency ≥ 80th percentile)")
    plt.axis('off')
    plt.show()

    # Convert the image to a numpy array and normalize it
    Xi_array = np.array(Xi)
    Xi_float = Xi_array.astype(np.float64) / 255.0


    # Generate random perturbations for LIME analysis
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

    # Collect predictions for each perturbation
    perturbed_images = np.stack([perturb_image(Xi, pert, superpixels) for pert in perturbations])
    perturbed_images = perturbed_images.astype(np.float32) / 255.0
    predictions = ai.model.predict(perturbed_images, verbose=0)
    
    # Select specific output level for LIME explanation
    explanation_level = 'species'
    output_levels = {'family': 0, 'genus': 1, 'species': 2}
    predictions = predictions[output_levels[explanation_level]]
    
    

    # Compute distances between each perturbation and the original image (all superpixels enabled)
    original_perturbation = np.ones((1, num_superpixels))
    distances = sklearn.metrics.pairwise_distances(perturbations, original_perturbation, metric='cosine').ravel()

    # Compute weights using a kernel function
    weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))

    # Choose the class to explain: if the model returns multiple classes, use the top predicted one; otherwise, default to class 0
    if predictions.shape[1] > 1:
        class_to_explain = top_pred_classes[0]
    else:
        class_to_explain = 0

    # Fit a simple linear regression model as a surrogate for LIME
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:, class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_
    r2_score = simpler_model.score(perturbations, predictions[:, class_to_explain], sample_weight=weights)
    print(f"LIME surrogate model R² score: {r2_score:.4f}")

    # Select the top superpixels based on the regression coefficients
    threshold_value = np.percentile(coeff, 100 - (num_top_features / len(coeff) * 100))
    top_features = np.where(coeff >= threshold_value)[0]

    # Create a mask that activates only the top superpixels
    top_mask = np.zeros(num_superpixels)
    top_mask[top_features] = True
    
    saliency_map = ai.generate_saliency_map('species', smooth_samples=20, smooth_noise=0.2)
    if saliency_map.ndim == 3 and saliency_map.shape[0] == 1:
        saliency_map = saliency_map[0]
    elif saliency_map.ndim == 4:
        saliency_map = saliency_map[0, 0]

    superpixel_saliency = []
    for i in range(num_superpixels):
        mask = (superpixels == i)
        if np.any(mask):
            mean_val = saliency_map[mask].mean()
        else:
            mean_val = 0
        superpixel_saliency.append(mean_val)
    superpixel_saliency = np.array(superpixel_saliency)
    saliency_threshold = np.median(superpixel_saliency)
    saliency_mask = (superpixel_saliency >= saliency_threshold)
    final_mask = top_mask * saliency_mask

    # Generate pixel-level mask
    pixel_mask = np.zeros(superpixels.shape)
    for i in range(num_superpixels):
        if final_mask[i]:
            pixel_mask[superpixels == i] = 1

    # Display the LIME explanation by showing the most influential superpixels
    explained_image = Xi_float / 2 + 0.5
    explained_image = explained_image * pixel_mask[:, :, np.newaxis]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axs[0, 0].imshow(Xi.astype(np.uint8))
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    # Filtered superpixels overlaid on the original image
    filtered_superpixels = np.copy(superpixels)
    filtered_superpixels[~saliency_threshold_mask] = -1  # mask non-salient regions

    segmented_image_filtered = skimage.segmentation.mark_boundaries(Xi_float, filtered_superpixels, color=(1, 1, 0))
    axs[0, 1].imshow(segmented_image_filtered)
    axs[0, 1].set_title("Saliency-Filtered Superpixels")
    axs[0, 1].axis('off')

    # LIME Explanation
    axs[1, 0].imshow(explained_image)
    axs[1, 0].set_title("LIME Explanation (Top Superpixels)")
    axs[1, 0].axis('off')

    # Saliency map overlay
    axs[1, 1].imshow(Xi.astype(np.uint8))
    axs[1, 1].imshow(saliency_map, cmap='jet', alpha=0.5)
    axs[1, 1].set_title("Saliency Map Overlay")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # Additional plots: Regular saliency map and Saliency map with LIME mask
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))

    # Regular saliency map
    axs2[0].imshow(saliency_map, cmap='jet')
    axs2[0].set_title("Saliency Map (Raw)")
    axs2[0].axis('off')

    # Saliency map with LIME mask overlay
    axs2[1].imshow(Xi.astype(np.uint8))
    axs2[1].imshow(saliency_map, cmap='jet', alpha=0.5)
    axs2[1].imshow(pixel_mask, cmap='gray', alpha=0.3)
    axs2[1].set_title("Saliency Map + LIME Mask")
    axs2[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Display SHAP heatmap overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(Xi.astype(np.uint8))
    plt.imshow(shap_map, cmap='jet', alpha=0.5)
    plt.title("SHAP Heatmap Overlay (Species Class)")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main(
        src_path=SRC_PATH,
        model_path=FTUN_MODEL_PATH,
        image_path=IMAGE_PATH,
        img_size=IMG_SIZE,
        seed=SEED,
        min_samples=90,
        num_segments=150,
        num_perturb=5000,
        kernel_width=0.20,
        num_top_features=30
    )