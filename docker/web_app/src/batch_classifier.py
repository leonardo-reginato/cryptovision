import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Fish Species Batch Classifier", page_icon="🐟", layout="wide")

# Settings
settings = {
    "model_path": '/Users/leonardo/Documents/Projects/cryptovision/models/phorcys_simple_hacpl_384_rn50v2_v2411131337.keras',
    "labels_path": "/Users/leonardo/Documents/Projects/cryptovision/data/processed/cv_images_dataset",
    "img_size": (384, 384),
    "min_confidence": 0.5,  # Minimum confidence threshold
}

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(settings["model_path"])

model = load_model()

# Generate family, genus, and species labels from the directory structure
@st.cache_data
def create_labels_from_path(path):
    family_labels, genus_labels, species_labels = set(), set(), set()
    for root_dir, dirs, _ in os.walk(path):
        for folder_name in dirs:
            family_name, genus_name, species_name = folder_name.split('_')
            family_labels.add(family_name)
            genus_labels.add(genus_name)
            species_labels.add(f"{genus_name} {species_name}")
    return sorted(list(family_labels)), sorted(list(genus_labels)), sorted(list(species_labels))

family_labels, genus_labels, species_labels = create_labels_from_path(settings["labels_path"])

# Function to predict family, genus, and species
def predict_image(image, model, family_labels, genus_labels, species_labels, img_size, min_confidence):
    # Resize the image to the model's input size
    img = image.resize(img_size)
    img_array = np.expand_dims(np.array(img), axis=0)

    # Model predictions
    family_pred, genus_pred, species_pred = model.predict(img_array, verbose=0)

    # Top-1 predictions
    family_index = np.argmax(family_pred[0])
    genus_index = np.argmax(genus_pred[0])
    species_index = np.argmax(species_pred[0])

    # Retrieve predictions and confidences
    family_conf = family_pred[0][family_index]
    genus_conf = genus_pred[0][genus_index]
    species_conf = species_pred[0][species_index]

    # Apply confidence threshold
    family = family_labels[family_index] if family_conf >= min_confidence else "Unknown"
    genus = genus_labels[genus_index] if genus_conf >= min_confidence else "Unknown"
    species = species_labels[species_index] if species_conf >= min_confidence else "Unknown"

    return family, family_conf, genus, genus_conf, species, species_conf

# Streamlit App Layout
st.title("Fish Species Classifier 🐟🔍")
st.markdown(
    """
    Upload a set of fish images, and this app will predict the family, genus, 
    and species for each image. Metrics will also be displayed based on the predictions.
    """
)

# File uploader
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    results = []
    start_time = time.time()  # Start timing
    with st.spinner("Classifying images..."):
        for uploaded_file in uploaded_files:
            # Open the uploaded image directly
            img = Image.open(uploaded_file).convert("RGB")

            # Extract ground truth species from the filename
            ground_truth_species = os.path.splitext(uploaded_file.name)[0].strip().lower()
            
            # Predict species
            family, family_conf, genus, genus_conf, species, species_conf = predict_image(
                img, model, family_labels, genus_labels, species_labels, 
                settings["img_size"], settings["min_confidence"]
            )
            
            # Compare ground truth with predicted species (case-insensitive comparison)
            is_correct = ground_truth_species == species.strip().lower()
            
            # Append results
            results.append((img, family, family_conf, genus, genus_conf, species, species_conf, is_correct))

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    total_images = len(results)
    avg_time_per_image = total_time / total_images if total_images > 0 else 0

    # Calculate metrics
    correct_predictions = sum(1 for _, _, _, _, _, _, _, is_correct in results if is_correct)
    percent_correct = (correct_predictions / total_images) * 100 if total_images > 0 else 0

    avg_family_conf = np.mean([family_conf for _, _, family_conf, _, _, _, _, _ in results]) * 100
    avg_genus_conf = np.mean([genus_conf for _, _, _, _, genus_conf, _, _, _ in results]) * 100
    avg_species_conf = np.mean([species_conf for _, _, _, _, _, _, species_conf, _ in results]) * 100

    # Display metrics
    st.subheader("Prediction Metrics")
    st.markdown(f"### Correct Predictions: {correct_predictions}/{total_images} ({percent_correct:.2f}%)")
    st.markdown(f"### Total Time: {total_time:.2f} seconds")
    st.markdown(f"### Average Time per Image: {avg_time_per_image:.2f} seconds")

    st.markdown("**Average Confidence Levels**")
    st.markdown("**Family:**")
    st.progress(avg_family_conf / 100)
    st.write(f"{avg_family_conf:.2f}%")
    
    st.markdown("**Genus:**")
    st.progress(avg_genus_conf / 100)
    st.write(f"{avg_genus_conf:.2f}%")
    
    st.markdown("**Species:**")
    st.progress(avg_species_conf / 100)
    st.write(f"{avg_species_conf:.2f}%")

    # Toggle for individual predictions
    with st.expander("Show Predictions and Results"):
        st.subheader("Classification Results")
        for img, family, family_conf, genus, genus_conf, species, species_conf, is_correct in results:
            st.image(img.resize(settings["img_size"]), use_column_width=False, caption="Predicted Image")

            # Display predictions
            st.markdown(
                f"**Family:** {family} ({family_conf * 100:.2f}%)  \n"
                f"**Genus:** {genus} ({genus_conf * 100:.2f}%)  \n"
                f"**Species:** {species} ({species_conf * 100:.2f}%)"
            )

            # Display correctness with icon
            icon = "✅" if is_correct else "❌"
            st.markdown(f"**Comparison Result:** {icon}")
            st.markdown("---")
else:
    st.info("Please upload images to start classification.")