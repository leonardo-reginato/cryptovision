import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Application Settings
settings = {
    "model_path": '/Users/leonardo/Documents/Projects/cryptovision/models/phorcys_large_hacpl_rn50v2_v2411051344.keras',
    "labels_path": "/Users/leonardo/Documents/Projects/cryptovision/data/processed/cv_images_dataset",
    "img_size": (299, 299),
}

# Load the model
cv_model = tf.keras.models.load_model(settings['model_path'])

# Define prediction and label generation functions
def predict_image(image, model, family_labels, genus_labels, species_labels, image_size=(299, 299), top_k=5):
    """ Predict the top-k family, genus, and species from an image using a trained model. """
    
    # Preprocess the image
    img = image.resize(image_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict family, genus, and species
    family_pred, genus_preds, species_preds = model.predict(img_array)

    # Extract top-k predictions for family, genus, and species
    top_k_family = [(family_labels[i], family_pred[0][i]) for i in np.argsort(family_pred[0])[-top_k:][::-1]]
    top_k_genus = [(genus_labels[i], genus_preds[0][i]) for i in np.argsort(genus_preds[0])[-top_k:][::-1]]
    top_k_species = [(species_labels[i], species_preds[0][i]) for i in np.argsort(species_preds[0])[-top_k:][::-1]]

    return top_k_family, top_k_genus, top_k_species

def create_labels_from_path(path):
    """ Create lists of unique family, genus, and species labels from directory structure. """
    
    family_labels, genus_labels, species_labels = set(), set(), set()
    
    for root_dir, dirs, files in os.walk(path):
        for folder_name in dirs:
            family_name, genus_name, species_name = folder_name.split('_')
            family_labels.add(family_name)
            genus_labels.add(genus_name)
            species_labels.add(f"{genus_name} {species_name}")

    return sorted(list(family_labels)), sorted(list(genus_labels)), sorted(list(species_labels))

# Load labels from the dataset path
family_labels, genus_labels, species_labels = create_labels_from_path(settings['labels_path'])

# Streamlit app layout and configuration
st.set_page_config(page_title="CryptoVision", page_icon="🐟")

# App logo and title
logo_path = "/CryptoVision/images/logo_brandllab_v2.png"
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    col1, col2, col3 = st.columns([1, 1, 1])
    col2.image(logo, use_column_width=False, width=200)

st.title("CryptoVision Image Classifier 🐟🔍")
st.markdown(
    """
    **CryptoVision** is a deep learning model designed to classify fish species from images. 
    Upload an image of a fish, click "Classify," and the model will predict its family, genus, and species. 
    Let's explore the underwater world together! 🌊🐠
    """
)

# Helper function for color-coding confidence
def color_confidence(confidence):
    if confidence > 0.9:
        return "green"
    elif 0.7 < confidence <= 0.9:
        return "blue"
    elif 0.6 < confidence <= 0.7:
        return "orange"
    else:
        return "red"

# Image uploader for user to provide an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Button to trigger classification
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        # Perform the prediction
        img = Image.open(uploaded_image)
        top_k_family, top_k_genus, top_k_species = predict_image(img, cv_model, family_labels, genus_labels, species_labels, image_size=settings['img_size'])

        # Display only the top-1 prediction for each level
        st.subheader("Prediction Results")
        
        # Display top-1 Family Prediction
        family, family_confidence = top_k_family[0]
        family_confidence_percentage = int(family_confidence * 100)
        family_color = color_confidence(family_confidence)
        
        st.write(f"**Family:** *{family}*")
        progress_bar = st.progress(family_confidence_percentage)
        st.markdown(f"<span style='color:{family_color}; font-size: 16px;'>{family_confidence_percentage}%</span>", unsafe_allow_html=True)

        # Display top-1 Genus Prediction
        genus, genus_confidence = top_k_genus[0]
        genus_confidence_percentage = int(genus_confidence * 100)
        genus_color = color_confidence(genus_confidence)
        
        st.write(f"**Genus:** *{genus}*")
        progress_bar = st.progress(genus_confidence_percentage)
        st.markdown(f"<span style='color:{genus_color}; font-size: 16px;'>{genus_confidence_percentage}%</span>", unsafe_allow_html=True)

        # Display top-1 Species Prediction
        species, species_confidence = top_k_species[0]
        species_confidence_percentage = int(species_confidence * 100)
        species_color = color_confidence(species_confidence)
        
        st.write(f"**Species:** *{species}*")
        progress_bar = st.progress(species_confidence_percentage)
        st.markdown(f"<span style='color:{species_color}; font-size: 16px;'>{species_confidence_percentage}%</span>", unsafe_allow_html=True)

    else:
        st.info("Click 'Classify' to see the predictions.")

# Optional: Add feedback section
st.subheader("Feedback")
st.write("Was the prediction correct in your opinion?")
user_feedback = st.radio("", ["Yes", "No"], index=0, key="user_feedback")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")