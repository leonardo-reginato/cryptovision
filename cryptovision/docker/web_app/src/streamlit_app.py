import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import hashlib
import pandas as pd
from datetime import datetime

# Application Settings
settings = {
    "model_path": '/Users/leonardo/Documents/Projects/cryptovision/models/phorcys_simple_hacpl_384_rn50v2_v2411131337.keras',
    "labels_path": "/Users/leonardo/Documents/Projects/cryptovision/data/processed/cv_images_dataset",
    "img_size": (384, 384),
    "saved_images_folder": "/Users/leonardo/Documents/Projects/cryptovision/saved_images",
    "log_file": "/Users/leonardo/Documents/Projects/cryptovision/predictions_log.csv",
    "min_confidence": 0.4,
    "image_header": "/Users/leonardo/Documents/Projects/cryptovision/cryptovision/docker/web_app/images/fish_colage_v2.png"
}

# Ensure directories and log file exist
os.makedirs(settings['saved_images_folder'], exist_ok=True)
if not os.path.exists(settings['log_file']):
    pd.DataFrame(columns=[
        "image_hash_id", "family_pred", "family_conf", "genus_pred", "genus_conf", "species_pred", "species_conf"
    ]).to_csv(settings['log_file'], index=False)

# Load model
cv_model = tf.keras.models.load_model(settings['model_path'])

# Define functions
def predict_image(image, model, family_labels, genus_labels, species_labels, image_size=(299, 299), top_k=5):
    image = image.convert("RGB")
    img = image.resize(image_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    family_pred, genus_preds, species_preds = model.predict(img_array)

    # Extract top-k predictions
    top_k_family = [(family_labels[i], family_pred[0][i]) for i in np.argsort(family_pred[0])[-top_k:][::-1]]
    top_k_genus = [(genus_labels[i], genus_preds[0][i]) for i in np.argsort(genus_preds[0])[-top_k:][::-1]]
    top_k_species = [(species_labels[i], species_preds[0][i]) for i in np.argsort(species_preds[0])[-top_k:][::-1]]

    return top_k_family, top_k_genus, top_k_species

def create_labels_from_path(path):
    family_labels, genus_labels, species_labels = set(), set(), set()
    for root_dir, dirs, _ in os.walk(path):
        for folder_name in dirs:
            family_name, genus_name, species_name = folder_name.split('_')
            family_labels.add(family_name)
            genus_labels.add(genus_name)
            species_labels.add(f"{genus_name} {species_name}")
    return sorted(list(family_labels)), sorted(list(genus_labels)), sorted(list(species_labels))

def save_image_with_hash(image, folder_path):
    img_hash = hashlib.sha256(image.tobytes()).hexdigest()[:8]  # Shorten hash to 8 characters
    img_save_path = os.path.join(folder_path, f"{img_hash}.png")
    image.save(img_save_path)
    return img_save_path, img_hash

def log_prediction_to_csv(image_hash_id, family_pred, family_conf, genus_pred, genus_conf, species_pred, species_conf, log_file):
    log_df = pd.read_csv(log_file)
    if image_hash_id in log_df['image_hash_id'].values:
        return  # If entry already exists, skip logging

    new_row = pd.DataFrame([{
        "image_hash_id": image_hash_id,
        "family_pred": family_pred,
        "family_conf": family_conf,
        "genus_pred": genus_pred,
        "genus_conf": genus_conf,
        "species_pred": species_pred,
        "species_conf": species_conf
    }])
    new_row.to_csv(log_file, mode='a', header=False, index=False)

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

# Load labels
family_labels, genus_labels, species_labels = create_labels_from_path(settings['labels_path'])

# Streamlit layout
st.set_page_config(page_title="CryptoVision", page_icon="🐟", layout="wide")
st.image(settings['image_header'], use_column_width=True)
st.title("CryptoVision Image Classifier 🐟🔍")
st.markdown(
    """
    **CryptoVision** is a deep learning model designed to classify fish species from images. 
    Upload an image of a fish, click "Classify," and the model will predict its family, genus, and species. 
    """
)

st.info("For research and enhancement purposes, your image will be saved in our secure database.")

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image preview and classify button only after image upload
if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image Preview", use_column_width=True)
    
    # Save the uploaded image with a unique hash ID
    saved_image_path, image_hash_id = save_image_with_hash(img, settings['saved_images_folder'])
    
    # Centralize the "Classify" button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        classify_button = st.button("Classify", use_container_width=True)

    # Perform prediction and display results if the "Classify" button is clicked
    if classify_button:
        
        # Perform the prediction
        top_k_family, top_k_genus, top_k_species = predict_image(
            img, cv_model, family_labels, genus_labels, species_labels, image_size=settings['img_size']
        )

        # Extract top-1 predictions and their confidence for each level
        family, family_conf = top_k_family[0]
        genus, genus_conf = top_k_genus[0] if family_conf >= 0.4 else ("", 0)
        species, species_conf = top_k_species[0] if genus_conf >= 0.4 else ("", 0)

        # Log prediction details to CSV only on the first click
        log_prediction_to_csv(
            image_hash_id, family, family_conf, genus, genus_conf, species, species_conf, settings['log_file']
        )

        # Display predictions with progress bars and confidence checks
        st.subheader("Prediction Results")
        def display_prediction(label, prediction, confidence):
            confidence_percentage = int(confidence * 100)
            color = color_confidence(confidence)
            st.markdown(f"<div style='font-size:20px;'><b>{label}:</b> <i>{prediction}</i> "
                        f"<span style='color:{color};'>{confidence_percentage}%</span></div>", unsafe_allow_html=True)
            st.progress(confidence_percentage)

        # Display family prediction if confidence is >= 40%
        if family_conf >= settings['min_confidence']:
            display_prediction("👪 Family", family, family_conf)
        else:
            st.warning("The model could not confidently predict the family.")

        # Display genus prediction if confidence is >= 40% and family confidence is adequate
        if genus_conf >= settings['min_confidence']:
            display_prediction("🌱 Genus", genus, genus_conf)
        elif family_conf >= settings['min_confidence']:
            st.warning("The model could not confidently predict the genus.")

        # Display species prediction if confidence is >= 40% and genus confidence is adequate
        if species_conf >= settings['min_confidence']:
            display_prediction("🐟 Species", species, species_conf)
        elif genus_conf >= settings['min_confidence']:
            st.warning("The model could not confidently predict the species.")

        # Toggle area for top-5 predictions
        with st.expander("See top 5 predictions"):
            st.subheader("Top-5 Predictions")
            
            # Display top-5 tables with headers
            def format_confidence(predictions):
                return {"Label": [label for label, _ in predictions], "Confidence": [f"{confidence * 100:.2f}%" for _, confidence in predictions]}

            st.write("**Family Predictions:**")
            st.table(format_confidence(top_k_family))
            
            st.write("**Genus Predictions:**")
            st.table(format_confidence(top_k_genus))

            st.write("**Species Predictions:**")
            st.table(format_confidence(top_k_species))

        # Feedback section
        st.subheader("Feedback")
        st.write("Was the prediction correct in your opinion?")
        user_feedback = st.radio("", ["Yes", "No"], index=0, key="user_feedback")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")