import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_efficientnet
import json
import hashlib
import os
import pandas as pd
from datetime import datetime

# Load class names and model based on selected model type
def load_class_names_and_model(model_type):
    model_paths = {
        "Mobile_Net_V2": {
            "class_indices": "/CryptoVision/models/species/BV_label_MOBv2_S930_202407042018/class_indices.json",
            "model": "/CryptoVision/models/species/BV_label_MOBv2_S930_202407042018/model.h5",
            "preprocessing": preprocess_input_mobilenet,
        },
        "EfficientNet_V2_B0_911": {
            "class_indices": "/CryptoVision/models/species/BV_label_EFFv2B0_S911_202407170557/class_indices.json",
            "model": "/CryptoVision/models/species/BV_label_EFFv2B0_S911_202407170557/model.h5",
            "preprocessing": preprocess_input_efficientnet,
        },
        "EfficientNet_V2_B0_910": {
            "class_indices": "/CryptoVision/models/species/BV_label_EFFv2B0_S910_202407182355/class_indices.json",
            "model": "/CryptoVision/models/species/BV_label_EFFv2B0_S910_202407182355/model.h5",
            "preprocessing": preprocess_input_efficientnet,
        },
        "EfficientNet_V2_B0_896": {
            "class_indices": "/CryptoVision/models/species/BV_label_EFFV2B0_S896_202407250113/class_indices.json",
            "model": "/CryptoVision/models/species/BV_label_EFFV2B0_S896_202407250113/model.h5",
            "preprocessing": preprocess_input_efficientnet,
        },
        "EfficientNet_V2_B0_912": {
            "class_indices": "/CryptoVision/models/species/BV_label_EFFV2B0_S912_202407311711/class_indices.json",
            "model": "/CryptoVision/models/species/BV_label_EFFV2B0_S912_202407311711/model.h5",
            "preprocessing": preprocess_input_efficientnet,
        }
    }
    try:
        with open(model_paths[model_type]['class_indices'], 'r') as f:
            class_names = json.load(f)
        model = load_model(model_paths[model_type]['model'])
        prerocess_input = model_paths[model_type]['preprocessing']
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        class_names, model = {}, None
    return class_names, model, prerocess_input

# Split taxonomy string
def split_taxonomy(taxonomy_string):
    return taxonomy_string.split("_")

# Predict image
def predict_image(model, img, preprocess_input):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    probs = model.predict(img_array)
    top_prob = probs.max()
    top_pred = class_names[str(np.argmax(probs))]
    return top_prob, top_pred

# Generate hash ID for the image
def generate_hash_id(img):
    hasher = hashlib.sha256()
    img_byte_array = img.tobytes()
    hasher.update(img_byte_array)
    return hasher.hexdigest()

# Save image and log interaction
def save_interaction(img, top_pred, top_prob, user_guess):
    img_hash = generate_hash_id(img)
    img_save_path = os.path.join('saved_images', f'{img_hash}.png')
    img.save(img_save_path)

    log_path = 'log/interaction_log.csv'
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=['Date', 'Image Hash ID', 'Predicted Family', 'Predicted Genus', 'Predicted Species', 'Top Percentage', 'User Guess', 'User Feedback'])
    else:
        df = pd.read_csv(log_path)

    family, genus, species = split_taxonomy(top_pred)
    new_row = pd.DataFrame([{
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Image Hash ID': img_hash,
        'Predicted Family': family,
        'Predicted Genus': genus,
        'Predicted Species': species,
        'Top Percentage': round(top_prob * 100, 2),
        'User Guess': user_guess,
        'User Feedback': ''  # Initialize with an empty feedback
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(log_path, index=False)
    return img_hash

# Update user feedback in the CSV log
def update_user_feedback(img_hash, feedback):
    log_path = 'log/interaction_log.csv'
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df.loc[df['Image Hash ID'] == img_hash, 'User Feedback'] = feedback
        df.to_csv(log_path, index=False)

# Streamlit app layout
st.set_page_config(page_title="CryptoVision", page_icon="🐟")

# Add logo and title to the main page
logo_path = "/CryptoVision/images/logo_brandllab_v2.png"
logo = Image.open(logo_path)
col1, col2, col3 = st.columns([1, 1, 1])
col2.image(logo, use_column_width=False, width=200)

st.title("CryptoVision Image Classifier 🐟🔍")
st.markdown(
    """
    **CryptoVision** is a deep learning model designed to classify fish species from images. 
    Upload an image of a fish, and the model will predict its family, genus, and species. 
    Let's explore the underwater world together! 🌊🐠
    """
)

# Sidebar model selector
model_type = st.sidebar.selectbox(
    "Select the Model Type",
    ("Mobile_Net_V2", "EfficientNet_V2_B0_911", "EfficientNet_V2_B0_910", "EfficientNet_V2_B0_896","EfficientNet_V2_B0_912"),
)

# Load class names and model
class_names, model, preprocess = load_class_names_and_model(model_type)

# Reset session state if a new image is uploaded
def reset_state():
    st.session_state.prediction_made = False
    st.session_state.top_prob = 0
    st.session_state.top_pred = ""
    st.session_state.progress_percent = 0
    st.session_state.img = None
    st.session_state.user_guess = ""
    st.session_state.img_hash = ""
    st.session_state.user_feedback = ""

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], on_change=reset_state)

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.session_state.user_guess = st.text_input("What species do you think this is?", st.session_state.get('user_guess', ''))

    col1, col2, col3 = st.columns([1, 7, 1])
    if col2.button("Classify", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please check the model path and try again.")
        else:
            top_prob, top_pred = predict_image(model, img, preprocess)
            progress_percent = round(top_prob * 100)
            st.session_state.prediction_made = True
            st.session_state.top_prob = top_prob
            st.session_state.top_pred = top_pred
            st.session_state.progress_percent = progress_percent
            st.session_state.img = img
            st.session_state.img_hash = save_interaction(img, top_pred, top_prob, st.session_state.user_guess)

if st.session_state.get('prediction_made', False):
    progress_percent = st.session_state.progress_percent
    top_pred = st.session_state.top_pred
    top_prob = st.session_state.top_prob
    
    if progress_percent < 20:
        st.write(f"**Prediction Confidence:** {progress_percent}%")
        st.error("The model was unable to predict the image correctly. Please upload another photo. 😞")
    else:
        family, genus, species = split_taxonomy(top_pred)
        st.write(f"### **Fish Species:** :blue[{family} {genus} {species}]")
        
        if progress_percent < 40:
            st.write(f"#### **Confidence:** :red[{progress_percent}%]")
        elif progress_percent >= 40 and progress_percent < 60:
            st.write(f"#### **Confidence:** :orange[{progress_percent}%]")
        elif progress_percent >= 60 and progress_percent < 80:
            st.write(f"#### **Confidence:** :yellow[{progress_percent}%]")
        elif progress_percent >= 80:
            st.write(f"#### **Confidence:** :green[{progress_percent}%]")

        reference_image_path = f"/CryptoVision/images/species_reference_image/{top_pred}.png"
        if os.path.exists(reference_image_path):
            ref_img = Image.open(reference_image_path)
            st.image(ref_img, caption="Reference Image", use_column_width=True)

        st.write("#### Was the prediction correct in your opinion? **(:red[Required])**")
        user_feedback = st.radio("", ["Yes", "No"], index=0, key="user_feedback")
        
        col11, col22, col33 = st.columns([1, 7, 1])
        if col22.button("Submit Feedback", use_container_width=True):
            if user_feedback:
                update_user_feedback(st.session_state.img_hash, user_feedback)
                st.success("Feedback saved successfully! 🎉")
            else:
                st.error("Please provide feedback before submitting.")