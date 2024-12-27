import os
import shutil
import random
import streamlit as st
from PIL import Image
from cryptovision.tools import CryptoVisionAI

# Directories and AI Model Setup
UNKNOWN_DIR = '/Volumes/T7_shield/CryptoVision/Data/fish&functions_lab/cryptovision_reviewed/unknown'
RECLASS_PATH = '/Volumes/T7_shield/CryptoVision/Data/fish&functions_lab/cryptovision_reviewed/species'
DATA_DIR = '/Users/leonardo/Documents/Projects/cryptovision/data/processed/cv_images_dataset'
MODEL_PATH = '/Users/leonardo/Documents/Projects/cryptovision/models/phorcys_v09_hacpl_rn50v2_v2411251155.keras'

# Initialize AI Model
names = {"family": [], "genus": [], "species": []}
for folder in os.listdir(DATA_DIR):
    if folder == ".DS_Store":
        continue
    family, genus, species = folder.split("_")
    names["family"].append(family)
    names["genus"].append(genus)
    names["species"].append(f"{genus} {species}")

ai = CryptoVisionAI(
    MODEL_PATH,
    sorted(set(names['family'])),
    sorted(set(names['genus'])),
    sorted(set(names['species']))
)

# Streamlit App
st.title("CryptoVision Image Reclassifier 🐟")

# Sidebar for Input Paths
with st.sidebar:
    st.header("Settings")
    folder_path = st.text_input("🔍 Image Folder Path:")
    suffix_options = ["_tocrop", "_poor", "_small", "none"]
    suffix = st.selectbox("Image Suffix:", suffix_options, index=suffix_options.index("none"))

# Initialize session state
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'images' not in st.session_state or folder_path != st.session_state.get('last_folder', None):
    if folder_path and os.path.exists(folder_path):
        st.session_state.images = [
            f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg')) and f != ".DS_Store"
        ]
        st.session_state.last_folder = folder_path
        st.session_state.image_index = 0
    else:
        st.session_state.images = []

num_images = len(st.session_state.images)

# Reset reclassification path for each image
def reset_reclassification_path():
    if 'reclassify_path' in st.session_state:
        del st.session_state.reclassify_path

# If there are no images, show a message
if folder_path and not os.path.exists(folder_path):
    st.error("The specified folder path does not exist. Please enter a valid path.")
elif folder_path and num_images == 0:
    st.write("No images found in this folder.")
elif num_images > 0 and st.session_state.image_index < num_images:
    image = st.session_state.images[st.session_state.image_index]
    img_path = os.path.join(folder_path, image)
    img = Image.open(img_path)
    st.image(img, caption=f"Reviewing: {image}", use_column_width=True)

    # Perform Classification
    fam_pred, gen_pred, spec_pred = ai.predict(img_path)
    confidence_levels = ai.confidence

    # Top-1 Classification
    family, genus, species = fam_pred, gen_pred, spec_pred
    species = species.split()[-1]  # Extract species from full name if needed

    confidence_percent_family = round(confidence_levels[0] * 100, 2)
    confidence_percent_genus = round(confidence_levels[1] * 100, 2)
    confidence_percent_species = round(confidence_levels[2] * 100, 2)

    classification = f"{family}_{genus}_{species}"
    suggested_path = os.path.join(RECLASS_PATH, classification)

    # Reset path when a new image is loaded
    if 'reclassify_path' not in st.session_state or st.session_state.current_image != image:
        st.session_state.reclassify_path = suggested_path
        st.session_state.current_image = image

    reclassify_path = st.text_input("📂 Reclassification Path:", st.session_state.reclassify_path)

    # Display Prediction and Confidence
    st.write(f"**Prediction:** {family} > {genus} > {species}")
    st.write(f"**Confidence:** {confidence_percent_family}% > {confidence_percent_genus}% > {confidence_percent_species}%")

    # Button Actions
    col1, col2, col3 = st.columns(3)

    def next_image():
        del st.session_state.images[st.session_state.image_index]
        reset_reclassification_path()
        if len(st.session_state.images) == 0:
            st.success("All images reviewed!")
        else:
            st.session_state.image_index = 0

    def reclassify_action():
        target_path = os.path.join(
            reclassify_path,
            f"{os.path.splitext(image)[0]}{'' if suffix == 'none' else suffix}{os.path.splitext(image)[1]}"
        )
        os.makedirs(reclassify_path, exist_ok=True)
        shutil.move(img_path, target_path)
        st.success(f"Moved to: {target_path}")
        next_image()

    def unknown_action():
        target_path = os.path.join(UNKNOWN_DIR, image)
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        shutil.move(img_path, target_path)
        st.warning(f"Moved to Unknown Folder.")
        next_image()

    with col1:
        st.button("✅ Reclassify", on_click=reclassify_action)
    with col2:
        st.button("⏭️ Skip", on_click=next_image)
    with col3:
        st.button("❓ Unknown", on_click=unknown_action)

    # Image Counter
    st.write(f"Image {st.session_state.image_index + 1} of {num_images}")

else:
    if not folder_path:
        st.warning("Please enter a valid folder path.")
    else:
        st.success("All images have been processed. 🎉")

