import os
import shutil
from pathlib import Path
from cryptovision.tools import CryptoVisionAI, get_taxonomic_mappings_from_folders

def organize_images_by_predictions(
    ai: CryptoVisionAI,
    image_directory: str,
    output_directory: str,
    confidence_thresholds: dict = {"family": 0.5, "genus": 0.6, "species": 0.7},
    taxonomy_directory: str = None
):
    """
    Organize images into directories based on model predictions and confidence levels.

    Parameters:
        ai (CryptoVisionAI): An instance of the CryptoVisionAI class.
        image_directory (str): Path to the directory containing images.
        output_directory (str): Path to save organized images.
        confidence_thresholds (dict): Minimum confidence scores for family, genus, and species predictions.
        taxonomy_directory (str): Path to a directory with subfolders named as 'family_genus_species' to generate taxonomy mappings.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Prepare "unknown" and "genus_only" directories
    unknown_dir = os.path.join(output_directory, "unknown")
    genus_only_dir = os.path.join(output_directory, "genus_only")
    os.makedirs(unknown_dir, exist_ok=True)
    os.makedirs(genus_only_dir, exist_ok=True)

    # Generate taxonomy mappings dynamically if taxonomy_directory is provided
    if taxonomy_directory:
        _, _, _, genus_to_family, species_to_genus = get_taxonomic_mappings_from_folders(taxonomy_directory)
    else:
        genus_to_family, species_to_genus = {}, {}

    # Process each image in the input directory
    for root, _, files in os.walk(image_directory):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")) or file.startswith("."):
                continue

            image_path = os.path.join(root, file)

            # Predict family, genus, and species using CryptoVisionAI
            prediction = ai.predict(image_path)
            family_name, genus_name, species_name = prediction

            # Retrieve confidence scores
            family_conf, genus_conf, species_conf = ai.confidence

            # Determine the target directory and filename based on confidence
            if species_conf >= confidence_thresholds["species"]:
                # Species-level classification
                target_dir = os.path.join(output_directory, family_name, genus_name, species_name)
                os.makedirs(target_dir, exist_ok=True)
                base_name, ext = os.path.splitext(file)
                new_name = f"{base_name}_{species_name}{ext}"
            elif genus_conf >= confidence_thresholds["genus"]:
                # Genus-level classification
                if family_name is None:
                    # If family is invalid, use genus_only directory
                    target_dir = os.path.join(genus_only_dir, genus_name)
                else:
                    target_dir = os.path.join(output_directory, family_name, genus_name)
                os.makedirs(target_dir, exist_ok=True)
                base_name, ext = os.path.splitext(file)
                new_name = f"{base_name}_{genus_name}{ext}"
            elif family_conf >= confidence_thresholds["family"]:
                # Family-level classification
                target_dir = os.path.join(output_directory, family_name)
                os.makedirs(target_dir, exist_ok=True)
                base_name, ext = os.path.splitext(file)
                new_name = f"{base_name}_{family_name}{ext}"
            else:
                # Unknown classification
                target_dir = unknown_dir
                new_name = file

            target_path = os.path.join(target_dir, new_name)

            # Copy the image to the target directory with the new name
            shutil.copy(image_path, target_path)

    print(f"Images organized in {output_directory}")

if __name__ == "__main__":
    
    # Initialize the CryptoVisionAI instance
    model_path = "/Users/leonardo/Documents/Projects/cryptovision/models/2024.11/phorcys_v09_hacpl_rn50v2_v2411251155.keras"
    taxonomy_dir = "/Users/leonardo/Documents/Projects/cryptovision/data/processed/cv_images_dataset"
    
    family_labels, genus_labels, species_labels, _, _ = get_taxonomic_mappings_from_folders(taxonomy_dir)
    
    ai = CryptoVisionAI(
        model_path=model_path,
        family_names=family_labels,
        genus_names=genus_labels,
        species_names=species_labels,
    )

    # Organize images
    organize_images_by_predictions(
        ai=ai,
        image_directory="/Volumes/T7_shield/CryptoVision/Data/others/hemingson_photos/selected",
        output_directory="/Volumes/T7_shield/CryptoVision/Data/others/hemingson_photos/cv_organized",
        confidence_thresholds={"family": 0.6, "genus": 0.7, "species": 0.8},
        taxonomy_directory=taxonomy_dir
    )