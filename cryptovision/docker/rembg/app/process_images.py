import os
from rembg import remove
from PIL import Image
import argparse
from tqdm import tqdm
import cv2
import numpy as np

def detect_background(image, threshold=200, border_size=10):
    """
    Detect if the background of an image is predominantly white or black.

    Args:
    - image (PIL.Image): Opened Pillow image.
    - threshold (int): Pixel intensity threshold (above = white, below = black).
    - border_size (int): Width of the border to sample pixels from.

    Returns:
    - str: "white" or "black" depending on the background.
    """
    # Convert Pillow image to NumPy array
    image_np = np.array(image)

    # Convert to grayscale if the image is not already
    if len(image_np.shape) == 3:  # If RGB or RGBA
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np  # Already grayscale

    # Extract border regions and flatten them
    top_border = gray[:border_size, :].flatten()  # Flatten top
    bottom_border = gray[-border_size:, :].flatten()  # Flatten bottom
    left_border = gray[:, :border_size].flatten()  # Flatten left
    right_border = gray[:, -border_size:].flatten()  # Flatten right
    
    # Concatenate all border pixels
    border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    
    # Count white and black pixels
    white_pixels = np.sum(border_pixels > threshold)
    black_pixels = np.sum(border_pixels <= threshold)
    
    # Decision
    return "white" if white_pixels > black_pixels else "black"

def process_folder(input_folder, output_folder):
    # Collect all image files from subdirectories
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.'):
                image_files.append(os.path.join(root, file))

    # Initialize progress bar
    with tqdm(total=len(image_files), desc="Processing Images", unit="img") as pbar:
        for input_path in image_files:
            try:
                # Get relative path and prepare output path
                relative_path = os.path.relpath(os.path.dirname(input_path), input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Load and resize the image (reduce by x4)
                with Image.open(input_path) as img:
                    width, height = img.size
                    #img_resized = img.resize((width // 4, height // 4))
                    
                    # Detect background color using the already opened image
                    bgcolor = detect_background(img)
                    
                    # Remove background with detected color
                    if bgcolor == "white":
                        img_rmbg = remove(img, bgcolor=(255, 255, 255, 255))
                    else:
                        img_rmbg = remove(img, bgcolor=(0, 0, 0, 255))
                    
                    # Convert to RGB if needed (JPEG does not support RGBA)
                    if img_rmbg.mode == 'RGBA':
                        img_rmbg = img_rmbg.convert('RGB')

                    # Save as JPEG
                    output_file_path = os.path.join(output_subfolder, f"{os.path.splitext(os.path.basename(input_path))[0]}.jpg")
                    img_rmbg.save(output_file_path, format='JPEG')
                    
                    # Update progress bar
                    pbar.set_postfix(file=os.path.basename(input_path))
                    pbar.update(1)

            except Exception as e:
                pbar.write(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch background removal from images.")
    parser.add_argument("--input", required=True, help="Path to input folder containing images.")
    parser.add_argument("--output", required=True, help="Path to output folder to save processed images.")
    args = parser.parse_args()

    process_folder(args.input, args.output)
