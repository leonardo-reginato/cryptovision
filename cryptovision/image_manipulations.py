import os
from PIL import Image

def copy_and_resize_image(image_path, target_directory):
    """
    Copy an image to a target directory, convert it to JPEG format, 
    and reduce its size to one-fourth of its original dimensions.

    Parameters:
        image_path (str): Path to the source image.
        target_directory (str): Path to the target directory.
    """
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Open the image using PIL
    with Image.open(image_path) as img:
        # Reduce the image dimensions to one-fourth
        #new_size = (img.width // 4, img.height // 4)
        #resized_img = img.resize(new_size, Image.ANTIALIAS)
        resized_img = img
        
        # Ensure the image is in RGB mode for JPEG
        if resized_img.mode != "RGB":
            resized_img = resized_img.convert("RGB")
        
        # Create the target file path with a .jpeg extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        target_path = os.path.join(target_directory, f"{base_name}.jpeg")
        
        # Save the resized image in JPEG format
        resized_img.save(target_path, format="JPEG", quality=85)
        
        print(f"Image saved to: {target_path}")



if __name__ == "__main__":
    
    source_directory = "/Volumes/T7_shield/CryptoVision/Data/others/hemingson_photos/original"
    target_directory = "/Volumes/T7_shield/CryptoVision/Data/others/hemingson_photos/resized"
    
    for image in os.listdir(source_directory):
        
        if image.startswith("."):
            continue
        
        image_path = os.path.join(source_directory, image)
        copy_and_resize_image(image_path, target_directory)
        
    print("Images copied and resized successfully.")
        
    pass