import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rembg import remove

def is_image_file(file_name:str, valid_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
    return file_name.lower().endswith(valid_extensions) and not file_name.startswith('.')

def augment_image(
    image_path: str, 
    num_variations: int, 
    background_dir: str = None, 
    rotation_range=45, 
    width_shift_range=0.25, 
    height_shift_range=0.25, 
    shear_range=0.25, 
    zoom_range=(0.75, 1.35), 
    horizontal_flip=True,
    vertical_flip=False, 
    brightness_range=(0.7, 1.3), 
    fill_mode='nearest', 
    bg_choices=('black', 'white', 'image'), 
    bg_probabilities=(0.4, 0.35, 0.25)
):
    image = Image.open(image_path)
    image = remove(image)
    
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_range=brightness_range,
        fill_mode=fill_mode
    )

    image_array = np.array(image).reshape((1,) + np.array(image).shape)  # Ensure correct shape

    augmented_images = []
    if background_dir:
        background_files = [f for f in os.listdir(background_dir) if is_image_file(f)]

    for i, batch in enumerate(datagen.flow(image_array, batch_size=1)):
        aug_image = Image.fromarray(batch[0].astype('uint8'))

        if background_dir:
            choice = random.choices(bg_choices, bg_probabilities)[0]
        else:
            choice = random.choice(('black', 'white'))

        if choice == 'black':
            background = Image.new('RGB', aug_image.size, (0, 0, 0))
        elif choice == 'white':
            background = Image.new('RGB', aug_image.size, (255, 255, 255))
        elif choice == 'image':
            bg_image_path = random.choice(background_files)
            background = Image.open(os.path.join(background_dir, bg_image_path)).resize(aug_image.size)

        background.paste(aug_image, (0, 0), aug_image)
        aug_image = background
        
        augmented_images.append(aug_image)
        if i + 1 >= num_variations:
            break
    
    return augmented_images
    

def augment_dataset(source_dir, target_dir, background_dir, num_aug_per_image=8, **kwargs):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for species_folder in tqdm(os.listdir(source_dir), desc="Processing species folders"):
        species_path = os.path.join(source_dir, species_folder)
        target_species_path = os.path.join(target_dir, species_folder)

        if not os.path.isdir(species_path):
            continue

        if not os.path.exists(target_species_path):
            os.makedirs(target_species_path)

        original_images = [f for f in os.listdir(species_path) if is_image_file(f)]

        for image_file in tqdm(original_images, desc=f"Augmenting {species_folder}", leave=False):
            image_path = os.path.join(species_path, image_file)
            augmented_images = augment_image(image_path, num_aug_per_image, background_dir, **kwargs)

            for idx, aug_image in enumerate(augmented_images):
                aug_image.save(os.path.join(target_species_path, f"{os.path.splitext(image_file)[0]}_aug_{idx}.png"))

augment_dataset(
    '/CryptoVision/data',
    '/CryptoVision/augmented_v2',
    '/CryptoVision/coral_reef_bg',
    num_aug_per_image=8
)