import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def load_image_and_display_image_info(image_path):
    image = Image.open(image_path)
    # Display image information and show the image
    image_info = {
        "format": image.format,
        "mode": image.mode,
        "size": image.size
    }
    print(image_info)
    return image

def visualize_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def apply_threshold(image_array, low_threshold, high_threshold):
    '''
    This function is used to clip the image array by the low and high threshold.
    '''
    # check if image array is numpy array
    if not isinstance(image_array, np.ndarray):
        raise ValueError("image_array must be a numpy array")
    
    img = np.clip(image_array, low_threshold, high_threshold)
    return img

data_folder = '/Users/xiaolongm/Desktop/real_diffraction_images/xray-images'
sub_folders = os.listdir(data_folder)
print(sub_folders)

fig, axes = plt.subplots(8, 5, figsize=(10, 10))

sub_folder_idx = 0
for sub_folder in sub_folders:
    if 'Tian' in sub_folder:
        path = os.path.join(data_folder, sub_folder)
        files = os.listdir(path)
        file_idx = 0
        for file in files:
            if file.endswith('.tif'):
                full_path = os.path.join(path, file)
                print(full_path)
                image = load_image_and_display_image_info(full_path)
                image_array = np.array(image)
                image_array_clipped = apply_threshold(image_array, low_threshold= 0, high_threshold= 64000)
                axes[sub_folder_idx, file_idx].imshow(image_array_clipped, cmap='jet')
                axes[sub_folder_idx, file_idx].axis('off')
                # add title
                title = file.replace('-', '_')
                title = title.split('_')[0]
                # put the title in the left
                axes[sub_folder_idx, file_idx].set_title(title, fontsize=4, fontweight='bold')
                file_idx += 1
        sub_folder_idx += 1

plt.show()

