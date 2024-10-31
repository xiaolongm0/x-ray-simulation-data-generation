'''
This script is used to preprocess and visualize the images from raw diffraction tif data.
'''

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology

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

def extract_ring(image_array_clipped):
    '''
    This function use traditional edge detection methods to extract the ring from the image.
    '''
    edges = filters.sobel(image_array_clipped)
    threshold_value = filters.threshold_otsu(edges)
    binary_mask = edges > threshold_value
    binary_mask_closed = morphology.binary_closing(binary_mask, morphology.disk(3))
    labeled_image = measure.label(binary_mask_closed)
    regions = measure.regionprops(labeled_image)
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        ring_mask = labeled_image == largest_region.label
        extracted_ring = np.where(ring_mask, image_array_clipped, 0)
        return extracted_ring
    else:
        return None

def save_image(image_array, output_path):
    image = Image.fromarray(image_array)
    image.save(output_path)

def main():
    # list all files in the data folder
    data_folder = '/Users/xiaolongm/Desktop/real_diffraction_images/xray-images/TianyiOctB21_Fan'
    files = os.listdir(data_folder)
    print(files)

    # display all images in the data folder
    for file in files:
        image_path = os.path.join(data_folder, file)
        image = load_image_and_display_image_info(image_path)
        image_array = np.array(image)
        image_array_clipped = apply_threshold(image_array, low_threshold= 0, high_threshold= 64000)
        visualize_image(image_array_clipped)

        # get file name without extension
        file_name = os.path.splitext(file)[0]

        # save image array as numpy file
        output_path = os.path.join('./images', file_name)

        # save image array as numpy file
        np.save(f'{output_path}.npy', image_array_clipped)

        # get file name without extension
        #file_name = os.path.splitext(file)[0]

        #output_path = os.path.join('./images', file_name)
        #save_image(image_array_clipped, output_path=f'{output_path}.png') # save extracted ring as PNG file

        # extract ring
        #extracted_ring = extract_ring(image_array_clipped)
        #visualize_image(extracted_ring)
        #save_image(extracted_ring, output_path='./data/extracted_ring.png') # save extracted ring as PNG file

if __name__ == "__main__":
    main()
