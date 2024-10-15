import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# list all files in the data folder
data_folder = '/Users/xiaolongm/Documents/GitHub/x-ray-simulation-data-generation/images/real_images/npy_files'
files = os.listdir(data_folder)
print(files)

for file in files:

    # load npy file from images folder
    gray = np.load(f'./images/real_images/npy_files/{file}')

    # normalize the gray image
    gray = normalize_image(gray)

    visualize_image(gray)

    # show the gray image and save it
    cv2.imshow('Gray Image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # get image size and center point
    h, w = gray.shape
    center_x, center_y = w // 2, h // 2

    # polar transformation
    max_radius = np.sqrt(center_x**2 + center_y**2)
    polar_image = cv2.linearPolar(gray, (center_x, center_y), max_radius, cv2.WARP_FILL_OUTLIERS)

    # convert to line (adjust y-axis range to expand)
    polar_image = np.roll(polar_image, w // 2, axis=1)

    # show the gray image and save it
    cv2.imshow('Gray Image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the gray image
    #cv2.imwrite('./images/real_images/gray_images/BYS-400-00000-00001.png', gray)

    # get image size and center point
    h, w = gray.shape
    center_x, center_y = w // 2, h // 2

    # shape of polar image
    print(polar_image.shape)

    # display image
    cv2.imshow('Polar Image', polar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()