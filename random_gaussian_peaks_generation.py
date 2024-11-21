import numpy as np
import matplotlib.pyplot as plt
import random

def add_gaussian_peak(canvas_image, center_x, center_y, amplitude, sigma_x, sigma_y):
    """
    Add a Gaussian peak to the canvas image.
    
    Parameters:
        canvas_image (ndarray): The canvas image.
        center_x (float): The x-coordinate of the Gaussian peak's center.
        center_y (float): The y-coordinate of the Gaussian peak's center.
        amplitude (float): The amplitude of the Gaussian peak.
        sigma_x (float): The standard deviation in the x direction.
        sigma_y (float): The standard deviation in the y direction.
    """
    y, x = np.indices(canvas_image.shape)
    gaussian = amplitude * np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) +
                                     ((y - center_y) ** 2) / (2 * sigma_y ** 2)))
    canvas_image += gaussian
    return canvas_image

def main():
    # Canvas size
    canvas_size = 2048
    canvas_image = np.zeros((canvas_size, canvas_size))

    # Randomly generate Gaussian peak parameters
    num_peaks = 50  # Number of Gaussian peaks
    for _ in range(num_peaks):
        center_x = random.uniform(0, canvas_size)  # Random x-coordinate for the center
        center_y = random.uniform(0, canvas_size)  # Random y-coordinate for the center
        amplitude = random.uniform(0.5, 1.0)  # Random amplitude
        sigma_x = random.uniform(10, 100)  # Random standard deviation in x direction
        sigma_y = random.uniform(10, 100)  # Random standard deviation in y direction
        canvas_image = add_gaussian_peak(canvas_image, center_x, center_y, amplitude, sigma_x, sigma_y)

    # Display the result
    plt.imshow(canvas_image, cmap='hot')
    plt.colorbar(label="Intensity")
    plt.axis('off')  # Turn off the axis
    plt.title("Random Gaussian Peaks")
    plt.show()

# main function
if __name__ == "__main__":
    main()
