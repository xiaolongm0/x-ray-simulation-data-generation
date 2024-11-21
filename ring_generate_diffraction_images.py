import numpy as np
import matplotlib.pyplot as plt
import random

def create_ring_image(canvas_image, center_x, center_y, outer_radius, ring_width, pixel_value):
    y, x = np.ogrid[-center_y:2048-center_y, -center_x:2048-center_x]
    outer_mask = x**2 + y**2 <= outer_radius**2
    inner_mask = x**2 + y**2 <= (outer_radius - ring_width)**2
    canvas_image[outer_mask] = pixel_value  # Set the pixels within the outer radius to 1
    canvas_image[inner_mask] = 0  # Reset the pixels within the inner radius to 0 (creating the ring)
    return canvas_image

def main():
    canvas_image = np.zeros((2048, 2048))

    # Parameters for the ring
    center_x, center_y = 1024, 1024  # Center of the ring
    outer_radius = 500  # Outer radius of the ring
    ring_width = 5  # Width of the ring

    outer_radiuss = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    for i in range(10):
        outer_radius = outer_radiuss[i]
        pixel_value = random.randint(0, 255)/255
        ring_image = create_ring_image(canvas_image, center_x, center_y, outer_radius, ring_width, pixel_value)

    plt.imshow(ring_image, cmap='gray')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

# main function
if __name__ == "__main__":
    main()

