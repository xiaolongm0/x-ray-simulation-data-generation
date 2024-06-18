'''
Using sliding window method to sample 1024*1024 images to fit the SAM input
'''

from PIL import Image
import os

def slide_and_save(image_path, window_size=(1024, 1024), stride=(400, 100), save_dir='sample_images'):
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate the number of sliding windows
    num_x = (width - window_size[0]) // stride[0] + 1
    num_y = (height - window_size[1]) // stride[1] + 1

    # Iterate over all window positions
    for i in range(num_x):
        for j in range(num_y):
            left = i * stride[0]
            upper = j * stride[1]
            right = left + window_size[0]
            lower = upper + window_size[1]

            # Extract the small image
            cropped_image = image.crop((left, upper, right, lower))

            # Save the small image
            cropped_image.save(os.path.join(save_dir, f'sample_{i}_{j}.png'))

# Example usage
slide_and_save('output_mask.png')
