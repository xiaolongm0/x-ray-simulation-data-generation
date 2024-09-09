'''
Using sliding window method to sample 1024*1024 images to fit the SAM input
'''

import os
import argparse
from PIL import Image

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

# define main function
def main():
    parser = argparse.ArgumentParser(description='get image tiles')
    parser.add_argument('--image', type=str, default='output_mask.png', help='Path to the input image')
    parser.add_argument('--save_dir', type=str, default='sample_images', help='Directory to save the sample images')
    parser.add_argument('--window_size', type=int, default=(1024, 1024), help='Size of the sliding window')
    parser.add_argument('--stride', type=int, default=(400, 100), help='Stride of the sliding window')
    args = parser.parse_args()
    slide_and_save(args.image, window_size=args.window_size, stride=args.stride, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
