from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from skimage import io

img = plt.imread('output.png')

#print(img.shape)
#plt.imshow(img)
#plt.show()

left = 300
top = 300
right = left + 512
bottom = top + 512

# sample a 512x512 image from the original image
sample_a_image = img[top:bottom, left:right]

io.imsave('sample_image_512.png', sample_a_image)

plt.imshow(sample_a_image)
#plt.imsave('sample_image_512.png', sample_a_image, cmap='gray', format='png')
plt.show()
