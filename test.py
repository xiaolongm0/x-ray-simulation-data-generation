import numpy as np

# load img npy files

img = np.load('img_0.npy')

# plot the image

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
