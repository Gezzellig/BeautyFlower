"""
Main program
- Loads the data
- Runs the network
"""

import network  as nw
import dataset  as ds
import numpy    as np
from skimage.transform import rescale, resize, downscale_local_mean
import random
import matplotlib.pyplot as plt

# Load data
data = ds.getData([2], show_info=True)

# Show a random example image
ds.showImageFromData(data, random.randint(0, len(data) - 1), seconds=0.5)
ds.showImageFromData(data, 0)

# Instanciate the network
gan = nw.BeautyFlower()

# Convert to 3 channel image shape
img = ds.toImage(data[0])

# Decrease the size 
imgLowRes = rescale(img, 1.0 / 2.0)

# Image needs to be packed into another array, because the net expects multiple images after each other.
# So for predicting just one image, we have to make it a [1, 32, 32, 3] array otherwise it won't work.
input_data = np.array([imgLowRes])

# Get predicted image back
output = gan.generateImage(input_data)

# Output shape will also be an array of arrays, so for just predicting one image, we take the first element of the output
print (output[0].shape)

# Plot the image using matplotlib
title = "Prediction"
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
ax.imshow(output[0],interpolation='none')
ax.set_title(title ,fontsize =15)
plt.show()