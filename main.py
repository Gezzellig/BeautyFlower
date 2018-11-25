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
from tqdm import tqdm               # Progress indicator

# Load data
data = ds.getData([2], show_info=True)

# Show a random example image
ds.showImageFromData(data, random.randint(0, len(data) - 1), seconds=0.5)

# Instanciate the network
gan = nw.BeautyFlower(n_residual_blocks=6)

images = data[0:150]
images = [ds.toImage(img) for img in images]
imgLowRes = [rescale(img, 0.5) for img in images]
inputLR = np.array(imgLowRes)
inputHR = np.array(images)

test_image = [data[2301]]
test_image = [ds.toImage(img) for img in test_image]
test_imgLowRes = [rescale(img, 0.5) for img in test_image]
test_inputLR = np.array(imgLowRes)
test_inputHR = np.array(test_image)

# Show low res image
title = "Original Low-res image"
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
ax.imshow(test_imgLowRes[0],interpolation='none')
ax.set_title(title ,fontsize =15)
plt.show()

for x in tqdm(range(1000), desc=" Training generator"):

    gan.trainGenerator(inputLR, inputHR)

    # Get predicted image back
    output = gan.generateImage(np.array([test_imgLowRes[0]]))

    # Plot the image using matplotlib
    if (x % 1000 == 999):
        title = "Prediction"
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.imshow(output[0],interpolation='none')
        ax.set_title(title ,fontsize =15)
        plt.show()