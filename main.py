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
ds.showImageFromData(data, 2)

# Instanciate the network
gan = nw.BeautyFlower()

for x in tqdm(range(1000), desc=" Training generator"):
    images = data[0:20]
    images = [ds.toImage(img) for img in images]
    imgLowRes = [rescale(img, 0.5) for img in images]
    inputLR = np.array(imgLowRes)
    inputHR = np.array(images)

    gan.trainGenerator(inputLR, inputHR)

    # Get predicted image back
    output = gan.generateImage(np.array([imgLowRes[2]]))

    # Plot the image using matplotlib
    if (x % 100 == 0):
        title = "Prediction"
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ax.imshow(output[0],interpolation='none')
        ax.set_title(title ,fontsize =15)
        plt.show()