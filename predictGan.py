import network
import sys
import imageio
import matplotlib.pyplot as plt
import numpy as np

gan_file_name = sys.argv[1]
image_file_name = sys.argv[2]
image = imageio.imread(image_file_name)
image_batch = np.array([image])
gan = network.BeautyFlower()
gan.load_weights(gan_file_name)
predicted_batch = gan.generator.predict(image_batch, batch_size=1, verbose=1, steps=None)
plt.figure()
print(predicted_batch)
plt.imshow(predicted_batch[0])
plt.figure()
plt.imshow(image)
plt.show()