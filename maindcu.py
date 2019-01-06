import network
from tqdm import tqdm
import matplotlib.pyplot as plt
from preload import load_preload_images_batch
import matplotlib.pyplot as plt
import time

print_summaries = False

#PARAMETERS:
# Epoch = Amount of times the whole dataset is trained
EPOCHS       = 100
INPUT_FOLDER = "data/learnset"
batch_size = 1



gan = network.BeautyFlower()

if print_summaries:
	print("generator summary:")
	gan.generator.summary()
	input()
	print("discriminator summary:")
	gan.discriminator.summary()
	input()
	print("combined MODELS")
	gan.combined_model.summary()


gan.load_weights('dcu_test1')


t = time.time()
for epoch in range(EPOCHS):
	print( "epoch " + str(epoch) )
	for batch_idx in tqdm(range (20)):
		hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=batch_size, batch_number=batch_idx)
		gan.train(bicubic, hr_images, batch_size)

	if epoch % 25 == 0:
		hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=1, batch_number=1)

		predicted = gan.generator.predict(bicubic)
		plt.figure(1)
		plt.imshow(bicubic[0])
		plt.show(block=False)
		plt.pause(0.01)

		plt.figure(2)
		plt.imshow(predicted[0])
		plt.show(block=False)
		plt.pause(0.01)
		gan.store_weights('dcu_test1')

print ("Finished with time: " + str(time.time() - t))
