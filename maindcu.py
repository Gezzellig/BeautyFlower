import network
from preload import load_preload_images_batch

import sys
import time
import os

print_summaries = True

#PARAMETERS:
# Epoch = Amount of times the whole dataset is trained
EPOCHS          = 5
INPUT_FOLDER    = "data/learnset"
batch_size 	= 2
AMOUNT_BATCHES  = 10 / batch_size

print("Building the gan")

gan = network.BeautyFlower()

if print_summaries:
	print ("epochs: " + str(EPOCHS))
	print ("batch_size: " + str(batch_size))
	print ("AMOUNT_BATCHES: " + str(AMOUNT_BATCHES))

	print("generator summary:")
	gan.generator.summary()
	print("discriminator summary:")
	gan.discriminator.summary()
	print("combined MODELS")
	gan.combined_model.summary()

#gan.load_weights('dcu_test1')
foldername = time.strftime("%d_%b_%Y_%H:%M:%S", time.gmtime());
print("output is stored in: {}".format(foldername))
os.mkdir("trained_networks/{}".format(foldername))

sys.stdout.flush()

for epoch in range(EPOCHS):
	print( "epoch " + str(epoch) )
	sys.stdout.flush()
	for batch_idx in range (int(AMOUNT_BATCHES)):
		hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=batch_size, batch_number=batch_idx)
		gan.train(bicubic, hr_images, batch_size)



	if epoch % 1 == 0:
		filename = "{}/gan{}e{}".format(foldername, time.strftime("%d_%b_%Y_%H:%M:%S", time.gmtime()), epoch)
		gan.store_weights(filename)
		print("stored: {}".format(filename))
