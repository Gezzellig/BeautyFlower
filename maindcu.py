import network
from preload import load_preload_images
import numpy as np

import sys
import time
import os

print ('Argument List:', str(sys.argv))

if len (sys.argv) == 5:
	EPOCHS = int(sys.argv[1])
	SAVE_INTERVAL = int(sys.argv[2])
	batch_size = int(sys.argv[3])
	DO_GEN_PRETRAIN = int(sys.argv[4])
else:
	print ("""Incorrect number of input arguments, please give input arguments in the following format:\n 
		python3 maindcu.py EPOCHS SAVE_INTERVAL BATCH_SIZE GEN_PRETRAIN""")
	exit()

print_summaries = True



print("Building the gan")

gan = network.BeautyFlower()

#PARAMETERS:
INPUT_FOLDER    = "data/learnset"
AMOUNT_BATCHES  = 5000 / batch_size

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

hr_images, lr_images, bicubic = load_preload_images(INPUT_FOLDER)
hr_images_batched = np.split(hr_images, AMOUNT_BATCHES)
lr_images_batched = np.split(lr_images, AMOUNT_BATCHES)
bicubic_batched = np.split(bicubic, AMOUNT_BATCHES)

print("Images loaded, time to learn!!")



for epoch in range(EPOCHS):
	print( "epoch " + str(epoch) )
	sys.stdout.flush()
	for batch_idx in range (int(AMOUNT_BATCHES)):	
		if DO_GEN_PRETRAIN == 0:
			gan.train(bicubic_batched[batch_idx], hr_images_batched[batch_idx], batch_size)
		else:
			gan.pretrain_generator_only(bicubic_batched[batch_idx], hr_images_batched[batch_idx])



	if epoch % SAVE_INTERVAL == 0:
		filename = "{}/gan{}e{}".format(foldername, time.strftime("%d_%b_%Y_%H:%M:%S", time.gmtime()), epoch)
		gan.store_weights(filename)
		print("stored: {}".format(filename))
