import network
from preload import load_preload_images_batch

print_summaries = True

#PARAMETERS:
# Epoch = Amount of times the whole dataset is trained
EPOCHS          = 20
INPUT_FOLDER    = "data/learnset"
batch_size 		= 25
AMOUNT_BATCHES  = 5000 / batch_size

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

for epoch in range(EPOCHS):
	print( "epoch " + str(epoch) )
	for batch_idx in range (int(AMOUNT_BATCHES)):
		hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=batch_size, batch_number=batch_idx)
		gan.train(bicubic, hr_images, batch_size)

#	if epoch % 250 == 0:
# 		gan.store_weights('dcu_test1')
