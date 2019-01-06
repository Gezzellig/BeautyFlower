import network
from tqdm import tqdm
import matplotlib.pyplot as plt
from preload import load_preload_images_batch
import matplotlib.pyplot as plt

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


#gan.load_weights('dcu_test1')

for epoch in range(EPOCHS):
    print( "epoch " + str(epoch) )
    for batch_idx in tqdm(range (20)):
        hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=batch_size, batch_number=batch_idx)
        gan.train(bicubic, hr_images, batch_size)


gan.store_weights('dcu_test1')

