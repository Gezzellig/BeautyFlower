import network
from tqdm import tqdm
from preload import load_preload_images_batch
import matplotlib.pyplot as plt


gan = network.BeautyFlower()

print("generator summary:")
gan.generator.summary()
input()

print("discriminator summary:")
gan.discriminator.summary()

# Amount of times the whole dataset is trained
EPOCHS       = 1
INPUT_FOLDER = "data/learnset"

batch_size = 2

for epoch in range(EPOCHS):
    for batch_idx in tqdm(range (20)):
        hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=batch_size, batch_number=batch_idx)
        gan.train(bicubic, hr_images, batch_size)