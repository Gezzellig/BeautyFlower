import network
from tqdm import tqdm
from preload import load_preload_images_batch

gan = network.BeautyFlower()
print("generator summary:")
gan.generator.summary()
input()
print("discriminator summary:")
gan.discriminator.summary()

# Amount of times the whole dataset is trained
EPOCHS       = 100
INPUT_FOLDER = "data/learnset"

for epoch in tqdm(range(EPOCHS)):
    for batch_idx in range (20):
        hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=100, batch_number=batch_idx)
        gan.train(lr_images, hr_images)