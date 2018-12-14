import network
from tqdm import tqdm
import matplotlib.pyplot as plt
from preload import load_preload_images_batch

gan = network.BeautyFlower()
print("generator summary:")
gan.generator.summary()
print("discriminator summary:")
gan.discriminator.summary()

# Amount of times the whole dataset is trained
GEN_EPOCHS   = 3
EPOCHS       = 1
INPUT_FOLDER = "data/learnset"

#gan.load_weights('dcu')

for epoch in range(GEN_EPOCHS):
    for batch_idx in tqdm(range(20)):
        hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=10, batch_number=batch_idx)
        gan.train_generator(lr_images, hr_images)

# for epoch in range(EPOCHS):
#     for batch_idx in tqdm(range (5)):
#         hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=1, batch_number=batch_idx)
#         gan.train(lr_images, hr_images)

# Save network
gan.store_weights('dcu')

# Show result
hr_images, lr_images, bicubic = load_preload_images_batch(INPUT_FOLDER, batch_size=5, batch_number=0)
result = gan.generate_image(lr_images)

plt.figure(1)
plt.subplot(211)
plt.imshow(hr_images[0])

plt.subplot(212)
plt.imshow(result[0])
plt.show()