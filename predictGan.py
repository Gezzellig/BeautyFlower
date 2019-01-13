import network
import sys, os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

# gan_file_name = sys.argv[1]
# image_file_name = sys.argv[2]

models = os.listdir("trained_networks")
print("Available models: ")
for idx, model in enumerate(models): 
    print ("(" + str(idx) + ") " + model)
choice = input ("Enter choice: ")

chosen_model = models[int(choice)]
print (chosen_model)


submodels = os.listdir("trained_networks/" + chosen_model)
submodels = [model for model in submodels if model[-4] == "g"]
print ("Available loadable models for model " + chosen_model)
for idx, model in enumerate(submodels):
    print ("(" + str(idx) + ") " + model)
choice = input ("Enter choice: ")
chosen_path_model = chosen_model + "/" + submodels[int(choice)][:-5]

print ("Loading model: " + chosen_path_model)

gan = network.BeautyFlower()
gan.load_weights(chosen_path_model)
while(True):
    print ("Predict on which image? (0000 - 4996)\nEnter -1 to exit.")
    chosen_image = input ()
    if int(chosen_image) == -1:
        exit()
    image_bc = imageio.imread("data/learnset/image" + str(chosen_image) + "/bicubic.png")
    image_or = imageio.imread("data/learnset/image" + str(chosen_image) + "/original.png")
    image_batch = np.array([image_bc])

    predicted_batch = gan.generator.predict(image_batch, batch_size=1, verbose=1, steps=None)
    print("original discrimator: {}".format(gan.discriminator.predict(image_batch, batch_size=1, verbose=1, steps=None)))
    print("prediction discrimator: {}".format(gan.discriminator.predict(predicted_batch, batch_size=1, verbose=1, steps=None)))


    image_pr = predicted_batch[0]
    psnr = "psnr: {:.4f} decibels".format(measure.compare_psnr(image_or, image_pr))
    ssim = "ssim: {:.4f}".format(measure.compare_ssim(image_or, image_pr, multichannel=True))
    print("\n***** Measurements *****")
    print(psnr)
    print(ssim)

    plt.figure()
    plt.imshow(predicted_batch[0])
    plt.figure()
    plt.imshow(image_bc)
    plt.show()