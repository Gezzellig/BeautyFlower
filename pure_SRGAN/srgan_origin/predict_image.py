import os
import cv2
import matplotlib.pyplot as plt


IMAGEPATH       = 'images_to_be_predicted/'
PREDICTEDPATH   = 'predicted_images/'

def load_model(modelname):
    pass

def show_image(image, title):
    im = plt.imread(image)
    print(im)
    plt.imshow(title, im)


# Main code
files = os.listdir(IMAGEPATH)
print ("Available images: ")

for idx, image in enumerate(files):
    print ("({}) {}".format(idx, image))
    

image_choice = input("Choose image path:")
image = files[int(image_choice)]
image_location = (IMAGEPATH + image)
print(image_location)

show_image(image_location, 'Original image' )