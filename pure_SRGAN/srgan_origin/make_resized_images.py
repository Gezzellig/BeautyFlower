import os
from scipy import misc

DATA_FOLDER = "data2017"

image_folders = os.listdir("data2017")

print ("Available image folers: ")

for idx, folder in enumerate(image_folders):
	print (" (" + str(idx) + ") - " + folder)

selected_folder = input("Enter the folder you want to resize: ")

path = DATA_FOLDER + "/" + image_folders[int(selected_folder)] + "/X4"

print ("Using: " +  path)

# Create the directories if they don't exist
if not os.path.exists(path.replace("bicubic", "resized")):
    os.makedirs(path.replace("bicubic", "resized"))

for image_name in os.listdir(path):
	image_path = path + "/" + image_name
	
	# Load the image
	print ("Processing: " + image_path)
	image = misc.imread(image_path)
	
	# Resize the image
	shape = image.shape
	shape = tuple(s * 4 for s in shape)
	resized_image = misc.imresize(image, shape, interp="bicubic")
	
	# Store the image
	save_name = image_path.replace("bicubic", "resized")
	print ("Storing to: " + save_name)
	misc.imsave(save_name, resized_image)
	
print ("Process finished")
