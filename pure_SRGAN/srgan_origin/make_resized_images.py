import os

DATA_FOLDER = "data2017"

image_folders = os.listdir("data2017")

print ("Available image folers: ")

for idx, folder in enumerate(image_folders):
	print (" (" + str(idx) + ") - " + folder)

selected_folder = input("Enter the folder you want to resize: ")

path = DATA_FOLDER + "/" + image_folders[int(selected_folder)] + "/X4"

print ("Using: " +  path)

for image_name in os.listdir(path):
	image_path = path + "/" + image_name
	print ("Processing: " + image_path)
	save_name = image_path.replace("bicubic", "resized")
	print ("Storing to: " + save_name)
