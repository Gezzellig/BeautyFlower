import os
from os import walk
import cv2

cwd = os.getcwd()
ROOT_PATH_HR = cwd + "/data/train_imagesHR/"
ROOT_PATH_LR = cwd + "/data/train_imagesLR/"
print(ROOT_PATH_HR)
print(ROOT_PATH_LR)


#single image = train_image_png_1185.png



for file_idx in range(5000):
    image = cv2.imread(ROOT_PATH_HR + 'train_image_png_' + str(file_idx) + '.png')
    image = cv2.resize(image, (48,48))
    cv2.imwrite(ROOT_PATH_LR + 'train_image_png_' + str(file_idx) + "LR" + '.png')