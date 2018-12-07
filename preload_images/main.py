
import numpy

from matplotlib import pyplot as plt
from scipy import misc as ms

import skimage

def image_difference(image1, image2):
    res_image = image1.copy()
    for i in range(0, len(image1)):
        for j in range(0, len(image1[i])):
            for k in range(0, len(image1[i][j])):
                print(image1[i][j][k], int(image1[i][j][k]) - int(image2[i][j][k]))
                res_image[i][j][k] = 127 - (int(image1[i][j][k]) - int(image2[i][j][k]))
    return res_image


def main():
    image_name = "../original/train_image_png_1.png"
    image = ms.imread(image_name, mode='RGB')
    print(image)
    plt.figure()
    plt.imshow(image)
    smaller = ms.imresize(image, size=[48, 48], interp='bicubic')
    plt.figure()
    plt.imshow(smaller)
    larger = ms.imresize(smaller, size=[96, 96], interp='bicubic')
    plt.figure()
    plt.imshow(larger)
    difference = image_difference(image, larger)
    plt.figure()
    plt.imshow(difference)
    plt.show()

    print("done")


if __name__ == "__main__":
    main()
