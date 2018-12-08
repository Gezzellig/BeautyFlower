import os
import sys
import imageio
from scipy import misc as ms
from tqdm import tqdm


def image_difference(image1, image2):
    res_image = image1.copy()
    for i in range(0, len(image1)):
        for j in range(0, len(image1[i])):
            for k in range(0, len(image1[i][j])):
                print(image1[i][j][k], int(image1[i][j][k]) - int(image2[i][j][k]))
                res_image[i][j][k] = 127 - (int(image1[i][j][k]) - int(image2[i][j][k]))
    return res_image


def store_image_versions(output_folder_path, original, smaller, bicubic):
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
    imageio.imwrite(output_folder_path + "/original.png", original)
    imageio.imwrite(output_folder_path + "/smaller.png", smaller)
    imageio.imwrite(output_folder_path + "/bicubic.png", bicubic)


def preload_image(image_path):
    original = imageio.imread(image_path)
    smaller = ms.imresize(original, size=[48, 48], interp='bicubic')
    bicubic = ms.imresize(smaller, size=[96, 96], interp='bicubic')
    return original, smaller, bicubic


def preload_all_images(input_folder, output_folder):
    image_names = os.listdir(input_folder)
    for i in tqdm(range(len(image_names)), desc=" Preloading images"):
        original, smaller, bicubic = preload_image(input_folder + "/" + image_names[i])
        store_image_versions(output_folder + "/image{:04d}".format(i), original, smaller, bicubic)


def load_image_versions(folder_name):
    original = imageio.imread(folder_name + "/original.png")
    smaller = imageio.imread(folder_name + "/smaller.png")
    bicubic = imageio.imread(folder_name + "/original.png")
    return original, smaller, bicubic


def load_preload_images(input_folder):
    orignal_images = []
    smaller_images = []
    bicubic_images = []
    folders = os.listdir(input_folder)
    for i in tqdm(range(len(folders)), desc=" Loading the preloaded images"):
        original, smaller, bicubic = load_image_versions(input_folder + "/" + folders[i])
        orignal_images.append(original)
        smaller_images.append(smaller)
        bicubic_images.append(bicubic)
    return orignal_images, smaller_images, bicubic_images


def main():
    if not len(sys.argv) == 3:
        print("Usage: preload.py <input_folder> <output_folder>")
        exit(-1)
    preload_all_images(sys.argv[1], sys.argv[2])
    print("Done!")


def direct_load(input_folder):
    image_names = os.listdir(input_folder)
    orignal_images = []
    smaller_images = []
    bicubic_images = []
    for i in tqdm(range(len(image_names)), desc=" loading images directly"):
        original, smaller, bicubic = preload_image(input_folder + "/" + image_names[i])
        orignal_images.append(original)
        smaller_images.append(smaller)
        bicubic_images.append(bicubic)
    return orignal_images, smaller_images, bicubic_images


if __name__ == "__main__":
    main()
