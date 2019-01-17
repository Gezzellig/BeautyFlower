import imageio
from scipy import misc as ms
import numpy as np


def resize_image(image_path, factor):
    original = imageio.imread(image_path)
    print(orignal.shape)
    return ms.imresize(original, size=[48, 48], interp='bicubic')

def main():
    if not len(sys.argv) == 1:
        print("Usage: preload.py <input_mage_path> <scale_factor>")
        exit(-1)
    resize_image(sys.argv[1], sys.argv[2])
    print("Done!")

if __name__ == "__main__":
    main()
