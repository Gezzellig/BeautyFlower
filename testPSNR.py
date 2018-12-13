from matplotlib import pyplot

from preload import load_image_versions
import skimage
import tensorflow as tf


def main():
    print("start")
    original, smaller, bicubic = load_image_versions("learnset/image0000")
    ssim = tf.image.ssim(original, bicubic)
    print("decibels {}".format(skimage.measure.compare_psnr(original, bicubic)))
    print("decibels {}".format(tf.image.psnr(original, bicubic, max_val=255)))
    print("SSIM {}".format(ssim))


if __name__ == "__main__":
    main()