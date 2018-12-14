from preload import load_image_versions
from skimage import measure


def main():
    print("start")
    original, smaller, bicubic = load_image_versions("learnset/image0000")

    print("psnr {:.4f} decibels".format(measure.compare_psnr(original, bicubic)))
    print("ssim {:.4f}".format(measure.compare_ssim(original, bicubic, multichannel=True)))


if __name__ == "__main__":
    main()
