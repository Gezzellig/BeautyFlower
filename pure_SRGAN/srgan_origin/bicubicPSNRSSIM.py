import csv

import scipy
import skimage
import matplotlib.pyplot as plt
import numpy as np
import tensorlayer as tl
from skimage import measure

validate_image_path_hr = "validateImages/hr"
validate_image_path_lr = "validateImages/lr"

images_names_hr = sorted(tl.files.load_file_list(path=validate_image_path_hr, regx='.*.png', printable=False))
images_hr_int = tl.vis.read_images(images_names_hr, path=validate_image_path_hr, n_threads=32)
#images_hr = [(im / 127.5) - 1 for im in images_hr_int]
images_names_lr = sorted(tl.files.load_file_list(path=validate_image_path_lr, regx='.*.png', printable=False))
images_lr_int = tl.vis.read_images(images_names_lr, path=validate_image_path_lr, n_threads=32)
#images_lr = [(im / 127.5) - 1 for im in images_lr_int]

bicubics_int = []
for image_lr_int in images_lr_int:
    bicubics_int.append(scipy.misc.imresize(image_lr_int, 4.0, interp='bicubic'))
bicubics_int = np.array(bicubics_int)
bicubics = [(im / 127.5) - 1 for im in bicubics_int]
images_hr = [(im / 127.5) - 1 for im in images_hr_int]
plt.figure()
plt.imshow(images_hr[0])
plt.figure()
plt.imshow(bicubics[0])
plt.show()

psnr_file = open("outProcess/psnrBicubic.csv", "w")
psnr_writer = csv.writer(psnr_file)
ssim_file = open("outProcess/ssimBicubic.csv", "w")
ssim_writer = csv.writer(ssim_file)

psnr = []
ssim = []
for i in range(0, len(images_hr)):
    print(len(images_hr[i]), len(bicubics[i]))

    psnr.append(measure.compare_psnr(images_hr[i], bicubics[i]))
    ssim.append(measure.compare_ssim(images_hr[i], bicubics[i], multichannel=True))
psnr_writer.writerow([0] + psnr)
ssim_writer.writerow([0] + ssim)

print(psnr)
print(ssim)

psnr_file.close()
ssim_file.close()

#psnr = measure.compare_psnr(image_hr, image_generated)
#ssim = measure.compare_ssim(image_hr, image_generated, multichannel=True)