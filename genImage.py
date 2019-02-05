from skimage import measure
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import matplotlib.pyplot as plt
import sys
import os

from model import SRGAN_g

if not (len(sys.argv) == 5 or len(sys.argv) == 4):
    print("not the proper amount of parameters passed. use: <model_path> <num_res_blocks> <input_image_folder> <output_image_folder>")
    exit()
model_path = sys.argv[1]
num_units = int(sys.argv[2])
validate_image_path_lr = sys.argv[3]
if len(sys.argv) == 5:
    output_path = sys.argv[4]
else:
    output_path = "."

images_names_lr = sorted(tl.files.load_file_list(path=validate_image_path_lr, regx='.*.png', printable=False))
images_lr_int = tl.vis.read_images(images_names_lr, path=validate_image_path_lr, n_threads=32)
images_lr = [(im / 127.5) - 1 for im in images_lr_int]
t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
net_g = SRGAN_g(t_image, num_units, is_train=False, reuse=False)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name=model_path, network=net_g)
count = 0
for image_lr in images_lr:
    im_lr = np.empty((len(image_lr), len(image_lr[0]), 3), dtype=np.float32)
    for y in range(len(image_lr)):
        for x in range(len(image_lr[0])):
            im_lr[y][x][0] = image_lr[y][x][0]
            im_lr[y][x][1] = image_lr[y][x][1]
            im_lr[y][x][2] = image_lr[y][x][2]

    out = sess.run(net_g.outputs, {t_image: [im_lr]})
    jej = ((out[0]+1)/2)
    plt.imsave("{}/{}R{}x4.png".format(output_path, images_names_lr[count].split(".")[0], num_units), jej)
    count = count + 1

