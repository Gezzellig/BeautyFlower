from skimage import measure
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import matplotlib.pyplot as plt
import sys
import os

from main import select_generator


model_path = sys.argv[1]
num_units = int(sys.argv[2])
validate_image_path_lr = sys.argv[3]

images_names_lr = sorted(tl.files.load_file_list(path=validate_image_path_lr, regx='.*.png', printable=False))
images_lr_int = tl.vis.read_images(images_names_lr, path=validate_image_path_lr, n_threads=32)
images_lr = [(im / 127.5) - 1 for im in images_lr_int]
t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
net_g = select_generator("upEnd", t_image, num_units, is_train=False, reuse=False)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name=model_path, network=net_g)
out = sess.run(net_g.outputs, {t_image: [images_lr[0]]})
jej = ((out[0]+1)/2)
plt.imsave("output.png", jej)
plt.imshow(jej)
plt.show()
