import csv

from skimage import measure
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import matplotlib.pyplot as plt
import sys
import os

from model import SRGAN_g

validate_image_path_hr = "validateImages/hr"
validate_image_path_lr = "validateImages/lr"


def get_psnr_ssim(image_hr, image_generated):
    psnr = measure.compare_psnr(image_hr, image_generated)
    ssim = measure.compare_ssim(image_hr, image_generated, multichannel=True)
    return psnr, ssim


def eval_one_model(model_path, sess, net_g, t_image, images_hr, images_lr):
    print("processing: {}".format(model_path))
    tl.files.load_and_assign_npz(sess=sess, name=model_path, network=net_g)
    psnr_res = []
    ssim_res = []
    for i in range(0, len(images_hr)):
        out = sess.run(net_g.outputs, {t_image: [images_lr[i]]})
        psnr, ssim = get_psnr_ssim(images_hr[i], out[0])
        psnr_res.append(psnr)
        ssim_res.append(ssim)
    return psnr_res, ssim_res


def eval_one_residual_number(num_units, super_model_path, images_hr, images_lr):
    checkpoint_path = "{}/checkpoint".format(super_model_path)
    model_paths = sorted(os.listdir(checkpoint_path))

    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, num_units, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    model_results = []
    for model_path in model_paths:

        #Skip discriminator files
        if model_path.split("_")[0] == "d":
            continue
        epoch = model_path.split("n")[1].split(".")[0]
        if epoch == "i":
            continue
        epoch = int(epoch)
        if epoch > 400:
            continue
        print(epoch)
        model_results.append((epoch, eval_one_model("{}/{}".format(checkpoint_path, model_path), sess, net_g, t_image, images_hr, images_lr)))
    #model_results.append(eval_one_model("resultsML/upEnd_4-19_Jan_2019_19-27-20/checkpoint/d_srgan95.npz", sess, net_g, t_image, images_hr, images_lr))
    sess.close()
    tf.reset_default_graph()
    return model_results

def eval_everything(super_path, images_hr, images_lr):
    model_paths = sorted(os.listdir(super_path))
    for model_path in model_paths:
        num_units = int(model_path.split("r")[1].split("-")[0])
        results = eval_one_residual_number(num_units, "{}/{}".format(super_path, model_path), images_hr, images_lr)
        results = sorted(results)
        psnr_file = open("outProcess/psnr{}.csv".format(num_units), "w")
        psnr_writer = csv.writer(psnr_file)
        ssim_file = open("outProcess/ssim{}.csv".format(num_units), "w")
        ssim_writer = csv.writer(ssim_file)

        for epoch, (psnr, ssim) in results:
            psnr_writer.writerow([epoch] + psnr)
            ssim_writer.writerow([epoch] + ssim)

        psnr_file.close()
        ssim_file.close()
        print("results of {} written to file".format(model_path))


def main():
    if sys.argv == 2:
        print("Please provide the path to all the execution of the models")
    super_path = sys.argv[1]

    images_names_hr = sorted(tl.files.load_file_list(path=validate_image_path_hr, regx='.*.png', printable=False))
    images_hr_int = tl.vis.read_images(images_names_hr, path=validate_image_path_hr, n_threads=32)
    images_hr = [(im / 127.5) - 1 for im in images_hr_int]
    images_names_lr = sorted(tl.files.load_file_list(path=validate_image_path_lr, regx='.*.png', printable=False))
    images_lr_int = tl.vis.read_images(images_names_lr, path=validate_image_path_lr, n_threads=32)
    images_lr = [(im / 127.5) - 1 for im in images_lr_int]
    eval_everything(super_path, images_hr, images_lr)




if __name__ == "__main__":
    main()