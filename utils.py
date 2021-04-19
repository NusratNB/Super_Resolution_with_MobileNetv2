import os
import cv2
import tensorflow as tf


def data_gen(path_to_images, data_type, path_save):
    img_names = os.listdir(path_to_images)
    img_names.sort()
    data = open(path_save + data_type + ".txt", "w")
    for i in range(len(img_names)):
        img = cv2.imread(path_to_images + img_names[i])
        img_h = img.shape[0]
        img_w = img.shape[1]
        count_w = int(img_w / 128)
        count_h = int(img_h / 64)
        for w in range(count_w):
            for h in range(count_h):
                line = img_names[i] + "_"+str(w) + "_" + str(h) + "\n"
                data.write(line)
    data.close()


def ssim_loss(predicted, gt, max_value=1.0):
    return 1 - tf.reduce_mean(tf.image.ssim(predicted, gt, max_val=max_value))


def psnr_loss(predicted, gt, max_value=1.0):
    return 1 - tf.reduce_mean(tf.image.psnr(predicted, gt, max_val=max_value))

