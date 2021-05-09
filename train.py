import os
from utils import patch_gen, ssim_loss, patch_gen
from data_gen import DataGenerator
from model import nn
import yaml


def train(sr_model, opt, loss, tr_hr, tr_lr, vl_hr, vl_lr, epochs, batch_size, path_save):
    patch_gen(tr_hr, "train", path_save)
    patch_gen(vl_hr, "validation", path_save)
    val_txt = open(path_save + "validation.txt", "r+")
    tr_txt = open(path_save + "train.txt", "r+")
    tr_patch_list = tr_txt.readlines()
    val_patch_list = (val_txt.readlines())
    if loss == "ssim_loss":
        loss = ssim_loss
    elif loss == "psnr_loss":
        loss = psnr_loss
    tr_generator = DataGenerator(tr_patch_list, batch_size, tr_hr, tr_lr)
    vl_generator = DataGenerator(val_patch_list, batch_size, vl_hr, vl_lr)
    sr_model.compile(optimizer=opt, loss=loss)
    sr_model.fit(tr_generator, epochs=epochs, validation_data=vl_generator)


if __name__ == '__main__':
    with open("config.yml", "r") as yamlfile:
        data = yaml.load(yamlfile)
        print("Read successful")
    model = nn(config['input_shape'])
    train(nn, config['opt'], config['loss'], config['tr_hr_path'], config['tr_lr_path'],
          config['val_hr_path'], config['val_lr_path'], config['epoch'], config['batch_size'],
          config['path_save'])
