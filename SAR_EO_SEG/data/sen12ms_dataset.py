'''
Author: wdj
Date: 2021-01-13 14:59:17
LastEditTime: 2021-01-20 18:45:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/data/sarlab_dataset.py
'''
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
# import torchvision
from torch.utils import data
from PIL import Image
from osgeo import gdal
from imgaug import augmenters as iaa
import torch.nn.functional as F
import cv2
from sar_project.data.dfc_sen12ms_dataset import S2Bands
from sar_project.data.dfc_sen12ms_dataset import Sensor
import rasterio
from skimage import io


class SEN12MS(data.Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if self.mode == "train":
            self.lab_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS/label/"))[:2401]
            self.sar_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS/sar"))[:2401]
            self.rgb_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS/rgb"))[:2401]
        if self.mode == "val":
            self.lab_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS_DFC/label/"))[-40:]
            self.sar_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS_DFC/sar"))[-40:]
            self.rgb_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS_DFC/rgb"))[-40:]
        if self.mode == 'test':
            self.lab_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS_DFC/label/"))[-40:]
            self.sar_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS_DFC/sar"))[-40:]
            self.rgb_list = sorted(os.listdir(
                "SAR_temp/datasets/SEN12MS_DFC/rgb"))[-40:]

    def __len__(self):
        return len(self.sar_list)

    def __getitem__(self, index):
        # rgbr = gdal.Open(os.path.join("./data/rgb/", self.rgbr_list[index])).ReadAsArray()
        # r = np.expand_dims(rgbr[0,:,:].copy(), 0)
        # rgbr = np.concatenate((rgbr, r), axis = 0)

        if self.mode == "train":
            lab_train = gdal.Open(os.path.join("SAR_temp/datasets/SEN12MS/label/",
                                               self.lab_list[index])).ReadAsArray() - 1
            sar_train = gdal.Open(os.path.join("SAR_temp/datasets/SEN12MS/sar",
                                               self.sar_list[index])).ReadAsArray() / 50.
            rgb_train = gdal.Open(os.path.join("SAR_temp/datasets/SEN12MS/rgb",
                                         self.rgb_list[index])).ReadAsArray() / 50.
            rgb_train_blue = rgb_train[1, :, :]
            rgb_train_blue = np.expand_dims(rgb_train_blue, axis=0)
            rgb_train_green = rgb_train[2, :, :]
            rgb_train_green = np.expand_dims(rgb_train_green, axis=0)
            rgb_train_red = rgb_train[3, :, :]
            rgb_train_red = np.expand_dims(rgb_train_red, axis=0)
            rgb_train = np.concatenate((rgb_train_red, rgb_train_green, rgb_train_blue),axis=0)

            # img = io.imread('SAR_temp/datasets/SEN12MS/label/ROIs0000_test_dfc_0_p18.tif')
            # img = img.numpy()
            # lab_un = np.unique(img)
            # print(lab_train)
            # print(lab_un)
            # lab_train[lab_train == lab_un[0]] = 0
            # lab_train[lab_train == lab_un[1]] = 1
            # lab_train[lab_train == lab_un[2]] = 2
            # lab_train[lab_train == lab_un[3]] = 3
            # lab_train[lab_train == lab_un[4]] = 4
            # lab_train[lab_train == lab_un[5]] = 5
            # lab_train[lab_train == lab_un[6]] = 6
            # lab_train[lab_train == lab_un[7]] = 7
            # for i in range(256):
            #     for j in range(256):
            #         l = lab_train[i, j]
            #         if l not in [0, 1, 2, 3, 4, 5, 6, 7]:
            #             lab_train[i, j] = 8

        if self.mode == "val":
            lab_val = gdal.Open(os.path.join("SAR_temp/datasets/SEN12MS_DFC/label/",
                                               self.lab_list[index])).ReadAsArray() - 1
            sar_val = gdal.Open(os.path.join("SAR_temp/datasets/SEN12MS_DFC/sar",
                                               self.sar_list[index])).ReadAsArray() / 50.
            rgb_val = gdal.Open(os.path.join("SAR_temp/datasets/SEN12MS_DFC/rgb",
                                         self.rgb_list[index])).ReadAsArray() / 50.
            rgb_val_blue = rgb_val[1, :, :]
            rgb_val_blue = np.expand_dims(rgb_val_blue, axis=0)
            rgb_val_green = rgb_val[2, :, :]
            rgb_val_green = np.expand_dims(rgb_val_green, axis=0)
            rgb_val_red = rgb_val[3, :, :]
            rgb_val_red = np.expand_dims(rgb_val_red, axis=0)
            rgb_val = np.concatenate((rgb_val_red, rgb_val_green, rgb_val_blue), axis=0)

            # img = io.imread('SAR_temp/datasets/SEN12MS/label/ROIs0000_test_dfc_0_p18.tif')
            # img = img.numpy()
            # lab_un = np.unique(img)
            # lab_val[lab_val == lab_un[0]] = 0
            # lab_val[lab_val == lab_un[1]] = 1
            # lab_val[lab_val == lab_un[2]] = 2
            # lab_val[lab_val == lab_un[3]] = 3
            # lab_val[lab_val == lab_un[4]] = 4
            # lab_val[lab_val == lab_un[5]] = 5
            # lab_val[lab_val == lab_un[6]] = 6
            # lab_val[lab_val == lab_un[7]] = 7
            # for i in range(256):
            #     for j in range(256):
            #         l = lab_val[i, j]
            #         if l not in [0, 1, 2, 3, 4, 5, 6, 7]:
            #             lab_val[i, j] = 8
        # patch_path_train = "SAR_temp/datasets/SEN12MS/rgb"
        # patch_path_val = "SAR_temp/datasets/SEN12MS_DFC/rgb"
        # bands = S2Bands.RGB
        # bands = [b.value for b in bands]
        # with gdal.Open(patch_path_train) as patch:
        #     rgb_train = patch.read(bands)/255
        # with gdal.Open(patch_path_val) as patch_1:
        #     rgb_val = patch_1.read(bands)/255
        # rgb = cv2.imread(os.path.join("SAR_temp/datasets/sar_rgb/rgb",
        #                               self.rgb_list[index]), -1)[:, :, ::-1].transpose(2, 0, 1) / 255.
        if self.mode == "train":
            lab, sar, rgb = lab_train, sar_train, rgb_train
        if self.mode == "val":
            lab, sar, rgb = lab_val, sar_val, rgb_val
        lab = torch.from_numpy(lab.astype(np.float32))
        sar = torch.from_numpy(sar.astype(np.float32))
        rgb = torch.from_numpy(rgb.astype(np.float32))
        if self.mode in ["train","val"]:
            return sar, lab, rgb
        else:
            return sar, lab, rgb # rgb, lab, sar

    def random_crop(self, lab, sar, rgb, crop_size):
        h = np.random.randint(0, sar.shape[1] - crop_size)
        w = np.random.randint(0, sar.shape[2] - crop_size)
        sar = sar[:, h:h + crop_size, w:w + crop_size]
        rgb = rgb[:, h:h + crop_size, w:w + crop_size]
        lab = lab[h:h + crop_size, w:w + crop_size]
        return lab, sar, rgb

    def center_crop(self, lab, sar, rgb, crop_size):
        lab = lab[194:194 + crop_size, 194:194 + crop_size]
        sar = sar[:, 194:194 + crop_size, 194:194 + crop_size]
        rgb = rgb[:, 194:194 + crop_size, 194:194 + crop_size]
        return lab, sar, rgb


if __name__ == '__main__':
    dst = SARLABDataSet()
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        sar, lab = data
        print(sar.size())
        # print(rgbr.max(), lab.max(), sar.max())
        # if i == 0:
        #     img = torchvision.utils.make_grid(imgs).numpy()
        #     img = np.transpose(img, (1, 2, 0))
        #     img = img[:, :, ::-1]
        #     plt.imshow(img)
        #     plt.show()
