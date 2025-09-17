'''
Author: wdj
Date: 2021-01-13 14:59:17
LastEditTime: 2021-01-14 18:02:39
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
import gdal
from imgaug import augmenters as iaa
import torch.nn.functional as F
import cv2


class SARLABDataSet(data.Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if self.mode == "train":
            self.lab_list = sorted(os.listdir(
                "./datasets/sar_rgb/label/"))[:-1000]
            self.sar_list = sorted(os.listdir(
                "./datasets/sar_rgb/sar"))[:-1000]
            self.rgb_list = sorted(os.listdir(
                "./result/test"))[:-1000]
        if self.mode == "val":
            self.lab_list = sorted(os.listdir(
                "./datasets/sar_rgb/label/"))[-1000:]
            self.sar_list = sorted(os.listdir(
                "./datasets/sar_rgb/sar"))[-1000:]
            self.rgb_list = sorted(os.listdir(
                "./result/test"))[-1000:]

    def __len__(self):
        return len(self.sar_list)

    def __getitem__(self, index):
        # rgbr = gdal.Open(os.path.join("./data/rgb/", self.rgbr_list[index])).ReadAsArray()
        # r = np.expand_dims(rgbr[0,:,:].copy(), 0)
        # rgbr = np.concatenate((rgbr, r), axis = 0)

        lab = gdal.Open(os.path.join("./datasets/sar_rgb/label/",
                                     self.lab_list[index])).ReadAsArray() / 255
        sar = gdal.Open(os.path.join("./datasets/sar_rgb/sar",
                                     self.sar_list[index])).ReadAsArray() / 50.
        rgb = cv2.imread(os.path.join("./result/test",
                                      self.rgb_list[index]))[:, :, ::-1].transpose(2, 0, 1) / 255.
        # rgb = gdal.Open(os.path.join("./result/test",
        #                              self.rgb_list[index])).ReadAsArray() / 255.
        sar = np.concatenate((sar, rgb))
        if self.mode == "train":
            lab, sar = self.random_crop(lab, sar, 512)
        if self.mode == "val":
            lab, sar = self.center_crop(lab, sar, 512)
            # print(rgbr.shape, lab.shape, sar.shape)
            # print(self.rgbr_list[index], self.lab_list[index])
            # resizer = iaa.Resize({"height": 450, "width": 450})
            # lab = resizer.augment_image(lab)
            # rgbr = rgbr.transpose((1, 2, 0))
            # rgbr = resizer.augment_image(rgbr)
            # rgbr = rgbr.transpose((2, 0, 1))

            # sar = sar.transpose((1, 2, 0))
            # sar = resizer.augment_image(sar)
            # sar = sar.transpose((2, 0, 1))

            # rgbr = torch.from_numpy(rgbr.astype(np.float32))/255.0
        lab = torch.from_numpy(lab.astype(np.float32))
        sar = torch.from_numpy(sar.astype(np.float32))
        return sar, lab

    def random_crop(self, lab, sar, crop_size):
        h = np.random.randint(0, sar.shape[1] - crop_size)
        w = np.random.randint(0, sar.shape[2] - crop_size)
        sar = sar[:, h:h + crop_size, w:w + crop_size]
        lab = lab[h:h + crop_size, w:w + crop_size]
        return lab, sar

    def center_crop(self, lab, sar, crop_size):
        lab = lab[194:194 + crop_size, 194:194 + crop_size]
        sar = sar[:, 194:194 + crop_size, 194:194 + crop_size]
        return lab, sar


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
