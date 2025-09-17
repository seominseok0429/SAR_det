'''
Author: wdj
Date: 2021-01-06 17:26:45
LastEditTime: 2021-01-12 18:34:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/data/sarrgb_dataset.py
'''

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import gdal
import torch
import numpy as np
import glob
import cv2

class SARRGBDataset(BaseDataset):
    """A dataset class for paired sar-rgb image dataset.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        if opt.isTrain:
            self.A_paths = sorted(glob.glob(os.path.join(opt.dataroot, "rgb") + "/*tif"))[:-1000]  # get rgb image paths
            self.B_paths = sorted(glob.glob(os.path.join(opt.dataroot, "sar") + "/*tif"))[:-1000]  # get sar image paths
        else:
            self.A_paths = sorted(glob.glob(os.path.join(opt.dataroot, "rgb") + "/*tif"))[:]  # get rgb image paths
            self.B_paths = sorted(glob.glob(os.path.join(opt.dataroot, "sar") + "/*tif"))[:]  # get sar image paths
        # self.L_paths = sorted(glob.glob(os.path.join(opt.dataroot, "label") + "/*tif"))[:-1000]  # get label image paths
        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = cv2.imread(A_path, -1)[:,:,::-1]
        B = cv2.imread(B_path, -1)
        A = cv2.resize(A, (768, 768)).transpose(2, 0, 1)
        B = cv2.resize(B, (768, 768)).transpose(2, 0, 1)
        # L_path = self.L_paths[index]
        # A = gdal.Open(A_path).ReadAsArray()
        # B = gdal.Open(B_path).ReadAsArray()
        # L = gdal.Open(L_path).ReadAsArray()
        # A, B, L = self.random_crop(A, B, L, 512)
        A = torch.from_numpy(A.astype(np.float32))/255.0
        B = torch.from_numpy(B.astype(np.float32))/50.0
        # L = torch.from_numpy(L.astype(np.float32))
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # A = A_transform(A)
        # B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def random_crop(self, A, B, L, crop_size):
        h = np.random.randint(0, A.shape[1] - crop_size)
        w = np.random.randint(0, A.shape[2] - crop_size)
        A = A[:, h:h + crop_size, w:w + crop_size]
        B = B[:, h:h + crop_size, w:w + crop_size]
        L = L[h:h + crop_size, w:w + crop_size]
        return A, B, L