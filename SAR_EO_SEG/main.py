import numpy as np
import torch
import torch.nn as nn
from skimage import io
from random import randint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
if __name__ == '__main__':
    # im = io.imread('E:\pythonProject\SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190823162315_20190823162606_tile_7863.tif')
    #im = im / 255
    # crop_size = 512
    # h = np.random.randint(0, 900 - crop_size)
    # w = np.random.randint(0, 900 - crop_size)
    # im = im[h:h + crop_size, w:w + crop_size]
    # im = im.astype(np.float32)
    # print(im.shape)


    # plt.figure()
    # cmap = 'nipy_spectral'
    # plt.imshow(im, cmap=plt.get_cmap(cmap))
    # plt.show()

    # x = np.linspace(0, 128, 128)
    # y = np.linspace(0, 128, 128)
    # z = np.zeros((128, 128))
    # for i, a in enumerate(x):
    #     for j, b in enumerate(y):
    #         z[i, j] = np.sin(a + b)

    # X, Y = np.meshgrid(x, y)

    # cm = plt.cm.get_cmap('jet')
    # plt.pcolormesh(X, Y, z.T, cmap=cm)
    # plt.colorbar()
    # plt.show()



    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.backends.cudnn.is_available())
    print(torch.cuda_version)
    print(torch.backends.cudnn.version())