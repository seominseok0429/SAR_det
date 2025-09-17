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
class RgbrDataSet(data.Dataset):
    def __init__(self):
        # self.root = "./data/rgbr/"
        self.rgbr_list = sorted(os.listdir("./data/rgb/"))[:-1000]
        self.lab_list = sorted(os.listdir("./data/label/"))[:-1000]
        self.sar_list = sorted(os.listdir("./data/sar/"))[:-1000]
        # print(len(self.rgbr_list))
        # print(len(self.lab_list))
        # print(len(self.sar_list))
    def __len__(self):
        return len(self.rgbr_list)


    def __getitem__(self, index):
        rgbr = gdal.Open(os.path.join("./data/rgb/", self.rgbr_list[index])).ReadAsArray()
        r = np.expand_dims(rgbr[0,:,:].copy(), 0)
        rgbr = np.concatenate((rgbr, r), axis = 0)
        
        lab = gdal.Open(os.path.join("./data/label/", self.lab_list[index])).ReadAsArray() / 255
        sar = gdal.Open(os.path.join("./data/sar/", self.sar_list[index])).ReadAsArray()
        rgbr, lab, sar = self.random_crop(rgbr, lab, sar, 512)
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

        rgbr = torch.from_numpy(rgbr.astype(np.float32))/255.0
        lab = torch.from_numpy(lab.astype(np.float32))
        sar = torch.from_numpy(sar.astype(np.float32))/50.0
        return rgbr, lab, sar 

    def random_crop(self, rgbr, lab, sar, crop_size):
        h = np.random.randint(0, sar.shape[1] - crop_size)
        w = np.random.randint(0, sar.shape[2] - crop_size)
        sar = sar[:, h:h + crop_size, w:w + crop_size]
        rgbr = rgbr[:, h:h + crop_size, w:w + crop_size]
        lab = lab[h:h + crop_size, w:w + crop_size]
        return rgbr, lab, sar
        # datafiles = self.files[index]

        # image = Image.open(datafiles["img"]).convert('RGB')
        # label = Image.open(datafiles["label"])
        # name = datafiles["name"]

        # resize
        # image = image.resize(self.crop_size, Image.BICUBIC)
        # label = label.resize(self.crop_size, Image.NEAREST)

        # image = np.asarray(image, np.float32)
        # label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        # label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v

        # size = image.shape
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = image.transpose((2, 0, 1))

        # return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = RgbrDataSet()
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        rgbr, lab, sar = data
        # print(rgbr.max(), lab.max(), sar.max())
        # if i == 0:
        #     img = torchvision.utils.make_grid(imgs).numpy()
        #     img = np.transpose(img, (1, 2, 0))
        #     img = img[:, :, ::-1]
        #     plt.imshow(img)
        #     plt.show()
