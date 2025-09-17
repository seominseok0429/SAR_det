# import os
# import os.path as osp
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import collections
# import torch
# # import torchvision
# from torch.utils import data
# from PIL import Image
# import gdal
# from imgaug import augmenters as iaa

# class SarDataSet(data.Dataset):
#     def __init__(self):
#         # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
#         self.sarimages_list = sorted(os.listdir("./data/sar/"))[:-1000]

#     def __len__(self):
#         return len(self.sarimages_list)

#     def __getitem__(self, index):
#         sar = gdal.Open(os.path.join("./data/sar/", self.sarimages_list[index])).ReadAsArray() / 255. * 3
#         resizer = iaa.Resize({"height": 450, "width": 450})
#         sar = sar.transpose((1, 2, 0))
#         sar = resizer.augment_image(sar)
#         sar = sar.transpose((2, 0, 1))
#         sar = torch.from_numpy(sar.astype(np.float32))

#         return sar

# # if __name__ == '__main__':
# #     dst = SarDataSet()
# #     trainloader = data.DataLoader(dst, batch_size=4)
# #     for i, data in enumerate(trainloader):
# #         imgs, labels = data
# #         if i == 0:
# #             img = torchvision.utils.make_grid(imgs).numpy()
# #             img = np.transpose(img, (1, 2, 0))
# #             img = img[:, :, ::-1]
# #             plt.imshow(img)
# #             plt.show()
