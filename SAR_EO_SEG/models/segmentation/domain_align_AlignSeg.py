import torch
from torch import nn
from .deform_conv_v2 import DeformConv2d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

class DomainAlign_AlignSeg(nn.Module):
    def __init__(self, align_class=2, num_class=2):
        super(DomainAlign_AlignSeg, self).__init__()

        self.change = nn.Conv2d(align_class, num_class, kernel_size=1, bias=False)
        self.delta_gen1 = nn.Sequential(
            nn.Conv2d(num_class * 2, num_class, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_class),
            nn.ReLU(),
            nn.Conv2d(num_class, 2, kernel_size=3, padding=1, bias=False)
            )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(num_class * 2, num_class, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_class),
            nn.ReLU(),
            nn.Conv2d(num_class, 2, kernel_size=3, padding=1, bias=False)
            )
        self.last = nn.Conv2d(2, 2, kernel_size=1, bias=False)

        self.delta_gen1[3].weight.data.zero_()
        self.delta_gen2[3].weight.data.zero_()
        self.last.weight.data.zero_()



    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[h / s,
                                w / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1,
                                                       1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=False)
        return output

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        norm = torch.tensor([[[[1, 1]]]]).type_as(input).to(input.device)

        delta_clam = torch.clamp(delta.permute(0, 2, 3, 1) / norm, -1, 1)
        grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, out_h), torch.linspace(-1, 1, out_w)), dim=-1).unsqueeze(0)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)

        grid = grid.detach() + delta_clam
        output = F.grid_sample(input, grid)
        return output

    # def forward(self, G_stage, L_stage):
    #     h, w = L_stage.size(2), L_stage.size(3)
    #     fusion = torch.cat((L_stage, G_stage), 1)
    #     delta = self.delta_gen1(fusion)
    #     L_stage = self.bilinear_interpolate_torch_gridsample(L_stage, (h, w), delta)
    #     concat = torch.cat((L_stage, G_stage), 1)
    #     output = self.last(concat)
    #     return output
        


    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        low_stage = self.change(low_stage)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)
        high_stage += low_stage

        # delta1 = delta1.data.cpu().numpy().squeeze()
        # delta2 = delta2.data.cpu().numpy().squeeze()
        # H = delta1[0,:,:].squeeze()
        # W = delta1[1,:,:].squeeze()

        # length = np.zeros((128, 128))
        # angle = np.zeros((128, 128))
        # for i in range(128):
        #     for j in range(128):
        #         x = H[i,j]
        #         y = W[i,j]
        #         length[i,j] = np.sqrt(x**2 + y**2)
        #         angle[i,j] = math.atan2(y, x)


        # print(delta1[0,:,:])

        # plt.figure()
        # x = np.linspace(0, 128, 128)
        # y = np.linspace(0, 128, 128)
        # z = np.zeros((128, 128))
        # for i, a in enumerate(x):
        #     for j, b in enumerate(y):
        #         z[i, j] = np.sqrt((delta1[0,i,j] ** 2) + (delta1[1,i,j] ** 2))
        # X, Y = np.meshgrid(x, y)
        # cm = plt.cm.get_cmap('jet')
        # plt.pcolormesh(X, Y, z.T, cmap=cm)
        # plt.colorbar()

        # cmap = 'YlGnBu'
        # ax = plt.subplot(1, 2, 1)
        # plt.imshow(length, cmap=plt.get_cmap(cmap))
        # plt.colorbar()
        # ax = plt.subplot(1, 2, 2)
        # plt.imshow(angle, cmap=plt.get_cmap(cmap))
        # plt.colorbar()
        # plt.show()

        return high_stage

