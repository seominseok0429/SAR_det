import torch
from torch import nn
import torch.nn.functional as F


class HRSAR(nn.Module):
    def __init__(self, n_classes=2):
        super(HRSAR, self).__init__()
        self.first_layer = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3, bias=None),
                                         nn.BatchNorm2d(16),
                                         nn.LeakyReLU())
        self.layer21 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=None),
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU())
        self.layer22 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, dilation=2, bias=None),
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU())
        self.layer23 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4, dilation=4, bias=None),
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU())
        self.down1 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1, stride=2, padding=0, bias=None),
                                     nn.BatchNorm2d(48),
                                     nn.LeakyReLU())
        self.down2 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1, stride=2, padding=0, bias=None),
                                   nn.BatchNorm2d(48),
                                   nn.LeakyReLU())
        self.down3 = nn.Sequential(nn.Conv2d(48, 48, kernel_size=1, stride=2, padding=0, bias=None),
                                   nn.BatchNorm2d(48),
                                   nn.LeakyReLU())
        self.up3 = nn.ConvTranspose2d(48, 16, kernel_size=4, stride=2, padding=1, bias=None)
        self.up2 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1, bias=None)
        self.up1 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1, bias=None)



    def forward(self, x):
        x = self.first_layer(x)
        x1 = self.layer21(x)
        x2 = self.layer22(x)
        x3 = self.layer23(x)
        concat = torch.cat((x1, x2), dim=1)
        concat = torch.cat((concat, x3), dim=1)
        down1 = self.down1(concat)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        up3 = self.up3(down3)
        up3 = torch.cat((up3, down2), dim=1)
        up2 = self.up2(up3)
        up2 = torch.cat((up2, down1), dim=1)
        up1 = self.up1(up2)
        return up1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


