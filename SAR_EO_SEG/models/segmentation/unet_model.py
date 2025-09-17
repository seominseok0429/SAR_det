""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from sar_project.models.segmentation.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.up(x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class U_Attention(nn.Module):
    def __init__(self, in_channels):
        super(U_Attention, self).__init__()
        self.x_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.g_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.stride_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        x_new = self.x_conv(x)
        x_new = self.stride_conv(x_new)
        g_new = self.g_conv(g)
        fuse = x_new + g_new
        fuse = self.relu(fuse)
        fuse = self.conv(fuse)
        fuse = self.sigmoid(fuse)
        fuse = F.interpolate(fuse, size=x.size()[2:], mode='bilinear', align_corners=True)
        att_out = x * fuse
        return att_out


class unet(nn.Module):
    def __init__(self, n_channels=2, n_classes=2, mode='up3', bilinear=True):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mode = mode

        if self.mode == 'up2':
            self.fuse = DoubleConv(256, n_channels)
            self.refuse = DoubleConv(n_classes, 256)

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        self.attention4 = U_Attention(256)
        self.attention3 = U_Attention(128)
        self.attention2 = U_Attention(64)
        self.attention1 = U_Attention(32)
        self.inter3 = DoubleConv(512, 256)
        self.inter2 = DoubleConv(256, 128)
        self.inter1 = DoubleConv(128, 64)
        self.inter0 = DoubleConv(64, 32)

    def forward(self, x):
        if self.mode == 'up2':
            x = self.fuse(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        g4 = self.down4(x4)
        att4 = self.attention4(x4, g4)
        g4 = self.up1(g4)
        g3 = torch.cat((att4, g4), axis=1)
        g3 = self.inter3(g3)
        att3 = self.attention3(x3, g3)
        g3 = self.up2(g3)
        g2 = torch.cat((att3, g3), axis=1)
        g2 = self.inter2(g2)
        att2 = self.attention2(x2, g2)
        g2 = self.up3(g2)
        g1 = torch.cat((att2, g2), axis=1)
        g1 = self.inter1(g1)
        att1 = self.attention1(x1, g1)
        g1 = self.up4(g1)
        g = torch.cat((att1, g1), axis=1)
        g = self.inter0(g)
        logits = self.outc(g)
        if self.mode == 'up2':
            logits = self.refuse(logits)
        return logits