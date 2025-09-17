import torch
from torch import nn
import torch.nn.functional as F

class Contrast_RS(nn.Module):
    def __init__(self, inc=2, outc=2):
        super(Contrast_RS, self).__init__()
        self.inc = inc
        self.outc = outc
        # 64,128
        self.encoder1 = nn.Sequential(nn.Conv2d(self.inc, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())
        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(256, self.outc, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(self.outc),
                                     nn.ReLU())

    def forward(self, x):
        en1_out = self.encoder1(x)
        en2_out = self.encoder2(en1_out)
        de1_out = self.decoder1(en2_out)
        de2_out = self.decoder2(de1_out)
        output = F.interpolate(de2_out, size=x.size()[2:], mode='bilinear', align_corners=True)
        return output

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