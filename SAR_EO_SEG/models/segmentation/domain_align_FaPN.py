import torch
from torch import nn
from sar_project.models.segmentation.deform_conv_v2 import DeformConv2d

class DomainAlign_FaPN(nn.Module):
    def __init__(self, num_class=2, inc=2, outc=2, kernel_size=3, stride=1, padding=1, bias=None, modulation=False):
        super(DomainAlign_FaPN, self).__init__()
        self.num_class = num_class
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.modulation = modulation
        # FSM Module
        self.fsm_avg_pool = nn.AdaptiveAvgPool2d((128, 128))
        self.fsm_fm = nn.Conv2d(self.num_class, self.num_class, 1, bias=False)
        self.fsm_fs = nn.Conv2d(self.num_class, self.num_class, 1, bias=False)
        # FAM Module
        self.deformconv = DeformConv2d(2 * self.inc, self.outc, self.kernel_size, self.padding, self.modulation)
        self.fusion = nn.Conv2d(2, 2, 1, bias=False)

    def forward(self, rgb, sar):
        rgb_avg = self.fsm_avg_pool(rgb)
        rgb_fm = torch.sigmoid(self.fsm_fm(rgb_avg))
        rgb_mat = rgb.matmul(rgb_fm) + rgb
        rgb_fs = self.fsm_fs(rgb_mat)
        concat_out = torch.cat((sar, rgb_fs), dim=1)
        fam_out = self.deformconv(concat_out)
        # align_out = fam_out + rgb_fs
        # align_out = self.fusion(align_out)
        return fam_out

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

