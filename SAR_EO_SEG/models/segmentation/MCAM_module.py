import torch
from torch import nn
import torch.nn.functional as F


class MCAM(nn.Module):
    def __init__(self, in_channels):
        super(MCAM, self).__init__()
        self.conv_opt = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.conv_sar = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax()

    def forward(self, opt, sar):
        b, c, h, w = opt.size()
        if opt != None:
            V_opt = self.conv_opt(opt)
            Q_opt = self.conv_opt(opt)
            K_opt = self.conv_opt(opt)
            S_opt = self.softmax(torch.matmul(Q_opt.view(b, c, h, w).transpose(2, 3), K_opt.view(b, c, h, w)))
            V_sar = self.conv_sar(sar)
            Q_sar = self.conv_sar(sar)
            K_sar = self.conv_sar(sar)
            S_sar = self.softmax(torch.matmul(Q_sar.view(b, c, h, w).transpose(2, 3), K_sar.view(b, c, h, w)))
            hadamard_product = S_opt * S_sar
            V_opt_hadamard = V_opt * hadamard_product
            V_sar_hadamard = V_sar * hadamard_product
            Att = V_opt_hadamard * V_sar_hadamard
        return Att

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


