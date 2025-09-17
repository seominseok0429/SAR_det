""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from sar_project.models.segmentation.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from sar_project.models.segmentation.backbone import resnet, xception, drn, mobilenet

class BYOL(nn.Module):
    def __init__(self, mode):
        super(BYOL, self).__init__()
        self.mode = mode
        BatchNorm = nn.BatchNorm2d
        self.encoder = resnet.ResNet101(16, BatchNorm, mode='rgb')
        self.sequeeze = nn.Conv2d(2048, 32, kernel_size=1)
        # projector
        self.projector = nn.Sequential(nn.Linear(32768, 2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 256),
                                     nn.ReLU())
        # predictor
        self.predictor = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU())

        self.classifier = nn.Conv2d(2048, 10, kernel_size=1)

    def forward(self, x, mode):
        c = x.size(1)
        if c == 3:
            input = x
        else:
            input = torch.cat((x,x), dim=1)
            input = input[:,:3,:,:]
        down1, down2, down3, down4, encoder_out = self.encoder(input)
        classifier_out = self.classifier(encoder_out)
        segment_out = F.interpolate(classifier_out, size=x.size()[
                          2:], mode='bilinear', align_corners=True)
        projector_in = self.sequeeze(encoder_out)
        # projector
        projector_in = torch.flatten(projector_in, start_dim=1)  # 展平数据，转化为[4,32768]
        projector_out = self.projector(projector_in)
        if mode == 'Teacher':
            byol_out = projector_out
        if mode == 'Student':
            byol_out = self.predictor(projector_out)

        return segment_out, byol_out

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

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = model.state_dict()
    new_dict = {}
    pre_dict = {k: v for k, v in model_state.items() if k in model_ema_state}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1-alpha) * pre_dict[key]
    ema_model.load_state_dict(new_dict)







