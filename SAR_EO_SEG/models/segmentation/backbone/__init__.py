'''
Author: your name
Date: 2021-01-13 14:56:48
LastEditTime: 2021-01-13 15:59:32
LastEditors: your name
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/models/segmentation/backbone/__init__.py
'''
# from sar_project.models.segmentation.backbone import resnet, xception, drn, mobilenet
from . import resnet, xception, drn, mobilenet


def build_backbone(mode, backbone, output_stride, BatchNorm):
    if backbone == 'resnet-50':
        return resnet.ResNet50(output_stride, BatchNorm)
    if backbone == 'resnet-101':
        return resnet.ResNet101(output_stride, BatchNorm, mode)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
