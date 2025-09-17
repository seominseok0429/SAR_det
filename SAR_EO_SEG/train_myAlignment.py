'''
Author: your name
Date: 2021-01-13 14:57:42
LastEditTime: 2021-01-20 20:09:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/datasets/trainsar.py
'''
# !/usr/bin/env python

import os
import sys
import glob
import math
import uuid
import shutil
import pathlib
import argparse
import numpy as np
# import pandas as pd
# import geopandas as gpd
import skimage
import torch
import torch.nn as nn
import tqdm
import gdal

# import solaris as sol

# from data.sarlab_dataset import SARLABDataSet
from sar_project.data.sarrgblab_dataset import SARLABDataSet
from sar_project.data.sen12ms_dataset import SEN12MS
from sar_project.models.segmentation.deeplab import DeepLab
from sar_project.models.segmentation.deeplab_myalign import DeepLab_myalign
from sar_project.util.loss import SegmentationLosses
from sar_project.models.segmentation.sync_batchnorm.replicate import patch_replication_callback
from torch.utils.data import DataLoader
from sar_project.util.lr_scheduler import LR_Scheduler
# from metrics import Evaluator
from sklearn.metrics import f1_score
from sar_project.util.evaluator import Evaluator
from sar_project.util.util import mkdir
from sar_project.options.train_options import TrainOptions
import torch.nn.functional as F

# 指定代码在服务器董某块显卡上跑，一般放在整段代码最前面，即import torch前
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def sar_train(args):
    # opt = TrainOptions().parse()
    # mkdir = Utils().mkdir
    train_set = SARLABDataSet(mode="train")
    val_set = SARLABDataSet(mode="val")
    # train_set = SEN12MS(mode="train")
    # val_set = SEN12MS(mode="val")
    train_loader = DataLoader(train_set, batch_size=8,
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, drop_last=True, num_workers=4)

    model_sar = DeepLab_myalign(mode='sar', num_classes=2)
    model_rgb = DeepLab_myalign(mode='rgb', num_classes=2)

    # train_params = model_sar.parameters()
    train_params = model_sar.parameters()
    optimizer = torch.optim.Adam(train_params, lr=args.lr)

    criterion_focal = SegmentationLosses().build_loss(mode_l="focal")
    criterion_ce = SegmentationLosses().build_loss(mode_l="ce")
    criterion_L1 = SegmentationLosses().build_loss(mode_l="L1")
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                             args.epochs, len(train_loader))

    model_sar = torch.nn.DataParallel(model_sar, device_ids=[0, 1])
    patch_replication_callback(model_sar)
    model_sar = model_sar.cuda()

    model_rgb = torch.nn.DataParallel(model_rgb, device_ids=[0, 1])
    patch_replication_callback(model_rgb)
    model_rgb = model_rgb.cuda()

    rgb_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_rgb_AlignSeg2/50.pth')
    # rgb_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_rgb_sen12ms/50.pth')
    model_rgb.module.load_state_dict(rgb_weights, strict=True)
    model_rgb.eval()

    # sar_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_sar_myalign7/100.pth')
    # model_sar.module.load_state_dict(sar_weights, strict=True)
    # model_sar.eval()

    # sar_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_sar_myalign13/51.pth')
    sar_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_sar_seg/50.pth')
    # sar_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_sar_sen12ms/50.pth')
    model_dict = model_sar.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in sar_weights.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model_sar.load_state_dict(model_dict)
    model_sar.eval()

    # rawsarrgb_weights = torch.load(
    #     "./checkpoint/segmentation/deeplab+rgb/49.pth")
    # model.module.load_state_dict(rawsarrgb_weights, strict=True)
    '''
    max_f1 = 0.0
    max_iou = 0.0
    max_acc = 0.0
    min_loss = float(1e5)
    '''
    evaluator = Evaluator(2)

    T = 5
    m = nn.Sigmoid()

    for epoch in range(args.epochs):
        train_loss = 0.0
        # train_loss_sar = 0.0
        # train_loss_rgb = 0.0
        # model_rgb_val.train()
        # zeros = torch.zeros([8,2,512, 512])
        model_rgb.train()
        tbar = tqdm.tqdm(train_loader)

        for i, sample in enumerate(tbar):
            sar, lab, rgb = sample
            sar, lab, rgb = sar.cuda(), lab.cuda(), rgb.cuda()
            with torch.no_grad():
                # up1_rgb, up2_rgb, up3_rgb, output_rgb = model_rgb(rgb)
                up1_sar, up2_sar, up3_sar, output_sar = model_sar(sar)
            scheduler(optimizer, i, epoch)
            optimizer.zero_grad()
            # up1_sar, up2_sar, up3_sar, output_sar = model_sar(sar)
            up1_rgb, up2_rgb, up3_rgb, output_rgb = model_rgb(rgb)
            loss = criterion_ce(output_rgb, lab) + criterion_focal(output_rgb, lab)
            up3_sar = m(up3_sar).data.cpu().numpy()
            up3_rgb = m(up3_rgb).data.cpu().numpy()
            loss += T ** 2 * np.mean(-np.sum((up3_sar / T) * np.log((up3_rgb / T) + 1e-5), axis=1))
            # loss = 0.1 * (criterion_L1(up1_rgb, up1_sar) + criterion_L1(up1_rgb, up1_sar)) + criterion_focal(output_sar, lab) + criterion_ce(output_sar, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))

        # validation
        if epoch % 5 == 0:
            evaluator.reset()
            model_rgb.eval()
            #     model_rgb_val.eval()
            vbar = tqdm.tqdm(val_loader, desc='\r')
            #     # pred_sum = np.array([])
            #     # lab_sum = np.array([])
            loss_sum = 0.0
            # zeros = torch.zeros([1,2,512,512])
            #     # val_num = len(val_loader)
            for i, sample in enumerate(vbar):
                sar, lab, rgb = sample
                sar, lab, rgb = sar.cuda(), lab.cuda(), rgb.cuda()
                with torch.no_grad():
                    # up1_out, up2_out, up3_out, output_sar = model_sar(sar)
                    up1_out, up2_out, up3_out, output_rgb = model_rgb(rgb)
                loss = criterion_focal(output_rgb, lab) + criterion_ce(output_rgb, lab)
                loss_sum += loss.item()
                vbar.set_description('val loss: %.5f' % (loss_sum / (i + 1)))
                pred = output_rgb.data.cpu().numpy()
                lab = lab.cpu().numpy().squeeze()
                pred = np.argmax(pred, axis=1).squeeze()
                # print(lab.shape, pred.shape)
                evaluator.add_batch(lab, pred)
                # evaluator.acc_f1(lab, pred)
                # pred = np.array(pred, dtype=np.uint8).reshape(-1)
                # lab = np.array(lab, dtype=np.uint8).reshape(-1)
                # print(f1_score(pred, lab))
                # pred_sum = np.append(pred_sum, pred)
                # lab_sum = np.append(lab_sum, lab)
            acc = evaluator.Pixel_Accuracy()
            # acc_per_class, acc = evaluator.Pixel_Accuracy_Class()
            iou = evaluator.Mean_Intersection_over_Union()[1]
            # iou_per_class, iou = evaluator.meanIntersectionOverUnion()
            f1 = evaluator.BcF1()  # f1_score(lab_sum, pred_sum)
            # f1_per_class, F1 = evaluator.BcF1()

            message = "epoch: %d, acc: %3f, iou: %3f, f1: %3f\n" % (epoch, acc, iou, f1)
            print(message)
            with open("sar_project/+deeplog_rgb+sar->rgb.txt", "a") as f:
                f.write(message + '\n')
                # f.write('acc:' + str(acc_per_class) + '\n')
                # f.write('iou:' + str(iou_per_class) + '\n')
                # f.write('f1:' + str(f1_per_class) + '\n')
            # acc, iou, f1, kappa = evaluator.compute_miou(lab, pred, 10)
            # message = "epoch: %d, acc: %3f, iou: %3f, f1: %3f\n" % (
            #     epoch, acc, iou, f1)
            # print(message)
            # weights_path = "sar_project/checkpoint/segmentation/deeplab_sar_block1&3+3"
            # mkdir(weights_path)
            # torch.save(model_sar.module.state_dict(),
            #            os.path.join(weights_path, "%d.pth" % (epoch)))
            # with open("sar_project/+deeplog_sar_block1&3+3.txt", "a") as f:
            #     f.write(message)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAR Segmentation Algorithm')
    # 调整学习率的lr_scheduler机制
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: auto)')
    parser.add_argument('--epochs', type=int, default=151, metavar='N',
                        help='number of epochs to train (default: 40)')

    args = parser.parse_args(sys.argv[1:])

    sar_train(args)
