'''
Author: your name
Date: 2021-01-13 14:57:42
LastEditTime: 2021-01-20 20:09:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/datasets/trainsar.py
'''
#!/usr/bin/env python

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
import tqdm
import gdal


# import solaris as sol

# from data.sarlab_dataset import SARLABDataSet
from sar_project.data.sarrgblab_dataset import SARLABDataSet
from sar_project.data.sen12ms_dataset import SEN12MS
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
# 指定代码在服务器董某块显卡上跑，一般放在整段代码最前面，即import torch前
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


def sar_train(args):
    # opt = TrainOptions().parse()
    # mkdir = Utils().mkdir
    train_set = SARLABDataSet(mode="train")
    val_set = SARLABDataSet(mode="val")
    # train_set = SEN12MS(mode="train")
    # val_set = SEN12MS(mode="val")
    train_loader = DataLoader(train_set, batch_size=4,
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, drop_last=True, num_workers=4)

    model = DeepLab_myalign(mode='rgb', num_classes=2)

    train_params = model.parameters()

    optimizer = torch.optim.Adam(train_params, lr=args.lr)

    criterion_focal = SegmentationLosses().build_loss(mode_l="focal")
    criterion_ce = SegmentationLosses().build_loss(mode_l="ce")
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                             args.epochs, len(train_loader))

    model = torch.nn.DataParallel(model,  device_ids=[0, 1])
    patch_replication_callback(model)
    model = model.cuda()

    # rgb_weights = torch.load('sar_project/checkpoint/segmentation/deeplab_rgb_seg/49.pth')
    # model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in rgb_weights.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # 3. load the new state dict
    # model.load_state_dict(model_dict)
    # model.eval()

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
    for epoch in range(args.epochs):
        train_loss = 0.0
        model.train()
        tbar = tqdm.tqdm(train_loader)
        for i, sample in enumerate(tbar):
            sar, lab, rgb = sample
            sar, lab, rgb = sar.cuda(), lab.cuda(), rgb.cuda()
            scheduler(optimizer, i, epoch)
            optimizer.zero_grad()
            up1, up2, up3, output = model(rgb)
            loss = criterion_focal(output, lab)+criterion_ce(output, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))
        # validation
        if epoch % 10 == 0:
            evaluator.reset()
            model.eval()
            vbar = tqdm.tqdm(val_loader, desc='\r')
            # pred_sum = np.array([])
            # lab_sum = np.array([])
            loss_sum = 0.0
            # val_num = len(val_loader)
            for i, sample in enumerate(vbar):
                sar, lab, rgb = sample
                sar, lab, rgb = sar.cuda(), lab.cuda(), rgb.cuda()
                with torch.no_grad():
                    up1, up2, up3, output = model(rgb)
                loss = criterion_focal(output, lab) + \
                    criterion_ce(output, lab)
                loss_sum += loss.item()
                vbar.set_description('val loss: %.5f' % (loss_sum / (i + 1)))
                pred = output.data.cpu().numpy()
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
            iou = evaluator.Mean_Intersection_over_Union()[1]
            # class_acc = evaluator.Pixel_Accuracy_Class()
            # iou = evaluator.meanIntersectionOverUnion()
            f1 = evaluator.BcF1()  # f1_score(lab_sum, pred_sum)
            # acc, iou, f1 = evaluator.compute_miou(lab, pred, 10)
            message = "epoch: %d, acc: %3f, iou: %3f, class_acc: %3f\n" % (
                epoch, acc, iou, f1)
            print(message)

            # if min_loss > (loss_sum/val_num):
            #     min_loss = loss_sum/val_num
            #     torch.save(model.module.state_dict(), os.path.join(
            #         "./model_weights", "min_loss.pth"))
            # if max_f1<f1:
            #     max_f1=f1
            weights_path = "sar_project/checkpoint/segmentation/deeplab_rgb_AlignSeg4"
            mkdir(weights_path)
            torch.save(model.module.state_dict(),
                       os.path.join(weights_path, "%d.pth" % epoch))
            with open("sar_project/+deeplog_rgb_AlignSeg4.txt", "a") as f:
                f.write(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAR Segmentation Algorithm')
    # 调整学习率的lr_scheduler机制
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: auto)')
    parser.add_argument('--epochs', type=int, default=51, metavar='N',
                        help='number of epochs to train (default: 40)')

    args = parser.parse_args(sys.argv[1:])

    sar_train(args)
