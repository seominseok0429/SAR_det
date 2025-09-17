'''
Author: wdj
Date: 2021-01-13 15:28:20
LastEditTime: 2021-01-13 15:30:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/util/loss.py
'''
import torch
import torch.nn as nn
import math

class SegmentationLosses(object):
    def __init__(self, weight=None, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode_l='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode_l == 'ce':
            return self.CrossEntropyLoss
        elif mode_l == 'focal':
            return self.FocalLoss
        elif mode_l == 'supervise':
            return self.RGBLoss
        elif mode_l == 'L1':
            return self.L1Loss
        elif mode_l == 'recon':
            return self.recon
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=255)
        if self.cuda:
            criterion = criterion.cuda()

        # if mode_l == 'ce':
        loss = criterion(logit, target.long())

        return loss

    def RGBLoss(self, logit, target, mode_l):
        n, c, h, w = logit.size()
        score, score_1, score_2 = 0,0,0
        # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=255)
        # if self.cuda:
        #     criterion = criterion.cuda()
        if mode_l == 'supervise':
            # loss = torch.sqrt(torch.sum(logit-target)**2)
            score1 = torch.cosine_similarity(logit, target, dim=2)
            score2 = torch.cosine_similarity(logit, target, dim=3)
            loss = (torch.sum(score1)+torch.sum(score2))/2
            loss = -loss/(c*n)
            loss = math.exp(loss)

            # for batch in range(n):
            #    logit_3 = torch.squeeze(logit[batch,:,:,:], dim=0)
            #    target_3 = torch.squeeze(target[batch,:, :, :], dim=0)
            #    for channel in range(c):
            #        logit_2 = torch.squeeze(logit_3[channel, :, :], dim=0)
            #        target_2 = torch.squeeze(target_3[channel, :, :], dim=0)
            #        # logit_1 = logit_2.resize(1024)
            #        # target_1 = target_2.resize(1024)
            #        score_1 += torch.cosine_similarity(logit_2, target_2, dim=0)
            #        score_2 += torch.cosine_similarity(logit_2, target_2, dim=1)
            #        score = (torch.sum(score_1)+torch.sum(score_2))/2
            # loss = -score/(c*n)

        return loss #math.exp(loss)


    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
    def L1Loss(self, logit, target, lamda = 0.01):
        n,c,h,w = logit.size()
        loss = torch.sum(abs(logit-target))
        loss = lamda * loss
        # loss = torch.tensor(loss)
        # loss.requires_grad_(True)
        return torch.log(loss)

    def recon(self, logit, target, lamda = 0.001):
        loss = torch.sum((logit-target)**2)
        loss = lamda * loss
        return torch.log(loss)