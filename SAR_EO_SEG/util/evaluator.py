'''
Author: your name
Date: 2021-01-13 16:31:28
LastEditTime: 2021-01-13 18:26:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/util/valor.py
'''

import numpy as np
import glob
import cv2
# import natsort
from tqdm import tqdm
import numpy as np
import os
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import cohen_kappa_score


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        self.acc = np.zeros((self.num_class))
        self.f1 = np.zeros((self.num_class))


    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        # acc = np.zeros(10)
        # for i in range(10):
        #     acc[i] = np.diag(self.confusion_matrix[i,:,:]).sum()/ \
        #         self.confusion_matrix[i,:,:].sum()
        # Acc = acc.sum()/10
        return Acc

    def Pixel_Accuracy_Class(self):
        # Acc = np.diag(self.confusion_matrix) / \
        #     self.confusion_matrix.sum(axis=1)
        # Acc = np.nanmean(Acc)

        Acc_per_class = np.zeros((10))
        for c in range(self.num_class):
            Acc_per_class[c] = self.confusion_matrix[c, c] / self.confusion_matrix[c, :].sum()
        Acc = np.nanmean(Acc_per_class)
        return Acc_per_class, Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(np.delete(MIoU, 1, axis=0))
        # iou = np.zeros((10,2))
        # miou = np.zeros(10)
        # for i in range(10):
        #     iou[i,:] = np.diag(self.confusion_matrix[i,:,:]) / (
        #         np.sum(self.confusion_matrix[i,:,:], axis=1) + np.sum(self.confusion_matrix[i,:,:], axis=0) -
        #         np.diag(self.confusion_matrix[i,:,:])
        #     )
        #     miou = np.nanmean(iou)
        # MIoU = miou.sum()/10
        return MIoU

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        # intersection = np.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        # union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
        #     self.confusion_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        # IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        # mIoU = np.nanmean(IoU)  # 求各类别IoU的平均

        miou_per_class = np.zeros(10)
        for c in range(self.num_class):
            intersection = self.confusion_matrix[c, c]
            union = np.sum(self.confusion_matrix[c, :]) + np.sum(self.confusion_matrix[:, c]) - intersection
            miou_per_class[c] = intersection / union
        miou = np.nanmean(miou_per_class)
        return miou_per_class, miou

    def BcF1(self):
        Precision = self.confusion_matrix[1, 1] / \
           (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        Recall = self.confusion_matrix[1, 1] / \
            (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        return F1

        # f1_per_class = np.zeros(10)
        # for c in range(self.num_class):
        #     fenzi = 2 * self.confusion_matrix[c, c]
        #     fenmu = np.sum(self.confusion_matrix[c, :]) + np.sum(self.confusion_matrix[:, c])
        #     f1_per_class[c] = fenzi / fenmu
        # F1 = np.nanmean(f1_per_class)
        # return f1_per_class, F1

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def cal_confu_matrix(self, gt_image, pre_image, class_num):
        confu_list = []
        for i in range(class_num):
            c = Counter(pre_image[np.where(gt_image == i)])
            single_row = []
            for j in range(class_num):
                single_row.append(c[j])
            confu_list.append(single_row)
        confu_matrix = np.array(confu_list) + 0.0001
        # return confu_matrix.astype(np.int32)
        return confu_matrix

    def metrics(self, gt_image, pre_image, class_num):
        '''
        :param confu_mat: 总的混淆矩阵
        backgound：是否干掉背景
        :return: txt写出混淆矩阵, precision，recall，IOU，f-score
        '''
        confu_mat_total = self.cal_confu_matrix(gt_image, pre_image, class_num)
        print(confu_mat_total)
        class_num = confu_mat_total.shape[0]
        confu_mat = confu_mat_total.astype(np.float32) + 0.0001
        col_sum = np.sum(confu_mat, axis=1)  # 按行求和
        raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

        '''计算各类面积比，以求OA值'''
        oa = 0
        for i in range(class_num):
            oa = oa + confu_mat[i, i]
        oa = oa / confu_mat.sum()

        Acc = []
        # Acc = np.diag(confu_mat) / \
        #       confu_mat.sum(axis=1)
        Acc = np.diag(confu_mat).sum() / \
              confu_mat.sum(axis=1)
        Acc = np.nanmean(Acc)

        '''Kappa'''
        pe_fz = 0
        for i in range(class_num):
            pe_fz += col_sum[i] * raw_sum[i]
        pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
        kappa = (oa - pe) / (1 - pe)

        # 将混淆矩阵写入excel中
        TP = []  # 识别中每类分类正确的个数
        FN = []
        FP = []

        for i in range(class_num):
            TP.append(confu_mat[i, i])

        # 计算f1-score
        TP = np.array(TP)
        FN = col_sum - TP
        FP = raw_sum - TP

        # 计算并写出precision，recall, f1-score，f1-m以及mIOU

        f1_m = []
        iou_m = []
        for i in range(class_num):
            # 写出f1-score
            f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
            f1_m.append(f1)
            iou = TP[i] / (TP[i] + FP[i] + FN[i])
            iou_m.append(iou)

        f1_m = np.array(f1_m)
        iou_m = np.array(iou_m)
        f1_m = f1_m.sum()/len(f1_m)
        iou_m = iou_m.sum()/len(iou_m)
        return Acc, iou_m, f1_m


    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # confusion_matrix = np.zeros((10, 2, 2))
        # gt_new = np.zeros((256, 256))
        # pre_new = np.zeros((256, 256))
        # for k in range(10):
        #     for i in range(256):
        #         for j in range(256):
        #             if gt_image[i, j] == k:
        #                 gt_new[i, j] = 1
        #             else:
        #                 gt_new[i,j] = 0
        #             if pre_image[i, j] == k:
        #                 pre_new[i, j] = 1
        #             else:
        #                 pre_new[i, j] = 0
        #     confusion_matrix[k,:,:] = self._generate_matrix(gt_new, pre_new) + 0.0001
        #     gt_new = np.zeros((256, 256))
        #     pre_new = np.zeros((256, 256))
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        # self.confusion_matrix += self.cal_confu_matrix(gt_image, pre_image, self.num_class)
        # self.confusion_matrix += confusion_matrix
        # self.confusion_matrix += self.fast_hist(gt_image.flatten(), pre_image.flatten(), self.num_class)

    def fast_hist(self, a, b, n):
        k = (a >= 0) & (a < 10)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,n)
        # return confusion_matrix(a, b)

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_miou(self, gt_image, pre_image, num_classes):

        # acc = precision_score(gt_image.astype('int').flatten(), pre_image.flatten(), average='micro')
        # f1 = f1_score(gt_image.astype('int').flatten(), pre_image.flatten(), labels=labels, average='micro')
        # hist = np.zeros((num_classes, num_classes))
        # hist += self.fast_hist(gt_image.astype('int').flatten(), pre_image.flatten(), num_classes)
        print(self.confusion_matrix)
        mIoUs = self.per_class_iu(self.confusion_matrix + 0.0001)
        MIoU = np.nanmean(mIoUs)
        acc = np.nanmean(self.acc)
        f1 = np.nanmean(self.f1)
        with open("sar_project/+deeplog_sar_RSDNet_sen12ms.txt", "a") as f:
            f.write('acc:'+str(self.acc)+'\n')
            f.write('miou:'+str(mIoUs)+'\n')
            f.write('f1'+str(self.f1)+'\n')
        return acc, MIoU, f1

    def acc_f1(self, gt_image, pre_image):
        labels = [0,1,2,3,4,5,6,7,8,9]

        # labels = [0, 1]
        self.acc += precision_score(gt_image.astype('int').flatten(), pre_image.flatten(), labels=labels, average=None)
        self.f1 += f1_score(gt_image.astype('int').flatten(), pre_image.flatten(), labels=labels, average=None)
        # self.kappa += cohen_kappa_score(gt_image.astype('int').flatten(), pre_image.flatten())


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))
        self.acc = np.zeros((self.num_class))
        self.f1 = np.zeros((self.num_class))







if __name__ == "__main__":
    images = natsort.natsorted(glob.glob('./reslut/mcfinet+seblock/*.tif'))
    evaluator = Evaluator(6)
    evaluator.reset()
    for i in tqdm(range(len(images))):

        img = cv2.imread(images[i])              # 3 channels
        labpath = os.path.join(
            "assess_classification_reference_implementation/assess_classification_2D_ISPRS_web/ISPRS_semantic_labeling_data_upload/gts/",
            os.path.basename(images[i]).replace("_class", ""))
        imgmask = rgb2mask(img)
        lab = cv2.imread(labpath)
        labmask = rgb2mask(lab)
        evaluator.add_batch(imgmask, labmask)

    mIOU = evaluator.Mean_Intersection_over_Union()
    print(mIOU)
