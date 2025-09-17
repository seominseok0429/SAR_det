'''
Author: wdj
Date: 2021-01-07 10:46:11
LastEditTime: 2021-01-13 17:19:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /pytorch-CycleGAN-and-pix2pix-master/util/utils.py
'''
# losses: same format as |losses| of plot_current_losses
import os
import numpy as np
import os
import sys
import ntpath
import time


class Utils():
    """This class includes several functions that can save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        self.opt = opt
        self.log_name = os.path.join(
            opt.checkpoints_dir, opt.name, 'loss_log.txt')

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
            epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    @classmethod
    def mkdirs(self, paths):
        """create empty directories if they don't exist

        Parameters:
            paths (str list) -- a list of directory paths
        """
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                self.mkdir(path)
        else:
            self.mkdir(paths)

    @classmethod
    def mkdir(self, path):
        """create a single empty directory if it didn't exist

        Parameters:
            path (str) -- a single directory path
        """
        if not os.path.exists(path):
            os.makedirs(path)
