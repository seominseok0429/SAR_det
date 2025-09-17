import torch
import torch.nn as nn
import torch.nn.functional as F
from sar_project.models.segmentation.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from sar_project.models.segmentation.backbone import build_backbone
from sar_project.models.segmentation.domain_align_AlignSeg import DomainAlign_AlignSeg
from sar_project.models.segmentation.contrastive_module import Contrast_RS
from sar_project.models.segmentation.attention_module import DAHead


class CMC(nn.Module):
    def __int__(self):