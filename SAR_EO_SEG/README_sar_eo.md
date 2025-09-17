# DHFA-Net
This is the official repo for our work 'Distilled Heterogeneous Feature Alignment Network for SAR Image Semantic Segmentation' (GRSL).
## Setup
We built and ran the repo with CUDA 11.0, Python 3.7.13, and Pytorch 1.13.1. For using this repo, we recommend creating a virtual environment by [Anaconda](https://www.anaconda.com/download). Please open a terminal in the root of the repo folder for running the following commands and scripts.
```pytorch
conda env create -f environment.yml
conda activate pytorch
```
## Pre-trained models

Dataset(s) | Model Name | Acc | mIoU | F1
---- | ----- | ------ | ------ | ------ 
[SpaceNet6](https://spacenet.ai/sn6-challenge/) | DHFA-Net_Binary_Seg | 0.9845 | 0.8272 | 0.9142
[SEN12MS](https://arxiv.org/pdf/1906.07789v1.pdf) | DHFA-Net_MultiClass_Seg | 0.8268 | 0.6641 | 0.7635 

## Model Training
For the SpaceNet6 dataset, please set ```num_classes=2``` in the ```train_myalignment.py```, while for the SEN12MS dataset, please set ```num_class=10```. The DHFA-Net could be trained simply by the following command:
```pytorch
pyhton3 train_myalignment.py
```
## Prediction
DHFA-Net could be tested by simply applying the following command:
```pytorch
python3 test_myaligment.py
```
And the segmentation maps are saved at ```/results/```.

## Citation
```pytorch
@ARTICLE{10175569,
  author={Gao, Mengyu and Xu, Jiping and Yu, Jiabin and Dong, Qiulei},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Distilled Heterogeneous Feature Alignment Network for SAR Image Semantic Segmentation}, 
  year={2023},
  volume={20},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2023.3293160}
}
```
