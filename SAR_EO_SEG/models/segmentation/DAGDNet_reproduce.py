import torch
import torch.nn as nn
import torch.nn.functional as F
from sar_project.models.segmentation.backbone import resnet

class DAGDNet_Student(nn.Module):
    def __int__(self, num_classes, backbone='resnet-34', output_stride=16):
        super(DAGDNet_Student, self).__int__()
        self.student_first = nn.Sequential(nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
                                              nn.BatchNorm(32),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.student_layer1 = Layer(in_channel=32, out_channel=64, stride=1, layers=3)
        self.teacher_layer2 = Layer(in_channel=64, out_channel=128, stride=2, layers=4)
        self.teacher_layer3 = Layer(in_channel=128, out_channel=256, stride=2, layers=6)
        self.teacher_layer4 = Layer(in_channel=256, out_channel=512, stride=1, layers=3)

        self.generator1 = Generator(in_channel=64)
        self.generator2 = Generator(in_channel=128)
        self.generator2 = Generator(in_channel=256)
        self.generator2 = Generator(in_channel=512)

        self.fc = nn.Linear(2048, 10240)

    def forward(self, x):
        x_0 = self.student_first(x)
        x_1_pre = self.student_layer1(x_0)
        x_1_EO, x_1_SAR, x_1_share, x_1 = self.generator1(x_1_pre)

        x_2_pre = self.student_layer2(x_1)
        x_2_EO, x_2_SAR, x_2_share, x_2 = self.generator2(x_2_pre)

        x_3_pre = self.student_layer3(x_2)
        x_3_EO, x_3_SAR, x_3_share, x_3 = self.generator3(x_3_pre)

        x_4_pre = self.student_layer4(x_3)
        x_4_EO, x_4_SAR, x_4_share, x_4 = self.generator4(x_4_pre)

        x_output = self.fc(x_4.view(128, 2048))
        return x_1_EO, x_1_SAR, x_1_share, x_2_EO, x_2_SAR, x_2_share, x_3_EO, x_3_SAR, x_3_share, x_4_EO, x_4_SAR, x_4_share, x_output.view(128, 10, 32, 32)

class Generator(nn.Module):
    def __int__(self, in_channel):
        super(Generator, self).__int__()
        self.conv_EO = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.conv_SAR = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.conv_share = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)

        self.conv = nn.Conv2d(3 * in_channel, in_channel, kernel_size=1, stride=1)

    def forward(self, x):
        x_EO = self.conv_EO(x)
        x_SAR = self.conv_SAR(x)
        x_share = self.conv_share(x)
        concat = torch.cat((x_EO, x_SAR), dim=1)
        concat = torch.cat((concat, x_share), dim=1)
        output = self.conv(concat)
        return x_EO, x_SAR, x_share, output

class DAGDNet_Teacher(nn.Module):
    def __int__(self, num_classes, backbone='resnet-34', output_stride=16):
        super(DAGDNet, self).__int__()
        self.teacher_EO_first = nn.Sequential(nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
                                              nn.BatchNorm(32),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.teacher_SAR_first = nn.Sequential(nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                                              nn.BatchNorm(32),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.teacher_EO_layer1 = Layer(in_channel=32, out_channel=64, stride=1, layers=3)
        self.teacher_EO_layer2 = Layer(in_channel=64, out_channel=128, stride=2, layers=4)
        self.teacher_EO_layer3 = Layer(in_channel=128, out_channel=256, stride=2, layers=6)
        self.teacher_EO_layer4 = Layer(in_channel=256, out_channel=512, stride=1, layers=3)

        self.teacher_SAR_layer1 = Layer(in_channel=32, out_channel=64, stride=1, layers=3)
        self.teacher_SAR_layer2 = Layer(in_channel=64, out_channel=128, stride=2, layers=4)
        self.teacher_SAR_layer3 = Layer(in_channel=128, out_channel=256, stride=2, layers=6)
        self.teacher_SAR_layer4 = Layer(in_channel=256, out_channel=512, stride=1, layers=3)

        self.IF_GFM_1 = IF_GFM(channel=64)
        self.IF_GFM_2 = IF_GFM(channel=128)
        self.IF_GFM_3 = IF_GFM(channel=256)
        self.IF_GFM_4 = IF_GFM(channel=512)

        self.fc_EO = nn.Linear(2048, 10240)
        self.fc_SAR = nn.Linear(2048, 10240)
        self.fc_share = nn.Linear(2048, 10240)

    def forward(self, rgb, sar):
        X_0 = self.teacher_EO_first(rgb)
        Y_0 = self.teacher_SAR_first(sar)

        X_1_pre = self.teacher_EO_layer1(X_0)
        Y_1_pre = self.teacher_SAR_layer1(Y_0)
        X_1, Y_1, share_1 = self.IF_GFM_1(X_1_pre, Y_1_pre)

        X_2_pre = self.teacher_EO_layer2(X_1)
        Y_2_pre = self.teacher_SAR_layer2(Y_1)
        X_2, Y_2, share_2 = self.IF_GFM_2(X_2_pre, Y_2_pre)

        X_3_pre = self.teacher_EO_layer3(X_2)
        Y_3_pre = self.teacher_SAR_layer3(Y_2)
        X_3, Y_3, share_3 = self.IF_GFM_3(X_3_pre, Y_3_pre)

        X_4_pre = self.teacher_EO_layer4(X_3)
        Y_4_pre = self.teacher_SAR_layer4(Y_3)
        X_4, Y_4, share_4 = self.IF_GFM_4(X_4_pre, Y_4_pre)

        X_output = self.fc_EO(X_4.view(128, 2048))
        Y_output = self.fc_EO(Y_4_4.view(128, 2048))
        share_output = self.fc_EO(share_4.view(128, 2048))

        return X_1, Y_1, share_1, X_2, Y_2, share_2, X_3, Y_3, share_3, X_4, Y_4, share_4, X_output.view(128, 10, 32, 32), Y_output.view(128, 10, 32, 32), share_output.view(128, 10, 32, 32)

class Layer(nn.Module):
    def __int__(self, in_channel, out_channel, stride, num_layers):
        super(Layer).__init__()
        layers = []
        layers.append(Basic_block(in_channel=in_channel, out_channel=out_channel, stride=stride))
        for i in range(num_layers-1):
            layers.append(Basic_block(in_channel=out_channel, out_channel=out_channel, stride=1))
        self.res_block = nn.Sequential(*layers)
    def forward(self, x):
        output = self.res_block(x)
        return output

class Basic_block(nn.Module):
    def __int__(self, in_channel, out_channel, stride):
        super(Basic_bloc, self).__int__()
        self.outchannel = out_channel
        self.block1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())
    def forward(self, x):
        x_1 = self.block1(x)
        x_2 = self.block2(x_1)
        if x.size()[1] == self.outchannel:
            output = x + x_2
        else:
            output = x_2
        return output

class IF_GFM(nn.Module):
    def __int__(self, size, channels):
        self.global_pooling = nn.AvgPool2d(kernel_size=size, stride=1)
        self.weight_EO = nn.Sequential(nn.Linear(channels, channels/2),
                                       nn.Linear(channels/2, channels),
                                       nn.Sigmoid())
        self.weight_SAR = nn.Sequential(nn.Linear(channels, channels / 2),
                                       nn.Linear(channels / 2, channels),
                                       nn.Sigmoid())
        self.conv_EO = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.conv_SAR = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.conv_share = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, X, Y):
        X_global = self.global_pooling(X)
        Y_global = self.global_pooling(Y)
        X_global = X_global.view(128, 64)
        Y_global = Y_global.view(128, 64)
        X_weight = self.weight_EO(X_global)
        Y_weight = self.weight_SAR(Y_global)
        X_weight_verse = torch.from_numpy(np.ones(128, 64)) - X_weight
        Y_weight_verse = torch.from_numpy(np.ones(128, 64)) - Y_weight
        X_weight = X_weight.view(128, 64, 1, 1)
        Y_weight = Y_weight.view(128, 64, 1, 1)
        X_weight_verse = X_weight_verse.view(128, 64, 1, 1)
        Y_weight_verse = Y_weight_verse.view(128, 64, 1, 1)

        Y_specific = self.conv_SAR(X_weight * Y)
        X_specific = self.conv_EO(Y_weight * X)
        Y_share = X_weight_verse * Y
        X_share = Y_weight_verse * X
        share = self.conv_share(torch.cat((X_share, Y_share), dim=1))

        return X_specific, Y_specific, share


