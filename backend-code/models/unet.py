#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from .network_blocks import DoubleConvBlock, AttentionBlock2d
from .esa_modules import ESA_blcok
# 可视化模型结构 以下是自己的修改
from torchvision import models
import torchsummary as summary
# 一下是自己写得
# from network_blocks import DoubleConvBlock, AttentionBlock2d
# from esa_modules import ESA_blcok

class UNet(nn.Module):

    def __init__(self, num_classes, in_channels, is_esa=False, is_grid=False):
        super().__init__()
        nb_filters = [32, 64, 128, 256, 512]
        self.is_esa = is_esa
        self.is_grid = is_grid
        # self.is_ese_1_0 = ESA_blcok(nb_filters[1])
        # self.is_ese_2_0 = ESA_blcok(nb_filters[2])
        # self.is_ese_3_0 = ESA_blcok(nb_filters[3])
        self.is_ese_4_0 = ESA_blcok(nb_filters[4])

        # 标准的2倍上采样和下采样，因为没有可以学习的参数，可以共享
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if self.is_esa:
            # 下采样的模块 (self, in_channels, middle_channels, out_channels, dropout=False, p=0.2) k=3, s=1, p=1
            self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])  # (3, 32, 32)
            self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])  # (32, 64, 64)
            self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])  # (64, 128, 128)
            self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])  # (128, 256, 256)
            self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])  # (256, 512, 512)

            # 上采样的模块
            self.conv3_1 = DoubleConvBlock(nb_filters[4] + nb_filters[3], nb_filters[3], nb_filters[3])
            self.conv2_2 = DoubleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2], nb_filters[2])
            self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1])
            self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])
        else:
            # 下采样的模块
            self.conv0_0 = DoubleConvBlock(in_channels, nb_filters[0], nb_filters[0])
            self.conv1_0 = DoubleConvBlock(nb_filters[0], nb_filters[1], nb_filters[1])
            self.conv2_0 = DoubleConvBlock(nb_filters[1], nb_filters[2], nb_filters[2])
            self.conv3_0 = DoubleConvBlock(nb_filters[2], nb_filters[3], nb_filters[3])
            self.conv4_0 = DoubleConvBlock(nb_filters[3], nb_filters[4], nb_filters[4])

            # 上采样的模块
            self.conv3_1 = DoubleConvBlock(nb_filters[4] + nb_filters[3], nb_filters[3], nb_filters[3])
            self.conv2_2 = DoubleConvBlock(nb_filters[3] + nb_filters[2], nb_filters[2], nb_filters[2])
            self.conv1_3 = DoubleConvBlock(nb_filters[2] + nb_filters[1], nb_filters[1], nb_filters[1])
            self.conv0_4 = DoubleConvBlock(nb_filters[1] + nb_filters[0], nb_filters[0], nb_filters[0])

        # 最后接一个Conv计算在所有类别上的分数
        self.final = nn.Conv2d(nb_filters[0], num_classes, kernel_size=1, stride=1)

        if self.is_grid:
            self.attention3_1 = AttentionBlock2d(nb_filters[3], nb_filters[4])  # 256 512
            self.attention2_2 = AttentionBlock2d(nb_filters[2], nb_filters[3])  # 128 256
            self.attention1_3 = AttentionBlock2d(nb_filters[1], nb_filters[2])  # 64 128
            self.attention0_4 = AttentionBlock2d(nb_filters[0], nb_filters[1])  # 32 64

    def forward(self, input):
        # 下采样编码
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.down(x0_0))

        x2_0 = self.conv2_0(self.down(x1_0))

        x3_0 = self.conv3_0(self.down(x2_0))

        x4_0 = self.conv4_0(self.down(x3_0))
        if self.is_esa:
            x4_0 = x4_0 + self.is_ese_4_0(x4_0)

        if self.is_grid:
            # 特征融合并进行上采样解码，使用concatenate进行特征融合
            x3_1 = self.conv3_1(torch.cat([self.attention3_1(x3_0, x4_0), self.up(x4_0)], dim=1))
            x2_2 = self.conv2_2(torch.cat([self.attention2_2(x2_0, x3_1), self.up(x3_1)], dim=1))
            x1_3 = self.conv1_3(torch.cat([self.attention1_3(x1_0, x2_2), self.up(x2_2)], dim=1))
            x0_4 = self.conv0_4(torch.cat([self.attention0_4(x0_0, x1_3), self.up(x1_3)], dim=1))
        else:
            # 特征融合并进行上采样解码，使用concatenate进行特征融合
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        # 计算每个类别上的得分
        output = self.final(x0_4)

        return output

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# unet = UNet(3, 3, True, True).to(device)
# summary.summary(unet,(3, 512, 512))
