# -*- coding: utf-8 -*-
"""
MiniFASNet Models for Face Anti-Spoofing
Compatible with pre-trained Minivision models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MiniFASNetV1(nn.Module):
    """
    MiniFASNet V1: Lightweight anti-spoofing network
    For pre-trained models: 2.7_80x80_MiniFASNetV2.pth variant
    """
    def __init__(self, conv6_kernel=16):
        super(MiniFASNetV1, self).__init__()
        self.conv1 = ConvBlock(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv6 = ConvBlock(64, 128, kernel_size=conv6_kernel, stride=1, padding=0)
        
        self.flatten = Flatten()
        self.fc = nn.Linear(128, 2)  # Binary classification: fake (0) or real (1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MiniFASNetV2(nn.Module):
    """
    MiniFASNet V2: Enhanced anti-spoofing network
    Similar to V1 but with additional refinements
    """
    def __init__(self, conv6_kernel=16):
        super(MiniFASNetV2, self).__init__()
        self.conv1 = ConvBlock(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv6 = ConvBlock(64, 128, kernel_size=conv6_kernel, stride=1, padding=0)
        
        self.flatten = Flatten()
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DepthWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseBlock, self).__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.prelu_dw = nn.PReLU(in_channels)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_channels)
        self.prelu_proj = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.prelu_dw(x)
        x = self.project(x)
        x = self.bn_proj(x)
        x = self.prelu_proj(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels // reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return orig * x

class MiniFASNetV1SE(nn.Module):
    """
    MiniFASNet V1 with Squeeze-and-Excitation blocks (compatible Minivision)
    """
    def __init__(self, conv6_kernel=16):
        super(MiniFASNetV1SE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.prelu1 = nn.PReLU(32)

        # Block 2
        self.conv2_dw = DepthWiseBlock(32, 32, stride=1)
        self.conv_23 = DepthWiseBlock(32, 64, stride=2)
        self.conv_3 = nn.Sequential(
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),
            DepthWiseBlock(64, 64, stride=1),
            SEModule(64)
        )
        self.conv_34 = DepthWiseBlock(64, 128, stride=2)
        self.conv_4 = nn.Sequential(
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            DepthWiseBlock(128, 128, stride=1),
            SEModule(128)
        )
        self.conv_45 = DepthWiseBlock(128, 256, stride=2)
        self.conv_5 = nn.Sequential(
            DepthWiseBlock(256, 256, stride=1),
            DepthWiseBlock(256, 256, stride=1),
            SEModule(256)
        )
        self.conv_6_sep = nn.Conv2d(256, 512, kernel_size=conv6_kernel, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.prelu6 = nn.PReLU(512)
        self.conv_6_dw = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6_dw = nn.BatchNorm2d(512)
        self.prelu6_dw = nn.PReLU(512)
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.bn6(x)
        x = self.prelu6(x)
        x = self.conv_6_dw(x)
        x = self.bn6_dw(x)
        x = self.prelu6_dw(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MiniFASNetV2SE(nn.Module):
    """
    MiniFASNet V2 with Squeeze-and-Excitation blocks
    Most advanced variant
    """
    def __init__(self, conv6_kernel=16):
        super(MiniFASNetV2SE, self).__init__()
        self.conv1 = ConvBlock(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv6 = ConvBlock(64, 128, kernel_size=conv6_kernel, stride=1, padding=0)
        
        self.flatten = Flatten()
        self.fc = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
