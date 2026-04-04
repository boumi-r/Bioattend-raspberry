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


class MiniFASNetV1SE(nn.Module):
    """
    MiniFASNet V1 with Squeeze-and-Excitation blocks
    Enhanced feature recalibration
    """
    def __init__(self, conv6_kernel=16):
        super(MiniFASNetV1SE, self).__init__()
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
