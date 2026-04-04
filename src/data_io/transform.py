# -*- coding: utf-8 -*-
"""
Image preprocessing transforms for anti-spoofing
Compatible with MiniFASNet input requirements
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms_list):
        self.transforms = transforms_list
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    """Convert numpy array (H, W, C) to tensor (C, H, W)"""
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # Ensure image is in BGR format for consistency
            if len(img.shape) == 3 and img.shape[2] == 3:
                # BGR -> RGB not needed, keep BGR for compatibility
                pass
            img = torch.from_numpy(img).float()
            # Convert H, W, C -> C, H, W
            if img.dim() == 3:
                img = img.permute(2, 0, 1)
            # Normalize to [0, 1]
            img = img / 255.0
            return img
        return img


class Normalize:
    """Normalize tensor with mean and std"""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return (img - self.mean) / self.std
        return img


class CenterCrop:
    """Center crop image to target size"""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            th, tw = self.size
            x = (w - tw) // 2
            y = (h - th) // 2
            return img[y:y+th, x:x+tw]
        return img


class Resize:
    """Resize image"""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return cv2.resize(img, (self.size[1], self.size[0]))
        elif isinstance(img, torch.Tensor):
            # For tensors: (C, H, W)
            img = img.unsqueeze(0)  # Add batch dimension
            img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
            return img.squeeze(0)
        return img


def get_default_transforms(size=80):
    """Get default transforms for model input"""
    return Compose([
        Resize((size, size)),
        ToTensor(),
    ])
