"""
SAM-Med3D Transform - 适配分割模型的数据变换
复用自 SAM-Med3D 项目
"""

import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F


class ResizeLongestSide3D:
    """
    将 3D volume 的长边缩放到指定尺寸，同时保持纵横比，并进行 padding 以适配 SAM-Med3D 的输入要求
    """
    
    def __init__(self, target_length: int = 128):
        """
        Args:
            target_length: 目标长边尺寸
        """
        self.target_length = target_length
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        应用变换
        
        Args:
            volume: 输入 3D volume，形状为 [C, D, H, W] 或 [D, H, W]
        
        Returns:
            torch.Tensor: 处理后的 volume，形状为 [C, target_length, target_length, target_length]
        """
        # 确保是 4D tensor [C, D, H, W]
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # [1, D, H, W]
        
        # 获取原始尺寸
        _, d, h, w = volume.shape
        
        # 计算缩放比例（长边缩放到 target_length）
        longest_side = max(d, h, w)
        scale = self.target_length / longest_side
        
        # 计算新尺寸
        new_d = int(d * scale)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize volume（保持纵横比）
        # 使用 trilinear 插值进行 3D resize
        resized_volume = F.interpolate(
            volume.unsqueeze(0),  # [1, C, D, H, W]
            size=(new_d, new_h, new_w),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # [C, new_d, new_h, new_w]
        
        # Padding 到目标尺寸（使得所有维度都是 target_length）
        pad_d = self.target_length - new_d
        pad_h = self.target_length - new_h
        pad_w = self.target_length - new_w
        
        # 计算每个维度的 padding（前后对称 padding）
        padding = (
            pad_w // 2, pad_w - pad_w // 2,  # W 维度 (左, 右)
            pad_h // 2, pad_h - pad_h // 2,  # H 维度 (上, 下)
            pad_d // 2, pad_d - pad_d // 2   # D 维度 (前, 后)
        )
        
        # 应用 padding
        padded_volume = F.pad(resized_volume, padding, mode='constant', value=0)
        
        return padded_volume  # [C, target_length, target_length, target_length]
    
    def apply_with_boxes(self, volume: torch.Tensor, boxes: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        应用变换并同时调整 bounding boxes
        
        Args:
            volume: 输入 3D volume
            boxes: bounding boxes，形状为 [N, 6]，格式为 [z1, y1, x1, z2, y2, x2]
        
        Returns:
            Tuple[torch.Tensor, np.ndarray]: 处理后的 volume 和调整后的 boxes
        """
        # 获取原始尺寸
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        _, d, h, w = volume.shape
        
        # 计算缩放比例
        longest_side = max(d, h, w)
        scale = self.target_length / longest_side
        
        # Resize volume
        transformed_volume = self(volume)
        
        # 调整 boxes 坐标
        new_d = int(d * scale)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        pad_d = self.target_length - new_d
        pad_h = self.target_length - new_h
        pad_w = self.target_length - new_w
        
        # 缩放 boxes
        boxes = boxes * scale
        
        # 添加 padding offset
        boxes[:, [0, 3]] += pad_d // 2  # z 坐标
        boxes[:, [1, 4]] += pad_h // 2  # y 坐标
        boxes[:, [2, 5]] += pad_w // 2  # x 坐标
        
        return transformed_volume, boxes
