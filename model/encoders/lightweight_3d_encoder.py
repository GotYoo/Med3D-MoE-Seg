"""
轻量级 3D CNN 编码器
用于Stage 1训练，避免BTB3D的显存占用
"""

import torch
import torch.nn as nn


class Lightweight3DEncoder(nn.Module):
    """轻量级3D编码器：适合有限显存"""
    
    def __init__(self, in_channels=1, hidden_dim=128, output_dim=512):
        super().__init__()
        
        # 简单的3D卷积网络
        self.encoder = nn.Sequential(
            # Block 1: 64x64x64 -> 32x32x32
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32x32x32 -> 16x16x16
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 16x16x16 -> 8x8x8
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 8x8x8 -> 4x4x4
            nn.Conv3d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 输出投影
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            features: [B, output_dim]
        """
        # 3D卷积特征提取
        features = self.encoder(x)  # [B, hidden_dim, D', H', W']
        
        # 全局池化
        pooled = self.global_pool(features)  # [B, hidden_dim, 1, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, hidden_dim]
        
        # 投影到输出维度
        output = self.projector(pooled)  # [B, output_dim]
        
        return output


def create_lightweight_encoder(output_dim=512):
    """创建轻量级3D编码器"""
    return Lightweight3DEncoder(
        in_channels=1,
        hidden_dim=128,
        output_dim=output_dim,
    )
