"""
Simple 3D CT Encoder
当 BTB3D CTViT 不可用时的简化 3D 编码器
"""

import torch
import torch.nn as nn


class Simple3DCTEncoder(nn.Module):
    """
    简化的 3D CT 编码器
    使用 3D ResNet 架构处理 CT Volume
    """
    def __init__(self, input_channels=1, output_dim=512):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 3D 卷积编码器（类似 ResNet18 3D）
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, output_dim, blocks=2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """构建 ResNet layer"""
        layers = []
        
        # 第一个 block 可能需要下采样
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        
        # 后续 blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W] or [B, D, H, W]
        Returns:
            features: [B, output_dim]
        """
        # 确保输入是 5D: [B, C, D, H, W]
        if x.ndim == 4:  # [B, D, H, W]
            x = x.unsqueeze(1)  # [B, 1, D, H, W]
        
        # 前向传播
        x = self.conv1(x)  # [B, 64, D/4, H/4, W/4]
        x = self.layer1(x)  # [B, 64, D/4, H/4, W/4]
        x = self.layer2(x)  # [B, 128, D/8, H/8, W/8]
        x = self.layer3(x)  # [B, 256, D/16, H/16, W/16]
        x = self.layer4(x)  # [B, 512, D/32, H/32, W/32]
        
        # 全局池化
        x = self.avgpool(x)  # [B, 512, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        
        return x


class BasicBlock3D(nn.Module):
    """3D ResNet Basic Block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
