"""
BTB3D Vision Tokenizer Adapter
直接使用 BTB3D 的 3D Vision Encoder（MAGViT-2）
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# 添加 BTB3D encoder-decoder 到 Python 路径
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent
BTB3D_ENCODER_PATH = _project_root / 'submodules/BTB3D/encoder-decoder'

if str(BTB3D_ENCODER_PATH) not in sys.path:
    sys.path.insert(0, str(BTB3D_ENCODER_PATH))

try:
    from modeling.magvit2 import Encoder as VisionEncoder3D
    from modeling.patcher_module import Patcher3D, UnPatcher3D
    HAS_BTB3D_ENCODER = True
    print("✓ BTB3D VisionEncoder3D loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import BTB3D VisionEncoder3D: {e}")
    HAS_BTB3D_ENCODER = False
    VisionEncoder3D = None


class BTB3DVisionTokenizer(nn.Module):
    """
    BTB3D 3D Vision Tokenizer Adapter
    使用 MAGViT-2 架构的 3D Encoder
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        filters: int = 128,
        token_size: int = 18,
        num_res_blocks: int = 4,
        temporal_downsample: tuple = (2, 4, 4),
        channel_multipliers: tuple = (1, 2, 2, 4),
        use_patcher: bool = True,
    ):
        """
        Args:
            checkpoint_path: BTB3D预训练权重路径 (.ckpt)
            filters: 基础通道数
            token_size: Token维度
            num_res_blocks: ResNet块数量
            temporal_downsample: 时间维度下采样
            channel_multipliers: 通道倍增器
            use_patcher: 是否使用Haar小波patcher
        """
        super().__init__()
        
        if not HAS_BTB3D_ENCODER:
            raise ImportError(
                "BTB3D VisionEncoder3D not available. "
                "Please check submodules/BTB3D/encoder-decoder"
            )
        
        self.token_size = token_size
        self.use_patcher = use_patcher
        
        # Haar Wavelet Patcher (2x下采样)
        if use_patcher:
            self.patcher = Patcher3D(patch_size=2, patch_method="haar")
            input_channels = 8  # Haar小波产生8个通道
        else:
            self.patcher = None
            input_channels = 1
        
        # BTB3D 3D Encoder
        self.encoder = VisionEncoder3D(
            filters=filters,
            chan_in=input_channels,
            chan_out=token_size,
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            channel_multipliers=channel_multipliers,
            skip_conv_first=False,
            skip_conv_last=False,
        )
        
        # 加载预训练权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        
        self.is_loaded = True
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载BTB3D预训练权重"""
        try:
            print(f"Loading BTB3D checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 提取encoder权重
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤只包含encoder的权重
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if 'encoder' in k:
                    # 移除前缀 'tokenizer.encoder.' 或 'encoder.'
                    new_key = k.replace('tokenizer.encoder.', '').replace('encoder.', '')
                    encoder_state_dict[new_key] = v
            
            # 加载权重
            if len(encoder_state_dict) > 0:
                missing, unexpected = self.encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"✓ Loaded {len(encoder_state_dict)} encoder layers")
                if len(missing) > 0:
                    print(f"  Missing keys: {len(missing)}")
                if len(unexpected) > 0:
                    print(f"  Unexpected keys: {len(unexpected)}")
            else:
                print("⚠ No encoder weights found in checkpoint, using random initialization")
            
            # 加载patcher权重（如果有）
            if self.use_patcher and self.patcher is not None:
                patcher_state_dict = {}
                for k, v in state_dict.items():
                    if 'patcher' in k:
                        new_key = k.replace('tokenizer.patcher.', '').replace('patcher.', '')
                        patcher_state_dict[new_key] = v
                
                if len(patcher_state_dict) > 0:
                    self.patcher.load_state_dict(patcher_state_dict, strict=False)
                    print(f"✓ Loaded patcher weights")
                    
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            print("Using random initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C, D, H, W] CT Volume (C通常为1)
        
        Returns:
            features: [B, token_size, D', H', W'] 编码后的3D tokens
        """
        # 输入归一化到 [-1, 1]
        if x.min() >= 0:  # 如果是 [0, 1]
            x = x * 2 - 1
        
        # Haar Wavelet Patcher (可选)
        if self.use_patcher and self.patcher is not None:
            x = self.patcher(x)  # [B, 8, D/2, H/2, W/2]
        
        # 3D Encoding
        z = self.encoder(x)  # [B, token_size, D', H', W']
        
        return z
    
    def encode_to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码为全局嵌入向量（用于对比学习）
        
        Args:
            x: [B, C, D, H, W]
        
        Returns:
            embedding: [B, embedding_dim]
        """
        z = self.forward(x)  # [B, token_size, D', H', W']
        
        # 全局平均池化
        embedding = z.mean(dim=[2, 3, 4])  # [B, token_size]
        
        return embedding
    
    @property
    def hidden_size(self):
        """返回token维度"""
        return self.token_size
    
    @property
    def output_dim(self):
        """返回输出维度（用于对比学习）"""
        return self.token_size


def create_btb3d_encoder(
    checkpoint_path: str = None,
    config: str = "8x8x8",
) -> BTB3DVisionTokenizer:
    """
    创建BTB3D Encoder的便捷函数
    
    Args:
        checkpoint_path: 预训练权重路径
        config: 配置类型 "8x8x8" 或 "16x16x8"
    
    Returns:
        BTB3DVisionTokenizer实例
    """
    if config == "8x8x8":
        # 8x8x8 下采样配置（匹配BTB3D checkpoint）
        return BTB3DVisionTokenizer(
            checkpoint_path=checkpoint_path,
            filters=96,  # 匹配checkpoint
            token_size=18,
            num_res_blocks=4,
            temporal_downsample=(2, 4, 4),  # 总下采样: 2*4*4 = 32
            channel_multipliers=(1, 2, 2, 4),
            use_patcher=True,  # Haar: 2x下采样
        )
    elif config == "16x16x8":
        # 16x16x8 下采样配置
        return BTB3DVisionTokenizer(
            checkpoint_path=checkpoint_path,
            filters=128,
            token_size=18,
            num_res_blocks=4,
            temporal_downsample=(4, 4, 4),  # 总下采样: 4*4*4 = 64
            channel_multipliers=(1, 2, 2, 4),
            use_patcher=True,
        )
    else:
        raise ValueError(f"Unknown config: {config}")


if __name__ == "__main__":
    # 测试
    print("Testing BTB3D Vision Tokenizer...")
    
    encoder = create_btb3d_encoder(
        checkpoint_path="checkpoints/ct_clip_pretrained.ckpt",
        config="8x8x8"
    )
    
    # 测试输入
    x = torch.randn(2, 1, 128, 128, 128)  # [B, C, D, H, W]
    
    # 前向传播
    z = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output tokens shape: {z.shape}")
    
    # 全局嵌入
    emb = encoder.encode_to_embedding(x)
    print(f"Global embedding shape: {emb.shape}")
    
    print("✓ BTB3D Vision Tokenizer test passed!")
