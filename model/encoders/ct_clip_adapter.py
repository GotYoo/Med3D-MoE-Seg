"""
CT-CLIP Vision Tower Adapter
直接复用 BTB3D 的 CT-CLIP 实现（使用 CTViT 3D 架构）
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Optional

# 导入 submodules 的组件
try:
    from model import BTB3D_CTCLIPVisionTower, HAS_BTB3D
except ImportError:
    # 如果主 __init__ 没有初始化，直接导入
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_current_dir))
    BTB3D_PATH = os.path.join(_project_root, 'submodules/BTB3D/report-generation')
    
    if BTB3D_PATH not in sys.path:
        sys.path.insert(0, BTB3D_PATH)
    
    try:
        from llava.model.multimodal_encoder.ct_clip import CLIPVisionTower as BTB3D_CTCLIPVisionTower
        HAS_BTB3D = True
    except ImportError:
        print("Warning: BTB3D CT-CLIP not found, using fallback")
        BTB3D_CTCLIPVisionTower = None
        HAS_BTB3D = False


class CTCLIPVisionTower(nn.Module):
    """
    CT-CLIP Vision Tower
    直接复用 BTB3D 的 CT-CLIP 实现（使用 CTViT 处理 3D CT 图像）
    """
    
    def __init__(self, vision_tower_path: str, config, delay_load: bool = False):
        """
        初始化 CT-CLIP Vision Tower
        
        Args:
            vision_tower_path: CT-CLIP 预训练权重路径
            config: 配置对象
            delay_load: 是否延迟加载模型
        """
        super().__init__()
        
        self.is_loaded = False
        self.is_frozen = False  # 添加 is_frozen 属性，默认为 False
        self.vision_tower_path = vision_tower_path
        self.config = config
        
        # 从 config 获取参数
        self.select_layer = getattr(config, 'mm_vision_select_layer', -1)
        self.select_feature = getattr(config, 'mm_vision_select_feature', 'patch')
        
        if HAS_BTB3D and BTB3D_CTCLIPVisionTower is not None:
            # 使用 BTB3D 的 CT-CLIP 实现
            print(f"Using BTB3D CT-CLIP implementation")
            
            # 创建一个 args 对象以匹配 BTB3D 的接口
            class Args:
                mm_vision_select_layer = self.select_layer
                mm_vision_select_feature = self.select_feature
                unfreeze_mm_vision_tower = False
            
            self.args = Args()
            
            if not delay_load:
                self.load_model()
        else:
            # 使用后备实现
            print("Warning: BTB3D not available, using fallback CLIP model")
            self._create_fallback_model()
    
    def load_model(self, device_map=None):
        """
        加载 CT-CLIP 模型（使用 BTB3D 的实现）
        """
        if self.is_loaded:
            print(f'{self.vision_tower_path} is already loaded, skipping.')
            return
        
        if HAS_BTB3D and BTB3D_CTCLIPVisionTower is not None:
            # 使用 BTB3D 的 CT-CLIP
            self.vision_tower = BTB3D_CTCLIPVisionTower(
                vision_tower=self.vision_tower_path,
                args=self.args,
                delay_load=False
            )
            print(f"Successfully loaded BTB3D CT-CLIP from {self.vision_tower_path}")
        else:
            self._create_fallback_model()
        
        self.is_loaded = True
    
    def _create_fallback_model(self):
        """创建后备模型（如果 BTB3D 不可用）"""
        # 优先使用简化的 3D 编码器（保留 3D 特性）
        try:
            from .simple_3d_encoder import Simple3DCTEncoder
            print("Using Simple 3D CT Encoder (maintains 3D processing)")
            self.vision_tower = Simple3DCTEncoder(input_channels=1, output_dim=512)
            self.use_3d_fallback = True
            self.is_loaded = True
            return
        except Exception as e:
            print(f"Failed to load Simple 3D Encoder: {e}")
        
        # 最后的后备：2D CLIP
        from transformers import CLIPVisionModel, CLIPVisionConfig
        
        print("Using fallback 2D CLIP model")
        clip_config = CLIPVisionConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
        )
        
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_path if os.path.exists(self.vision_tower_path) 
                else "openai/clip-vit-base-patch16"
            )
        except:
            self.vision_tower = CLIPVisionModel(clip_config)
        
        self.use_3d_fallback = False
        self.is_loaded = True
    
    def freeze(self):
        """冻结模型参数"""
        if hasattr(self, 'vision_tower'):
            self.vision_tower.eval() # 同样设置为 eval 模式
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.is_frozen = True
            print("Vision tower frozen")
    
    def unfreeze(self):
        """解冻模型参数"""
        if hasattr(self, 'vision_tower'):
            self.vision_tower.train()
            for param in self.vision_tower.parameters():
                param.requires_grad = True
            self.is_frozen = False
            print("Vision tower unfrozen")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: [B, C, D, H, W] 3D 医学图像（对于 BTB3D CT-CLIP）
                   或 [B, C, H, W] 2D 图像（对于后备 CLIP）
        
        Returns:
            image_features: [B, num_patches, hidden_size] 或 [B, hidden_size] 图像特征向量
        """
        if not self.is_loaded:
            self.load_model()
            
        # [Device Fix] Ensure vision_tower is on the same device as input images
        # This is critical for frozen modules which might be missed by simple .to(device) calls or ZeRO partitioning
        if hasattr(self, 'vision_tower'):
             # We check the first parameter's device
             try:
                 first_param = next(self.vision_tower.parameters(), None)
                 if first_param is not None and first_param.device != images.device:
                     # print(f"[Debug] Moving Vision Tower from {first_param.device} to {images.device}")
                     self.vision_tower.to(images.device)
                 else:
                     # If no parameters (unlikely), or params match, check buffers (like running_mean)
                     first_buffer = next(self.vision_tower.buffers(), None)
                     if first_buffer is not None and first_buffer.device != images.device:
                         # print(f"[Debug] Moving Vision Tower Buffers from {first_buffer.device} to {images.device}")
                         self.vision_tower.to(images.device)
             except Exception as e:
                 print(f"[Warning] Failed to check/move vision tower device: {e}")
        
        # 调用底层的 vision_tower
        if HAS_BTB3D and hasattr(self, 'vision_tower') and isinstance(self.vision_tower, BTB3D_CTCLIPVisionTower):
            # BTB3D CT-CLIP 会自动处理 3D 输入
            # BTB3D 的 forward 期望 [B, D, H, W] (没有 channel 维度)
            if images.ndim == 5:  # [B, C, D, H, W]
                # 移除 channel 维度（假设 C=1）
                images = images.squeeze(1)  # [B, D, H, W]
            image_features = self.vision_tower(images)
        elif hasattr(self, 'use_3d_fallback') and self.use_3d_fallback:
            # 使用简化的 3D 编码器
            from .simple_3d_encoder import Simple3DCTEncoder
            if isinstance(self.vision_tower, Simple3DCTEncoder):
                # 直接处理 3D 输入 [B, C, D, H, W]
                image_features = self.vision_tower(images)  # [B, 512]
            else:
                # 降级到 2D
                if images.ndim == 5:  # [B, C, D, H, W]
                    B, C, D, H, W = images.shape
                    images_2d = images[:, :, D//2, :, :]  # [B, C, H, W]
                else:
                    images_2d = images
                
                if images_2d.shape[-2:] != (224, 224):
                    images_2d = torch.nn.functional.interpolate(
                        images_2d, size=(224, 224), mode='bilinear', align_corners=False)
                
                if images_2d.shape[1] == 1:
                    images_2d = images_2d.repeat(1, 3, 1, 1)
                
                outputs = self.vision_tower(pixel_values=images_2d, return_dict=True)
                image_features = outputs.last_hidden_state
        else:
            # 后备模型：处理 2D 切片
            if images.ndim == 5:  # [B, C, D, H, W]
                B, C, D, H, W = images.shape
                # 取中间切片
                images_2d = images[:, :, D//2, :, :]  # [B, C, H, W]
            else:
                images_2d = images
            
            # Resize 到 224x224（CLIP 要求）
            if images_2d.shape[-2:] != (224, 224):
                images_2d = torch.nn.functional.interpolate(
                    images_2d,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False,
                )
            
            # 转换为 3 通道（RGB）
            if images_2d.shape[1] == 1:
                images_2d = images_2d.repeat(1, 3, 1, 1)
            
            outputs = self.vision_tower(pixel_values=images_2d, return_dict=True)
            image_features = outputs.last_hidden_state
        
        return image_features
    
    @property
    def hidden_size(self):
        """返回隐藏层维度"""
        if HAS_BTB3D and hasattr(self, 'vision_tower') and isinstance(self.vision_tower, BTB3D_CTCLIPVisionTower):
            # BTB3D CT-CLIP 使用 CTViT，dim=512
            return 512
        if hasattr(self, 'use_3d_fallback') and self.use_3d_fallback:
             # simple_3d_encoder uses 512 by default
            return getattr(self.vision_tower, 'output_dim', 512)
        return 768  # 后备 CLIP 维度
    
    @property
    def num_patches(self):
        """返回 patch 数量"""
        # 这取决于具体的模型配置
        return 256
    
    @property
    def dtype(self):
        """返回模型的数据类型"""
        if hasattr(self, 'vision_tower'):
            try:
                return next(self.vision_tower.parameters()).dtype
            except:
                return torch.float32
        return torch.float32
    
    @property
    def device(self):
        """返回模型所在的设备"""
        if hasattr(self, 'vision_tower'):
            try:
                return next(self.vision_tower.parameters()).device
            except:
                return torch.device('cpu')
        return torch.device('cpu')
    
    @property
    def dummy_feature(self):
        """返回虚拟特征（用于测试）"""
        return torch.zeros(1, self.num_patches, self.hidden_size, 
                          device=self.device, dtype=self.dtype)


# 保持向后兼容
BTB3DVisionTower = CTCLIPVisionTower