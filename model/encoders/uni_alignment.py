"""
Unified Alignment Module
Stage 1: 视觉-文本统一对齐模块
使用对比学习进行多模态特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class UnifiedAlignmentModule(nn.Module):
    """
    统一对齐模块
    
    功能：
    1. 将图像和文本特征投影到共同的潜在空间
    2. 使用对比学习 (InfoNCE Loss) 进行对齐
    3. 最大化正样本对的相似度，最小化负样本对的相似度
    """
    
    def __init__(self, 
                 image_dim: int = 512,
                 text_dim: int = 768,
                 latent_dim: int = 512,
                 temperature: float = 0.07):
        """
        初始化统一对齐模块
        
        Args:
            image_dim: 图像特征维度 (e.g., CT-CLIP output dim)
            text_dim: 文本特征维度 (e.g., BioBERT output dim = 768)
            latent_dim: 共同潜在空间维度
            temperature: 对比损失的温度参数
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.temperature = temperature
        
        # 图像投影层：image_dim -> latent_dim
        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 文本投影层：text_dim -> latent_dim
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 可学习的温度参数（可选）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
    
    def forward(self, 
                image_features: torch.Tensor,
                text_features: torch.Tensor,
                return_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            image_features: 图像特征 [B, image_dim]
            text_features: 文本特征 [B, text_dim]
            return_loss: 是否计算对比损失
            
        Returns:
            outputs: 包含对齐特征和损失的字典
                - image_embeds: 投影后的图像特征 [B, latent_dim]
                - text_embeds: 投影后的文本特征 [B, latent_dim]
                - contrastive_loss: 对比损失 (如果 return_loss=True)
        """
        # 投影到共同潜在空间
        image_embeds = self.image_projector(image_features)  # [B, latent_dim]
        text_embeds = self.text_projector(text_features)      # [B, latent_dim]
        
        # L2 归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        outputs = {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
        }
        
        # 计算对比损失
        if return_loss:
            contrastive_loss = self.compute_contrastive_loss(image_embeds, text_embeds)
            outputs['contrastive_loss'] = contrastive_loss
        
        return outputs
    
    def compute_contrastive_loss(self, 
                                  image_embeds: torch.Tensor,
                                  text_embeds: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失 (InfoNCE Loss / NT-Xent Loss)
        
        Args:
            image_embeds: 归一化后的图像特征 [B, latent_dim]
            text_embeds: 归一化后的文本特征 [B, latent_dim]
            
        Returns:
            loss: 对比损失标量
        """
        batch_size = image_embeds.shape[0]
        
        # 计算相似度矩阵：[B, B]
        # logits[i, j] 表示第 i 个图像与第 j 个文本的相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()  # [B, B]
        logits_per_text = logits_per_image.t()  # [B, B]
        
        # 创建标签：对角线为正样本对 (i, i)
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # 计算交叉熵损失（双向）
        loss_i2t = F.cross_entropy(logits_per_image, labels)  # Image -> Text
        loss_t2i = F.cross_entropy(logits_per_text, labels)   # Text -> Image
        
        # 平均损失
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        return contrastive_loss
    
    def get_similarity_matrix(self, 
                              image_embeds: torch.Tensor,
                              text_embeds: torch.Tensor) -> torch.Tensor:
        """
        获取图像-文本相似度矩阵
        
        Args:
            image_embeds: 图像特征 [B, latent_dim]
            text_embeds: 文本特征 [B, latent_dim]
            
        Returns:
            similarity_matrix: 相似度矩阵 [B, B]
        """
        # 归一化
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 计算余弦相似度
        similarity = image_embeds @ text_embeds.t()
        
        return similarity


class ContrastiveLoss(nn.Module):
    """
    独立的对比损失模块（可选）
    实现 InfoNCE / NT-Xent Loss
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                features_a: torch.Tensor,
                features_b: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            features_a: 模态 A 的特征 [B, D]
            features_b: 模态 B 的特征 [B, D]
            
        Returns:
            loss: 对比损失
        """
        batch_size = features_a.shape[0]
        
        # 归一化
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # 相似度矩阵
        logits = (features_a @ features_b.t()) / self.temperature
        
        # 标签
        labels = torch.arange(batch_size, device=features_a.device)
        
        # 双向损失
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        
        return (loss_a + loss_b) / 2.0
