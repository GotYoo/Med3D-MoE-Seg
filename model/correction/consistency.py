"""
Consistency Checker
Stage 4: Text-Mask 一致性检查器
用于评估生成的报告和分割掩码之间的一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class ConsistencyChecker(nn.Module):
    """
    一致性检查器
    
    使用 Cross-Attention 机制评估生成的文本报告与分割掩码的一致性
    
    架构：
    - Query: Mask 特征（下采样后）
    - Key/Value: Text Embeddings（来自 LLM）
    - Output: 匹配置信度分数 (0-1)
    """
    
    def __init__(self,
                 mask_channels: int = 256,
                 text_hidden_size: int = 4096,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化一致性检查器
        
        Args:
            mask_channels: Mask 特征通道数
            text_hidden_size: 文本特征维度（LLM hidden size）
            embed_dim: Cross-Attention 的嵌入维度
            num_heads: Multi-head Attention 的头数
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.mask_channels = mask_channels
        self.text_hidden_size = text_hidden_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Mask Encoder: 将 3D mask 下采样并编码
        self.mask_encoder = MaskEncoder(
            in_channels=1,  # 输入是单通道 mask
            out_channels=mask_channels,
            embed_dim=embed_dim
        )
        
        # Text Encoder: 投影 LLM 输出到统一空间
        self.text_encoder = nn.Sequential(
            nn.Linear(text_hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-Attention: Mask 作为 Query, Text 作为 Key/Value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Score Predictor: 输出一致性分数 (0-1)
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 0-1 范围的分数
        )
        
        print(f"ConsistencyChecker initialized:")
        print(f"  - Mask channels: {mask_channels}")
        print(f"  - Text hidden size: {text_hidden_size}")
        print(f"  - Embed dim: {embed_dim}")
        print(f"  - Num heads: {num_heads}")
    
    def forward(self,
                mask_output: torch.Tensor,
                text_embeds: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算 Text-Mask 匹配分数
        
        Args:
            mask_output: 分割掩码 [B, 1, D, H, W] 或 [B, D, H, W]
            text_embeds: 文本 Embeddings [B, seq_len, text_hidden_size]
            return_attention: 是否返回 attention weights
            
        Returns:
            outputs: 包含一致性分数的字典
                - consistency_score: 一致性分数 [B, 1]
                - attention_weights: Attention 权重 [B, num_heads, 1, seq_len] (可选)
        """
        # 编码 Mask 特征
        if mask_output.dim() == 4:
            mask_output = mask_output.unsqueeze(1)  # [B, 1, D, H, W]
        
        mask_features = self.mask_encoder(mask_output)  # [B, 1, embed_dim]
        
        # 编码 Text 特征
        text_features = self.text_encoder(text_embeds)  # [B, seq_len, embed_dim]
        
        # Cross-Attention: Mask (Query) attends to Text (Key/Value)
        attended_features, attention_weights = self.cross_attention(
            query=mask_features,     # [B, 1, embed_dim]
            key=text_features,       # [B, seq_len, embed_dim]
            value=text_features,     # [B, seq_len, embed_dim]
            need_weights=return_attention
        )  # attended_features: [B, 1, embed_dim]
        
        # 预测一致性分数
        attended_features = attended_features.squeeze(1)  # [B, embed_dim]
        consistency_score = self.score_predictor(attended_features)  # [B, 1]
        
        outputs = {
            'consistency_score': consistency_score,
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def check_consistency(self,
                          mask_output: torch.Tensor,
                          text_embeds: torch.Tensor,
                          threshold: float = 0.7) -> Tuple[bool, float]:
        """
        检查一致性（推理时使用）
        
        Args:
            mask_output: 分割掩码
            text_embeds: 文本 Embeddings
            threshold: 一致性阈值
            
        Returns:
            is_consistent: 是否一致
            score: 一致性分数
        """
        with torch.no_grad():
            outputs = self.forward(mask_output, text_embeds, return_attention=False)
            score = outputs['consistency_score'].item()
            is_consistent = score >= threshold
        
        return is_consistent, score
    
    def compute_matching_loss(self,
                              mask_output: torch.Tensor,
                              text_embeds: torch.Tensor,
                              target_score: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算匹配损失（训练时使用）
        
        Args:
            mask_output: 分割掩码
            text_embeds: 文本 Embeddings
            target_score: 目标分数 [B, 1]，如果为 None 则假设正样本（目标=1）
            
        Returns:
            loss: 匹配损失
        """
        outputs = self.forward(mask_output, text_embeds, return_attention=False)
        pred_score = outputs['consistency_score']
        
        # 如果没有提供目标分数，假设为正样本（完全一致）
        if target_score is None:
            target_score = torch.ones_like(pred_score)
        
        # Binary Cross Entropy Loss
        loss = F.binary_cross_entropy(pred_score, target_score)
        
        return loss


class MaskEncoder(nn.Module):
    """
    Mask 编码器
    将 3D 分割掩码下采样并编码为特征向量
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 256,
                 embed_dim: int = 512):
        """
        初始化 Mask 编码器
        
        Args:
            in_channels: 输入通道数（通常为 1）
            out_channels: 中间特征通道数
            embed_dim: 输出嵌入维度
        """
        super().__init__()
        
        # 3D 卷积下采样网络
        self.conv_layers = nn.Sequential(
            # Stage 1: [B, 1, D, H, W] -> [B, 64, D/2, H/2, W/2]
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Stage 2: [B, 64, D/2, H/2, W/2] -> [B, 128, D/4, H/4, W/4]
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3: [B, 128, D/4, H/4, W/4] -> [B, 256, D/8, H/8, W/8]
            nn.Conv3d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 投影到 embed_dim
        self.projector = nn.Sequential(
            nn.Linear(out_channels, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            mask: 输入掩码 [B, 1, D, H, W]
            
        Returns:
            features: 编码后的特征 [B, 1, embed_dim]
        """
        # 卷积特征提取
        features = self.conv_layers(mask)  # [B, out_channels, D/8, H/8, W/8]
        
        # 全局池化
        features = self.global_pool(features)  # [B, out_channels, 1, 1, 1]
        features = features.view(features.size(0), -1)  # [B, out_channels]
        
        # 投影到 embed_dim
        features = self.projector(features)  # [B, embed_dim]
        
        # 添加序列维度以适配 Attention
        features = features.unsqueeze(1)  # [B, 1, embed_dim]
        
        return features


class SelfCorrectionLoop:
    """
    自我修正循环
    
    使用 ConsistencyChecker 进行迭代改进
    """
    
    def __init__(self,
                 consistency_checker: ConsistencyChecker,
                 max_iterations: int = 3,
                 threshold: float = 0.7):
        """
        初始化自我修正循环
        
        Args:
            consistency_checker: 一致性检查器
            max_iterations: 最大迭代次数
            threshold: 一致性阈值
        """
        self.consistency_checker = consistency_checker
        self.max_iterations = max_iterations
        self.threshold = threshold
    
    def __call__(self,
                 model,
                 initial_output: Dict[str, torch.Tensor],
                 refine_fn) -> Dict[str, torch.Tensor]:
        """
        执行自我修正循环
        
        Args:
            model: 主模型（用于生成）
            initial_output: 初始输出（包含 mask 和 text）
            refine_fn: 细化函数，接收当前输出并返回改进的输出
            
        Returns:
            final_output: 最终输出
        """
        current_output = initial_output
        
        for iteration in range(self.max_iterations):
            # 检查一致性
            is_consistent, score = self.consistency_checker.check_consistency(
                mask_output=current_output['mask'],
                text_embeds=current_output['text_embeds'],
                threshold=self.threshold
            )
            
            print(f"Iteration {iteration + 1}: Consistency score = {score:.4f}")
            
            # 如果已经一致，提前终止
            if is_consistent:
                print(f"Consistency achieved! (score >= {self.threshold})")
                break
            
            # 否则，细化输出
            print("Refining output...")
            current_output = refine_fn(current_output, score)
        
        return current_output
