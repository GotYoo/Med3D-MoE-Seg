"""
Stage 1: 多模态对齐模型
只包含编码器和对齐模块，不需要 LLM

Architecture:
    CT Volume → BTB3D MAGViT-2 (3D) → 18-dim tokens → Global Pool → 512-dim
    CT Slices → MedPLIB (2D) → 1024-dim
    Report Text → BioBERT → 768-dim
    ↓
    Unified Alignment Module → Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional


class AlignmentModel(nn.Module):
    """Stage 1 对齐模型：双视觉编码器 + 文本编码器 + 对齐模块"""
    
    def __init__(
        self,
        ct_clip_config: Dict,
        pixel_config: Dict,
        text_config: Dict,
        alignment_config: Dict,
    ):
        super().__init__()
        
        # ===== 1. CT-CLIP 3D Vision Encoder =====
        # 支持切换：BTB3D（高质量但占显存）或 Lightweight（显存友好）
        use_lightweight = ct_clip_config.get('use_lightweight', True)
        latent_dim = alignment_config.get('latent_dim', 512)
        
        if use_lightweight:
            # 轻量级3D CNN（显存优化）
            from model.encoders.lightweight_3d_encoder import create_lightweight_encoder
            self.ct_clip_encoder = create_lightweight_encoder(output_dim=latent_dim)
            self.ct_clip_dim = latent_dim
            print(f"✓ Using Lightweight 3D Encoder (output_dim={latent_dim})")
        else:
            # BTB3D MAGViT-2（需要更多显存）
            from model.encoders.btb3d_vision_tokenizer import create_btb3d_encoder
            checkpoint_path = ct_clip_config.get('vision_tower', 'checkpoints/ct_clip_pretrained.ckpt')
            self.ct_clip_encoder = create_btb3d_encoder(checkpoint_path, config="8x8x8")
            self.ct_clip_dim = 18  # MAGViT-2 token_size
            print(f"✓ Using BTB3D MAGViT-2 Encoder (token_size={self.ct_clip_dim})")
        
        self.use_lightweight_encoder = use_lightweight
        
        # ===== 2. Pixel Encoder (MedPLIB 2D) =====
        # 这里简化为 CLIP ViT，实际应该用 MedPLIB 的完整实现
        from transformers import CLIPVisionModel
        pixel_tower = pixel_config.get('vision_tower', 'openai/clip-vit-large-patch14-336')
        self.pixel_encoder = CLIPVisionModel.from_pretrained(pixel_tower)
        self.pixel_dim = pixel_config.get('pixel_hidden_size', 1024)
        
        # 冻结CLIP以节省显存
        freeze_pixel = pixel_config.get('freeze_vision_tower', True)
        if freeze_pixel:
            for param in self.pixel_encoder.parameters():
                param.requires_grad = False
            print("✓ Pixel Encoder (CLIP) frozen")
        
        # MedPLIB 投影层
        self.pixel_projector = nn.Sequential(
            nn.Linear(self.pixel_encoder.config.hidden_size, self.pixel_dim),
            nn.GELU(),
            nn.Linear(self.pixel_dim, self.pixel_dim),
        )
        
        # ===== 3. Text Encoder (BioBERT) =====
        text_model = text_config.get('model_name', 'dmis-lab/biobert-v1.1')
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_dim = text_config.get('hidden_size', 768)
        
        # 冻结 BioBERT（节省显存，只训练投影层）
        freeze_biobert = text_config.get('freeze_biobert', True)
        if freeze_biobert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("✓ Text Encoder (BioBERT) frozen")
        else:
            # 部分冻结：冻结前N层
            freeze_layers = text_config.get('freeze_layers', 10)
            if freeze_layers > 0:
                for i, layer in enumerate(self.text_encoder.encoder.layer):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"✓ BioBERT: Frozen first {freeze_layers} layers")
        
        # ===== 4. Unified Alignment Module =====
        latent_dim = alignment_config.get('latent_dim', 512)
        
        # 投影到统一空间（ct_clip_encoder已经输出latent_dim，不需要额外投影）
        self.pixel_proj = nn.Linear(self.pixel_dim, latent_dim)
        self.text_proj = nn.Linear(self.text_dim, latent_dim)
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(
            torch.tensor(alignment_config.get('contrastive_temperature', 0.07))
        )
        
        # Cross-attention（可选）
        self.use_cross_attention = alignment_config.get('use_cross_attention', True)
        if self.use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=8,
                batch_first=True,
            )
        
        self.latent_dim = latent_dim
    
    def encode_ct_volume(self, ct_volume: torch.Tensor) -> torch.Tensor:
        """
        编码 3D CT Volume
        Args:
            ct_volume: [B, C, D, H, W]
        Returns:
            features: [B, latent_dim]
        """
        if self.use_lightweight_encoder:
            # 轻量级encoder直接返回 [B, latent_dim]
            features = self.ct_clip_encoder(ct_volume)
        else:
            # BTB3D返回 [B, 18]，需要投影到latent_dim
            features = self.ct_clip_encoder.encode_to_embedding(ct_volume)  # [B, 18]
            if hasattr(self, 'ct_clip_proj'):
                features = self.ct_clip_proj(features)  # [B, latent_dim]
        return features
    
    def encode_ct_slices(self, ct_slices: torch.Tensor) -> torch.Tensor:
        """
        编码 2D CT Slices（像素级）
        Args:
            ct_slices: [B, N, C, H, W] 或 [B*N, C, H, W]
        Returns:
            features: [B, 1024] 或 [B, N, 1024]
        """
        original_shape = ct_slices.shape
        if len(original_shape) == 5:  # [B, N, C, H, W]
            B, N, C, H, W = original_shape
            ct_slices = ct_slices.view(B * N, C, H, W)
            pooled = True
        else:
            pooled = False
        
        # CLIP ViT 编码
        outputs = self.pixel_encoder(ct_slices)
        features = outputs.pooler_output  # [B*N, hidden_size]
        
        # 投影
        features = self.pixel_projector(features)  # [B*N, pixel_dim]
        features = self.pixel_proj(features)  # [B*N, latent_dim]
        
        if pooled:
            features = features.view(B, N, -1).mean(dim=1)  # [B, latent_dim]
        
        return features
    
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码文本（医学报告）
        Args:
            text_inputs: {'input_ids': [B, L], 'attention_mask': [B, L]}
        Returns:
            features: [B, 768]
        """
        outputs = self.text_encoder(**text_inputs)
        # 使用 [CLS] token
        features = outputs.last_hidden_state[:, 0, :]  # [B, text_dim]
        return self.text_proj(features)  # [B, latent_dim]
    
    def contrastive_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE 对比学习损失
        Args:
            features_a: [B, latent_dim]
            features_b: [B, latent_dim]
        Returns:
            loss: scalar
        """
        # L2 归一化
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # 调试：检查特征是否有效
        if torch.isnan(features_a).any() or torch.isnan(features_b).any():
            print(f"WARNING: NaN in features!")
            return torch.tensor(0.0, device=features_a.device, requires_grad=True)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features_a, features_b.T) / self.temperature  # [B, B]
        
        # 标签：对角线为正样本
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        
        # 双向对比损失
        loss_a2b = F.cross_entropy(similarity, labels)
        loss_b2a = F.cross_entropy(similarity.T, labels)
        
        return (loss_a2b + loss_b2a) / 2
    
    def forward(
        self,
        ct_volume: torch.Tensor,
        ct_slices: Optional[torch.Tensor] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            ct_volume: [B, C, D, H, W]
            ct_slices: [B, N, C, H, W]（可选）
            text_inputs: 文本输入（可选）
            return_embeddings: 是否返回所有嵌入
        Returns:
            outputs: {
                'loss': 总损失,
                'ct_clip_text_loss': CT-CLIP ↔ Text,
                'pixel_text_loss': Pixel ↔ Text,
                'ct_clip_pixel_loss': CT-CLIP ↔ Pixel,
                'embeddings': {...}（可选）
            }
        """
        outputs = {}
        
        # 1. 编码 CT Volume (3D 全局)
        ct_clip_features = self.encode_ct_volume(ct_volume)  # [B, latent_dim]
        
        # 2. 编码 CT Slices (2D 像素级)
        pixel_features = None
        if ct_slices is not None:
            pixel_features = self.encode_ct_slices(ct_slices)  # [B, latent_dim]
        
        # 3. 编码文本
        text_features = None
        if text_inputs is not None:
            text_features = self.encode_text(text_inputs)  # [B, latent_dim]
        
        # 4. 计算对比损失
        total_loss = 0.0
        loss_count = 0
        
        # Loss 1: CT-CLIP ↔ Text
        if text_features is not None:
            loss_ct_text = self.contrastive_loss(ct_clip_features, text_features)
            outputs['ct_clip_text_loss'] = loss_ct_text
            total_loss += loss_ct_text
            loss_count += 1
        
        # Loss 2: Pixel ↔ Text
        if pixel_features is not None and text_features is not None:
            loss_pixel_text = self.contrastive_loss(pixel_features, text_features)
            outputs['pixel_text_loss'] = loss_pixel_text
            total_loss += 0.8 * loss_pixel_text  # 权重 0.8
            loss_count += 1
        
        # Loss 3: CT-CLIP ↔ Pixel（视觉内部对齐）
        if pixel_features is not None:
            loss_ct_pixel = self.contrastive_loss(ct_clip_features, pixel_features)
            outputs['ct_clip_pixel_loss'] = loss_ct_pixel
            total_loss += 0.5 * loss_ct_pixel  # 权重 0.5
            loss_count += 1
        
        outputs['loss'] = total_loss
        
        # 5. 返回嵌入（用于评估）
        if return_embeddings:
            outputs['embeddings'] = {
                'ct_clip': ct_clip_features,
                'pixel': pixel_features,
                'text': text_features,
            }
        
        return outputs
    
    def get_text_tokenizer(self):
        """返回文本 tokenizer（用于数据处理）"""
        return self.text_tokenizer


# Backward compatibility: older checkpoints may reference this name
# when pickled. Alias to the current AlignmentModel.
MultiModalAlignmentModule = AlignmentModel
