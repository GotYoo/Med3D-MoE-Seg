"""
SAM-Med3D Mask Decoder Adapter
封装 SAM-Med3D 作为 Mask Decoder，用于生成 3D 分割掩码
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SAMMed3DMaskDecoder(nn.Module):
    """
    SAM-Med3D Mask Decoder
    接收图像嵌入和提示嵌入，生成 3D 分割掩码
    """
    
    def __init__(
        self, 
        model_type: str = "vit_b_ori",
        checkpoint_path: Optional[str] = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        prompt_embed_dim: int = 256,
        llm_hidden_size: int = 4096
    ):
        """
        初始化 SAM-Med3D Decoder
        
        Args:
            model_type: SAM 模型类型 ('vit_b_ori', 'vit_l', etc.)
            checkpoint_path: SAM-Med3D 预训练权重路径
            freeze_image_encoder: 是否冻结图像编码器
            freeze_prompt_encoder: 是否冻结提示编码器
            prompt_embed_dim: SAM 提示嵌入维度（通常是 256）
            llm_hidden_size: LLM 隐藏层维度
        """
        super().__init__()
        
        self.model_type = model_type
        self.prompt_embed_dim = prompt_embed_dim
        
        # 加载 SAM-Med3D 模型
        self._load_sam_model(checkpoint_path, model_type)
        
        # 冻结策略
        if freeze_image_encoder and hasattr(self, 'sam'):
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        if freeze_prompt_encoder and hasattr(self, 'sam'):
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
        
        # Mask Decoder 必须参与训练
        if hasattr(self, 'sam'):
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = True
        
        # 投影层：将 LLM 的 hidden state 投影为 SAM 的 Prompt Embedding
        self.prompt_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(llm_hidden_size // 2, prompt_embed_dim),
            nn.LayerNorm(prompt_embed_dim)
        )
    
    def _load_sam_model(self, checkpoint_path, model_type):
        """
        加载 SAM-Med3D 模型
        
        Args:
            checkpoint_path: 权重路径
            model_type: 模型类型
        """
        try:
            # 尝试从 segment_anything_3d 加载
            # 注意：需要安装 SAM-Med3D 库
            from segment_anything_3d import sam_model_registry3D
            
            self.sam = sam_model_registry3D[model_type](checkpoint=checkpoint_path)
            print(f"Successfully loaded SAM-Med3D model: {model_type}")
            
        except ImportError:
            print("Warning: segment_anything_3d not found. Using placeholder model.")
            # 创建占位符模型
            self._create_placeholder_model()
        except Exception as e:
            print(f"Error loading SAM-Med3D: {e}")
            self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """创建占位符模型（用于开发/测试）"""
        class PlaceholderSAM(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Identity()
                self.prompt_encoder = nn.Identity()
                self.mask_decoder = nn.Identity()
        
        self.sam = PlaceholderSAM()
    
    def forward_image_features(self, images_3d: torch.Tensor) -> torch.Tensor:
        """
        提取图像的 dense embedding（用于 Mask Decoder）
        
        Args:
            images_3d: [B, C, D, H, W] 3D 医学图像
        
        Returns:
            image_embeddings: [B, embed_dim, D', H', W'] 图像嵌入
        """
        if not hasattr(self, 'sam'):
            raise ValueError("SAM model not loaded")
        
        # 使用 SAM-Med3D 的图像编码器
        with torch.set_grad_enabled(self.training and 
                                    any(p.requires_grad for p in self.sam.image_encoder.parameters())):
            image_embeddings = self.sam.image_encoder(images_3d)
        
        return image_embeddings
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        seg_token_hidden_states: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        生成 3D 分割掩码
        
        Args:
            image_embeddings: [B, embed_dim, D', H', W'] 来自 forward_image_features 的图像嵌入
            seg_token_hidden_states: [B, llm_hidden_size] 来自 LLM 的 <SEG> token 的 hidden state
            return_logits: 是否返回 logits（用于计算损失）
        
        Returns:
            masks: [B, 1, D, H, W] 3D 分割掩码（或 logits）
        """
        batch_size = image_embeddings.shape[0]
        
        # 将 LLM 的 hidden state 投影为 SAM 的 Prompt Embedding
        # seg_token_hidden_states: [B, llm_hidden_size] -> [B, prompt_embed_dim]
        prompt_embeddings = self.prompt_projector(seg_token_hidden_states)
        
        # 扩展维度以匹配 SAM 的输入格式: [B, 1, prompt_embed_dim]
        prompt_embeddings = prompt_embeddings.unsqueeze(1)
        
        # 使用 SAM 的 Prompt Encoder（这里我们直接使用投影后的 embedding 作为 sparse prompt）
        # 在真实实现中，可能需要调用 sam.prompt_encoder
        sparse_embeddings = prompt_embeddings
        dense_embeddings = torch.zeros(
            batch_size, 
            self.prompt_embed_dim,
            *image_embeddings.shape[2:],  # 空间维度
            device=image_embeddings.device,
            dtype=image_embeddings.dtype
        )
        
        # 使用 SAM 的 Mask Decoder 生成掩码
        try:
            # 获取位置编码
            if hasattr(self.sam.prompt_encoder, 'get_dense_pe'):
                image_pe = self.sam.prompt_encoder.get_dense_pe()
            else:
                image_pe = None
            
            # 解码掩码
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 上采样到原始分辨率（如果需要）
            # low_res_masks: [B, 1, D', H', W']
            # 可以使用 F.interpolate 上采样
            
            if return_logits:
                return low_res_masks
            else:
                return torch.sigmoid(low_res_masks)
                
        except Exception as e:
            print(f"Error in mask decoder: {e}")
            # 返回占位符掩码
            return torch.zeros(
                batch_size, 1, 128, 128, 128,
                device=image_embeddings.device,
                dtype=image_embeddings.dtype
            )
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        seg_token_projected: torch.Tensor
    ) -> torch.Tensor:
        """
        预测掩码（用于推理）
        
        Args:
            image_embeddings: 图像嵌入
            seg_token_projected: 投影后的 SEG token embedding
        
        Returns:
            masks: 预测的 3D 掩码
        """
        return self.forward(image_embeddings, seg_token_projected, return_logits=False)