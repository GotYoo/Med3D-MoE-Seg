"""
BioBERT Text Encoder
Stage 1: 处理 Clinical Report/History 文本输入
用于编码临床报告、病史等医学文本
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from transformers import AutoModel, AutoTokenizer


class BioBERTEncoder(nn.Module):
    """
    BioBERT 文本编码器
    用于处理临床报告和病史文本
    
    基于 dmis-lab/biobert-v1.1 预训练模型
    输出 [CLS] token embedding 作为文本特征表示
    """
    
    def __init__(self, 
                 model_name: str = "dmis-lab/biobert-v1.1",
                 freeze_layers: Optional[int] = None,
                 output_dim: int = 768):
        """
        初始化 BioBERT 编码器
        
        Args:
            model_name: BioBERT 模型名称，默认使用 dmis-lab/biobert-v1.1
            freeze_layers: 冻结前 N 层，None 表示不冻结
            output_dim: 输出特征维度，默认 768 (BERT-base hidden size)
        """
        super().__init__()
        
        # 加载 BioBERT 预训练模型
        print(f"Loading BioBERT from {model_name}...")
        self.biobert = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # 冻结指定的层
        if freeze_layers is not None and freeze_layers > 0:
            self._freeze_layers(freeze_layers)
            print(f"Frozen first {freeze_layers} layers of BioBERT")
        
        # 如果需要的话，添加输出投影层
        self.hidden_size = self.biobert.config.hidden_size  # 通常是 768
        
    def _freeze_layers(self, num_layers: int):
        """
        冻结 BioBERT 的前 N 层
        
        Args:
            num_layers: 要冻结的层数
        """
        # 冻结 embeddings
        for param in self.biobert.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结指定数量的 encoder 层
        for layer_idx in range(min(num_layers, len(self.biobert.encoder.layer))):
            for param in self.biobert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            text_features: [CLS] token embedding [B, 768]
        """
        # 通过 BioBERT 编码
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 提取 [CLS] token 的 embedding (第一个 token)
        # last_hidden_state: [B, seq_len, hidden_size]
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        
        return cls_embedding
    
    def freeze(self):
        """冻结整个模型"""
        for param in self.biobert.parameters():
            param.requires_grad = False
        print("BioBERT encoder frozen")
    
    def unfreeze(self):
        """解冻整个模型"""
        for param in self.biobert.parameters():
            param.requires_grad = True
        print("BioBERT encoder unfrozen")
    
    @property
    def device(self):
        """返回模型所在设备"""
        return next(self.biobert.parameters()).device


# 辅助函数：创建 BioBERT tokenizer
def create_biobert_tokenizer(model_name: str = "dmis-lab/biobert-v1.1"):
    """
    创建 BioBERT tokenizer
    
    Args:
        model_name: BioBERT 模型名称
        
    Returns:
        tokenizer: BioBERT tokenizer
    """
    return AutoTokenizer.from_pretrained(model_name)
