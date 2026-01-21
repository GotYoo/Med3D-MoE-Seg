"""CLI argument dataclasses + DataCollator.

Refactor-only move from train_net.py to reduce file length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练 LLaMA 模型路径（Stage 3/4 需要，Stage 1/2 不需要）"}
    )
    model_type: str = field(
        default="med3d_moe_llama",
        metadata={"help": "模型类型"}
    )
    vision_tower: str = field(
        default=None,
        metadata={"help": "CT-CLIP 预训练权重路径"}
    )
    mm_projector_type: str = field(
        default="mlp2x_gelu",
        metadata={"help": "多模态投影层类型: linear, mlp2x_gelu"}
    )
    
    # MoE 配置
    num_experts: int = field(
        default=8,
        metadata={"help": "MoE 专家数量"}
    )
    num_experts_per_tok: int = field(
        default=2,
        metadata={"help": "每个 token 激活的专家数量（Top-K）"}
    )
    
    # 分割配置
    seg_token_idx: int = field(
        default=32000,
        metadata={"help": "<SEG> token 的索引"}
    )
    image_token_index: int = field(
        default=-200,
        metadata={"help": "<image> token 的索引"}
    )
    
    # SAM 配置
    sam_type: str = field(
        default="vit_b_ori",
        metadata={"help": "SAM-Med3D 模型类型"}
    )
    sam_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "SAM-Med3D 预训练权重路径"}
    )
    
    # RAG 配置
    rag_enabled: bool = field(
        default=True,
        metadata={"help": "是否启用 RAG 知识检索"}
    )
    rag_knowledge_embeddings: Optional[str] = field(
        default=None,
        metadata={"help": "RAG 知识库 embeddings 文件路径"}
    )
    rag_knowledge_texts: Optional[str] = field(
        default=None,
        metadata={"help": "RAG 知识库文本文件路径"}
    )
    rag_top_k: int = field(
        default=3,
        metadata={"help": "RAG 检索 Top-K 数量"}
    )
    
    # 训练策略
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "是否冻结视觉编码器"}
    )
    freeze_llm_backbone: bool = field(
        default=False,
        metadata={"help": "是否冻结 LLM 主干"}
    )
    tune_mm_projector: bool = field(
        default=True,
        metadata={"help": "是否训练多模态投影层"}
    )
    
    # LoRA 配置
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用 LoRA"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "LoRA 目标模块"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_root: str = field(
        metadata={"help": "NIfTI 文件根目录"}
    )
    ann_file: str = field(
        default=None,
        metadata={"help": "标注文件路径（JSON 格式，旧版）"}
    )
    
    # 新版数据划分（由 prepare_data_split.py 生成）
    dataset_type: str = field(
        default="LIDCDataset",
        metadata={"help": "数据集类型: LIDCDataset, BTB3D"}
    )
    train_json: Optional[str] = field(
        default=None,
        metadata={"help": "训练集 JSON 文件路径"}
    )
    val_json: Optional[str] = field(
        default=None,
        metadata={"help": "验证集 JSON 文件路径"}
    )
    test_json: Optional[str] = field(
        default=None,
        metadata={"help": "测试集 JSON 文件路径"}
    )
    
    image_size: List[int] = field(
        default_factory=lambda: [128, 128, 128],
        metadata={"help": "图像尺寸 [D, H, W]"}
    )
    
    # 数据加载
    num_workers: int = field(
        default=4,
        metadata={"help": "数据加载的工作进程数"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    
    # YAML 配置文件（推荐）
    config_file: Optional[str] = field(
        default=None,
        metadata={"help": "YAML 配置文件路径（如 config/med3d_lisa_full.yaml）"}
    )


@dataclass
class TrainingArgumentsWithLoss(TrainingArguments):
    """扩展的训练参数，包含损失权重"""
    seg_loss_weight: float = field(
        default=1.0,
        metadata={"help": "分割损失权重"}
    )
    bce_weight: float = field(
        default=0.5,
        metadata={"help": "BCE 损失权重"}
    )
    dice_weight: float = field(
        default=0.5,
        metadata={"help": "Dice 损失权重"}
    )
    
    # WandB
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["wandb"],
        metadata={"help": "日志记录工具"}
    )


class Med3DDataCollator:
    """自定义 DataCollator，处理 3D 图像和文本"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        if not batch:
            return {}
        
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        masks = [item.get('mask') for item in batch]
        
        # Tokenize 文本
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        labels = input_ids.clone()
        
        # 处理 padding 标签
        if self.tokenizer.pad_token_id is not None:
             labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 堆叠图像
        images = torch.stack(images, dim=0)
        
        # 处理掩码
        masks_tensor = None
        if all(m is not None for m in masks):
            masks_tensor = torch.stack(masks, dim=0)
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images,
            'masks': masks_tensor
        }


