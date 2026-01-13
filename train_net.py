"""
Med3D-MoE-Seg 训练入口
基于 transformers.Trainer 和 DeepSpeed 的分布式训练
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint

from model.meta_arch.med3d_lisa import Med3DLISA, Med3DLISAConfig
from data.builder import build_dataloader

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


class Med3DLISATrainer(Trainer):
    """
    自定义 Trainer，支持多模态损失计算
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失函数
        
        Args:
            model: Med3DLISA 模型
            inputs: 输入数据
            return_outputs: 是否返回模型输出
        
        Returns:
            loss 或 (loss, outputs)
        """
        # 提取输入
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        images = inputs.get("images")
        labels = inputs.get("labels")
        masks_gt = inputs.get("masks", None)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels,
            masks_gt=masks_gt,
            output_hidden_states=True,
            return_dict=True
        )
        
        loss = outputs['loss']
        
        # 记录各项损失（用于 logging）
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            if outputs.get('lm_loss') is not None:
                self.log({"lm_loss": outputs['lm_loss'].item()})
            if outputs.get('seg_loss') is not None:
                self.log({"seg_loss": outputs['seg_loss'].item()})
        
        return (loss, outputs) if return_outputs else loss


def load_tokenizer(model_args, stage=None):
    """
    加载 Tokenizer 并添加特殊 token
    
    Args:
        model_args: 模型参数
        stage: 训练阶段（1=对齐, 2=分割, 3=MoE+LLM, 4=端到端）
    
    Returns:
        tokenizer or None (Stage 1 不需要)
    """
    # Stage 1 只做对齐，不需要 tokenizer
    if stage == 1:
        logger.info("Stage 1: Skipping tokenizer loading (not needed for alignment)")
        return None
    
    # Normalize and fallback when missing/placeholder
    candidate_name = model_args.model_name_or_path
    if candidate_name is None or str(candidate_name).lower() in {"none", "null", ""}:
        candidate_name = os.environ.get("MODEL_NAME_OR_PATH", "gpt2")
        logger.warning(f"model_name_or_path not provided; falling back to {candidate_name}")
        model_args.model_name_or_path = candidate_name
    model_name_or_path = candidate_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 添加特殊 token
    special_tokens_dict = {
        'additional_special_tokens': ['<image>', '<SEG>']
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    logger.info(f"Added {num_added_tokens} special tokens to tokenizer")
    logger.info(f"Special tokens: {tokenizer.additional_special_tokens}")
    
    return tokenizer


def build_model(model_args, tokenizer, config=None, stage=None):
    """
    构建 Med3D-LISA 模型
    
    Args:
        model_args: 模型参数
        tokenizer: Tokenizer（Stage 1 可以为 None）
        config: YAML 配置
        stage: 训练阶段
    
    Returns:
        model
    """
    # Stage 1：只构建对齐模块，不需要完整的 Med3DLISA
    if stage == 1:
        logger.info("Stage 1: Building alignment-only model (CT-CLIP + MedPLIB + BioBERT)")
        from model.alignment import AlignmentModel
        
        # 从 config 读取参数
        model_config = config.get('model', {})
        alignment_config = model_config.get('alignment', {})
        
        model = AlignmentModel(
            ct_clip_config=model_config.get('ct_clip_encoder', {}),
            pixel_config=model_config.get('pixel_encoder', {}),
            text_config=model_config.get('text_encoder', {}),
            alignment_config=alignment_config,
        )
        
        logger.info("Stage 1 alignment model created (no LLM required)")
        return model
    
    # Stage 2-4：需要完整模型
    # 创建配置
    config = Med3DLISAConfig(
        model_type=model_args.model_type,
        hidden_size=4096,  # Llama-2-7B
        num_experts=model_args.num_experts,
        num_experts_per_tok=model_args.num_experts_per_tok,
        vision_tower=model_args.vision_tower,
        mm_projector_type=model_args.mm_projector_type,
        seg_token_idx=model_args.seg_token_idx,
        image_token_index=model_args.image_token_index,
        sam_type=model_args.sam_type,
        sam_checkpoint=model_args.sam_checkpoint,
        vocab_size=len(tokenizer),
    )
    
    # 创建模型
    model = Med3DLISA(config)
    
    # 如果添加了新 token，需要调整 embedding 层
    model.resize_token_embeddings(len(tokenizer))
    
    # 冻结策略
    if model_args.freeze_vision_tower:
        logger.info("Freezing vision tower")
        model.get_model().get_vision_tower().freeze()
    
    if model_args.freeze_llm_backbone:
        logger.info("Freezing LLM backbone")
        for param in model.get_model().get_model().parameters():
            param.requires_grad = False
    
    if not model_args.tune_mm_projector:
        logger.info("Freezing mm_projector")
        for param in model.get_model().mm_projector.parameters():
            param.requires_grad = False
    
    # LoRA
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model
        
        logger.info("Applying LoRA")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules or ["q_proj", "v_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def main():
    """主函数"""
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsWithLoss))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 如果提供了 YAML 配置文件，优先使用
    config = None
    if data_args.config_file is not None:
        import yaml
        logger.info(f"Loading config from {data_args.config_file}")
        with open(data_args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 从配置文件更新参数
        if 'data' in config:
            data_config = config['data']
            if data_args.train_json is None:
                data_args.train_json = data_config.get('train_json')
            if data_args.val_json is None:
                data_args.val_json = data_config.get('val_json')
            if data_args.test_json is None:
                data_args.test_json = data_config.get('test_json')
            data_args.dataset_type = data_config.get('dataset_type', 'LIDCDataset')
        
        # 从config覆盖TrainingArguments的关键参数
        if 'training' in config:
            train_cfg = config['training']
            if 'num_train_epochs' in train_cfg:
                training_args.num_train_epochs = train_cfg['num_train_epochs']
            if 'per_device_train_batch_size' in train_cfg:
                training_args.per_device_train_batch_size = train_cfg['per_device_train_batch_size']
            if 'gradient_accumulation_steps' in train_cfg:
                training_args.gradient_accumulation_steps = train_cfg['gradient_accumulation_steps']
            if 'learning_rate' in train_cfg:
                training_args.learning_rate = train_cfg['learning_rate']
            if 'bf16' in train_cfg:
                training_args.bf16 = train_cfg['bf16']
            if 'gradient_checkpointing' in train_cfg:
                training_args.gradient_checkpointing = train_cfg['gradient_checkpointing']
            logger.info("✓ Training args overridden from config file")
        
        # 更新 RAG 参数
        if 'model' in config and 'rag' in config['model']:
            rag_config = config['model']['rag']
            if model_args.rag_knowledge_embeddings is None:
                model_args.rag_knowledge_embeddings = rag_config.get('knowledge_embeddings')
            if model_args.rag_knowledge_texts is None:
                model_args.rag_knowledge_texts = rag_config.get('knowledge_texts')
            model_args.rag_enabled = rag_config.get('enabled', True)
            model_args.rag_top_k = rag_config.get('top_k', 3)
    
    # 检测当前训练阶段（必须在使用stage之前）
    stage = None
    if config is not None and 'training' in config:
        stage = config['training'].get('stage', None)
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(logging.INFO)
    
    # 简洁日志：只输出关键训练参数
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Stage: {stage}")
    logger.info(f"  Output: {training_args.output_dir}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size per GPU: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Mixed precision: bf16={training_args.bf16}, fp16={training_args.fp16}")
    logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
    logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
    logger.info("=" * 60)
    
    logger.info(f"RAG enabled: {model_args.rag_enabled}")
    if model_args.rag_enabled:
        logger.info(f"  - Knowledge embeddings: {model_args.rag_knowledge_embeddings}")
        logger.info(f"  - Knowledge texts: {model_args.rag_knowledge_texts}")
        logger.info(f"  - Top-K: {model_args.rag_top_k}")
    
    # 检测 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")
    
    # 加载 tokenizer（Stage 1 跳过）
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_args, stage=stage)
    
    # 构建模型
    logger.info("Building model...")
    model = build_model(model_args, tokenizer, config=config, stage=stage)
    
    # 构建数据加载器
    logger.info("Building dataloaders...")
    
    # Stage 1：使用对齐数据集
    if stage == 1:
        from data.alignment_dataset import LIDCAlignmentDataset, alignment_collate_fn
        from functools import partial
        
        # 获取 BioBERT tokenizer（从模型中）
        text_tokenizer = model.get_text_tokenizer()
        
        # 创建数据集
        # 从配置文件读取数据参数
        data_config = config.get('data', {})
        image_size = tuple(data_config.get('image_size', [96, 96, 96]))
        num_slices = data_config.get('num_slices', 8)
        
        logger.info(f"Data configuration:")
        logger.info(f"  Image size: {image_size}")
        logger.info(f"  Num slices: {num_slices}")
        
        train_dataset = LIDCAlignmentDataset(
            data_root=data_args.data_root,
            annotation_file=data_args.train_json,
            image_size=image_size,
            num_slices=num_slices,
        )
        
        val_dataset = None
        if data_args.val_json:
            val_dataset = LIDCAlignmentDataset(
                data_root=data_args.data_root,
                annotation_file=data_args.val_json,
                image_size=image_size,
                num_slices=num_slices,
            )
        
        # 创建 DataLoader
        from torch.utils.data import DataLoader
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=partial(alignment_collate_fn, tokenizer=text_tokenizer),
            pin_memory=True,
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=partial(alignment_collate_fn, tokenizer=text_tokenizer),
                pin_memory=True,
            )
        
        logger.info(f"Stage 1 alignment datasets created")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"  - Val samples: {len(val_dataset)}")
    
    elif config is not None and data_args.dataset_type == 'LIDCDataset':
        # 使用新版 LIDC 数据集（从 YAML 配置）
        from data.builder import build_dataloaders_from_config
        
        train_dataloader, val_dataloader, test_dataloader = build_dataloaders_from_config(
            config, tokenizer=None
        )
        logger.info(f"Using LIDCDataset with patient-wise split")
        logger.info(f"  - Train samples: {len(train_dataloader.dataset)}")
        logger.info(f"  - Val samples: {len(val_dataloader.dataset)}")
        logger.info(f"  - Test samples: {len(test_dataloader.dataset)}")
    else:
        # 使用旧版数据集（保持向后兼容）
        from data.builder import build_dataloader
        
        dataset_config = {
            'data_root': data_args.data_root,
            'ann_file': data_args.ann_file,
            'image_size': data_args.image_size,
        }
        
        train_dataloader = build_dataloader(
            dataset_config,
            tokenizer,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=data_args.num_workers,
        )
        val_dataloader = None
        test_dataloader = None
        logger.info(f"Using legacy BTB3D dataset") 
        logger.info(f"  - Train samples: {len(train_dataloader.dataset)}")
    
    # 初始化 Trainer
    if stage == 1:
        # Stage 1：多GPU训练循环
        logger.info("*** Stage 1: Alignment Training ***")
        
        # 显存优化：设置每个GPU最多使用90%显存
        n_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {n_gpus}")
        
        for i in range(n_gpus):
            # 限制每个GPU使用90%显存，避免碎片化
            torch.cuda.set_per_process_memory_fraction(0.9, device=i)
            logger.info(f"GPU {i}: Set memory fraction to 0.9")
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 设置设备（支持多GPU）
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Primary device: {device}")
        
        # 确保所有子模块都在cuda:0上（解决DataParallel设备不一致问题）
        def move_to_device(module):
            """递归移动所有子模块到指定设备"""
            for child in module.children():
                move_to_device(child)
            # 移动当前模块的参数和缓冲区
            for param in module.parameters(recurse=False):
                if param.device != device:
                    param.data = param.data.to(device)
            for buffer in module.buffers(recurse=False):
                if buffer.device != device:
                    buffer.data = buffer.data.to(device)
        
        # 先将模型完全移到cuda:0
        logger.info("Moving model to cuda:0...")
        move_to_device(model)
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        
        # 如果有多个GPU，使用DataParallel
        if n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
            logger.info(f"Using DataParallel across GPUs: {list(range(n_gpus))}")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
        
        # 学习率调度器
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_args.num_train_epochs * len(train_dataloader),
        )
        
        # 训练循环
        global_step = 0
        best_loss = float('inf')
        
        from tqdm import tqdm
        
        logger.info("Starting training loop...")
        for epoch in range(int(training_args.num_train_epochs)):
            model.train()
            epoch_loss = 0.0
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")
            logger.info(f"{'='*60}")
            
            # 使用tqdm显示进度条
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch+1}",
                ncols=100,
            )
            
            for batch_idx, batch in progress_bar:
                
                # 将数据移到设备
                ct_volume = batch['ct_volume'].to(device)
                ct_slices = batch['ct_slices'].to(device)
                text_inputs = {
                    k: v.to(device) for k, v in batch['text_inputs'].items()
                }
                
                if batch_idx == 0:
                    logger.info(f"First batch loaded, shapes: ct_volume={ct_volume.shape}, ct_slices={ct_slices.shape}")
                
                # 前向传播
                outputs = model(
                    ct_volume=ct_volume,
                    ct_slices=ct_slices,
                    text_inputs=text_inputs,
                )
                
                loss = outputs['loss']
                
                # 调试：检查loss是否有效
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"DEBUG: outputs keys: {outputs.keys()}")
                    logger.info(f"DEBUG: loss value: {loss}")
                    logger.info(f"DEBUG: loss type: {type(loss)}")
                    if 'ct_clip_text_loss' in outputs:
                        logger.info(f"DEBUG: ct_clip_text_loss: {outputs['ct_clip_text_loss']}")
                
                # DataParallel会返回多个GPU的loss，需要取平均
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # 更新进度条显示
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',  # 增加精度到6位小数
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # 定期日志
                if global_step % training_args.logging_steps == 0:
                    logger.info(
                        f"Step {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
            
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Average Loss: {avg_loss:.6f}")  # 增加精度
            logger.info(f"  Total Loss: {epoch_loss:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = Path(training_args.output_dir) / 'checkpoints' / 'best_model'
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / 'alignment_model.pt')
                logger.info(f"Saved best model with loss {best_loss:.4f}")
        
        logger.info("Stage 1 training completed!")
        return
    
    # Stage 2-4：使用完整 Trainer
    trainer = Med3DLISATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    if training_args.do_train:
        logger.info("*** Training ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # 保存模型
        trainer.save_model()
        
        # 保存训练结果
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # 评估
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
