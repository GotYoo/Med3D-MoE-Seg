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
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.alignment import AlignmentModel

import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from model.meta_arch.med3d_lisa import Med3DLISA, Med3DLISAConfig
from data.builder import build_dataloader
from deepspeed.moe.utils import is_moe_param, configure_moe_param_groups

os.environ["TOKENIZERS_PARALLELISM"] = "false"
_log_level = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=getattr(logging, _log_level, logging.WARNING),
)
logger = logging.getLogger(__name__)

# Backward-compat for older logging guards
local_rank = int(os.environ.get("LOCAL_RANK", -1))

# Reduce HF/Transformers verbosity by default
try:
    transformers.utils.logging.set_verbosity_warning()
except Exception:
    pass


from train_utils.args_and_collators import ModelArguments, DataArguments, TrainingArgumentsWithLoss, Med3DDataCollator
from train_utils.moe_optimizer import create_optimizer_moe
from train_utils.stage1_manual import run_stage1_manual_alignment

class Med3DLISATrainer(Trainer):
    """
    自定义 Trainer，支持多模态损失计算
    """
    
    def create_optimizer(self):
        """Setup the optimizer with MoE support.

        Refactored: delegating to train_utils.moe_optimizer.create_optimizer_moe
        to keep this file shorter without changing behavior.
        """
        return create_optimizer_moe(self, logger)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算损失函数
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
    """
    # Stage 1 只做对齐，不需要 tokenizer
    if stage == 1:
        logger.debug("Stage 1: Skipping tokenizer loading (not needed for alignment)")
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
    
    logger.debug(f"Added {num_added_tokens} special tokens to tokenizer")
    logger.debug(f"Special tokens: {tokenizer.additional_special_tokens}")
    
    return tokenizer


def build_model(model_args, tokenizer, config=None, stage=None, training_args=None):
    """
    构建 Med3D-LISA 模型
    """
    # Stage 1：只构建对齐模块，不需要完整的 Med3DLISA
    if stage == 1:
        logger.debug("Stage 1: Building alignment-only model (Unified ViT + Distillation)")
        # 【Fix】新的 AlignmentModel 只接受一个 config 参数
        # 所有的参数配置都在 config['model'] 字典里
        from model.alignment import AlignmentModel
        model = AlignmentModel(config) 
        return model
    
    # Stage 2-4：需要完整模型
    # 从 YAML config 中提取模型配置（如果存在）
    vision_tower = model_args.vision_tower
    if config is not None and 'model' in config:
        model_config = config['model']
        # 提取 vision 配置
        if 'vision' in model_config and vision_tower is None:
            vision_cfg = model_config['vision']
            vision_tower = vision_cfg.get('vision_tower')
            if vision_tower == "null" or vision_tower == "":
                vision_tower = None
            logger.debug(f"Using vision_tower from YAML config: {vision_tower}")
    
    # 创建配置
    lisa_config = Med3DLISAConfig(
        model_type=model_args.model_type,
        hidden_size=4096,  # Llama-2-7B
        num_experts=model_args.num_experts,
        num_experts_per_tok=model_args.num_experts_per_tok,
        vision_tower=vision_tower,
        mm_projector_type=model_args.mm_projector_type,
        seg_token_idx=model_args.seg_token_idx,
        image_token_index=model_args.image_token_index,
        sam_type=model_args.sam_type,
        sam_checkpoint=model_args.sam_checkpoint,
        vocab_size=len(tokenizer),
    )
    
    # 强制在 bfloat16 下初始化模型，节省一半内存并避免转换时的 OOM
    # 7B model float32 = 28GB > 24GB VRAM. bfloat16 = 14GB < 24GB VRAM.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # 检查是否使用 4-bit 量化
    if getattr(model_args, 'load_in_4bit', False):
        logger.debug("Initializing model with 4-bit quantization (BitsAndBytes)...")
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        lisa_config.quantization_config = quantization_config
        
        logger.debug(f"Loading 4-bit model via from_pretrained using {model_args.model_name_or_path}...")
        # 使用 from_pretrained 的 "meta device" 初始化机制来避免 OOM
        # 必须传入 strict=False (忽略缺失的 vision_tower 等权重)
        model = Med3DLISA.from_pretrained(
            model_args.model_name_or_path,
            config=lisa_config,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
            strict=False,
            device_map={"": 0}, # 强制映射到当前GPU (0)
            low_cpu_mem_usage=True
        )
        logger.debug("  ✓ 4-bit model loaded successfully")
        
    else:
        logger.debug(f"Initializing model frame with dtype={dtype} to save memory...")
        # 保存当前的默认 dtype
        orig_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            model = Med3DLISA(lisa_config)
        finally:
            torch.set_default_dtype(orig_dtype)

    # 如果添加了新 token，需要调整 embedding 层
    # 注意：resize 后新 embedding 可能是 float32，需要转换
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(dtype)
    
    # 冻结策略
    if model_args.freeze_vision_tower:
        vision_tower = model.get_model().get_vision_tower()
        if vision_tower is not None:
            logger.debug("Freezing vision tower")
            vision_tower.freeze()
            logger.debug("Vision tower frozen")
        else:
            logger.debug("No vision tower to freeze (vision_tower is None)")
    
    if model_args.freeze_llm_backbone:
        logger.debug("Freezing LLM backbone (Main Params)")
        # 冻结所有主干参数
        for _, param in model.named_parameters():
            param.requires_grad = False

        # [Fix] Unfreeze MoE parameters if present (router + experts)
        moe_params = 0
        for name, param in model.named_parameters():
            if is_moe_param(param) or 'experts' in name or 'deepspeed_moe' in name or 'moe_layer' in name:
                param.requires_grad = True
                moe_params += 1

        if moe_params > 0:
            logger.debug(f"Unfrozen {moe_params} MoE parameters for training")
        else:
            logger.warning("freeze_llm_backbone is set, but no MoE parameters were unfrozen; DeepSpeed MoE will fail.")
        
        # 确保梯度检查点仍然可用 (input_require_grad hook)
        if training_args and training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    
    if not model_args.tune_mm_projector:
        logger.debug("Freezing mm_projector")
        for param in model.get_model().mm_projector.parameters():
            param.requires_grad = False
    
    # LoRA
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model
        
        logger.debug("Applying LoRA")
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
    # 如果 sys.argv 中有 --local_rank，但是 TrainingArguments 默认不带 local_rank ...
    # HfArgumentParser 会自动处理大部分参数。
    # 这里我们手动处理一下 config 文件加载，以防 argparse 顺序问题导致配置未生效。
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsWithLoss))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 如果提供了 YAML 配置文件，优先使用
    config = None
    if data_args.config_file is not None:
        import yaml
        logger.debug(f"Loading config from {data_args.config_file}")
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
            logger.debug("✓ Training args overridden from config file")
        
        # 更新 RAG 参数
        if 'model' in config and 'rag' in config['model']:
            rag_config = config['model']['rag']
            if model_args.rag_knowledge_embeddings is None:
                model_args.rag_knowledge_embeddings = rag_config.get('knowledge_embeddings')
            if model_args.rag_knowledge_texts is None:
                model_args.rag_knowledge_texts = rag_config.get('knowledge_texts')
            model_args.rag_enabled = rag_config.get('enabled', True)
            model_args.rag_top_k = rag_config.get('top_k', 3)
            
        # 补充：从 config['model']['llm'] 更新 model_args
        if 'model' in config and 'llm' in config['model']:
            llm_cfg = config['model']['llm']
            if 'lora_enable' in llm_cfg:
                model_args.use_lora = llm_cfg['lora_enable']
            if 'lora_r' in llm_cfg:
                model_args.lora_r = llm_cfg['lora_r']
            if 'lora_alpha' in llm_cfg:
                model_args.lora_alpha = llm_cfg['lora_alpha']
            if 'lora_dropout' in llm_cfg:
                model_args.lora_dropout = llm_cfg['lora_dropout']
            if 'lora_target_modules' in llm_cfg:
                model_args.lora_target_modules = llm_cfg['lora_target_modules']
            
            # 处理 model_name_or_path
            if 'model_name_or_path' in llm_cfg:
                 model_args.model_name_or_path = llm_cfg['model_name_or_path']
            # 处理冻结 - Config 优先
            if 'freeze_llm_backbone' in llm_cfg:
                model_args.freeze_llm_backbone = llm_cfg.get('freeze_llm_backbone', True)
            # 处理 4-bit 量化配置
            if 'load_in_4bit' in llm_cfg:
                model_args.load_in_4bit = llm_cfg['load_in_4bit']
                 
            logger.debug(f"✓ Model args updated from config: use_lora={model_args.use_lora}, load_in_4bit={getattr(model_args, 'load_in_4bit', False)}")

        # 补充：从 config['model']['vision'] 更新 vision args
        if 'model' in config and 'vision' in config['model']:
            vis_cfg = config['model']['vision']
            if 'freeze_vision_tower' in vis_cfg:
                model_args.freeze_vision_tower = vis_cfg['freeze_vision_tower']
            if 'vision_tower' in vis_cfg:
                 model_args.vision_tower = vis_cfg['vision_tower']
        
        # 补充：从 config['model']['moe'] 更新 num_experts
        if 'model' in config and 'moe' in config['model']:
            moe_cfg = config['model']['moe']
            if 'num_experts' in moe_cfg:
                model_args.num_experts = moe_cfg['num_experts']
            if 'num_experts_per_tok' in moe_cfg:
                model_args.num_experts_per_tok = moe_cfg['num_experts_per_tok']
            logger.debug(f"✓ MoE args updated from config: experts={model_args.num_experts}, top_k={model_args.num_experts_per_tok}")
    
    # 检测当前训练阶段（必须在使用stage之前）
    stage = None
    if config is not None and 'training' in config:
        stage = config['training'].get('stage', None)
    
    # 简洁日志：只输出关键训练参数
    # 检测 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.debug(f"Checkpoint detected, resuming training at {last_checkpoint}")
    
    # 加载 tokenizer（Stage 1 跳过）
    logger.debug("Loading tokenizer...")
    tokenizer = load_tokenizer(model_args, stage=stage)
    
    # 构建模型
    logger.debug("Building model...")
    model = build_model(model_args, tokenizer, config=config, stage=stage, training_args=training_args)
    
    # 内存优化：在 Trainer 初始化前将模型转换为半精度
    if training_args.bf16:
        logger.debug("Ensuring model is in bfloat16...")
        model = model.to(torch.bfloat16)
    elif training_args.fp16:
        logger.debug("Ensuring model is in float16...")
        model = model.to(torch.float16)
    
    # 构建数据加载器
    logger.debug("Building dataloaders...")
    
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
        
        logger.debug(f"Data configuration:")
        logger.debug(f"  Image size: {image_size}")
        logger.debug(f"  Num slices: {num_slices}")
        
        train_dataset = LIDCAlignmentDataset(
            data_root=data_args.data_root,
            data_source=data_args.train_json,
            image_size=image_size,
            num_slices=num_slices,
            require_mask=config.get('data', {}).get('require_mask', False)
        )
        
        val_dataset = None
        if data_args.val_json:
            val_dataset = LIDCAlignmentDataset(
                data_root=data_args.data_root,
                data_source=data_args.val_json,
                image_size=image_size,
                num_slices=num_slices,
                require_mask=config.get('data', {}).get('require_mask', False)
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
        
        logger.debug(f"Stage 1 alignment datasets created")
        logger.debug(f"  - Train samples: {len(train_dataset)}")
        if val_dataset:
            logger.debug(f"  - Val samples: {len(val_dataset)}")
    
    elif (config is not None and data_args.dataset_type in ['LIDCDataset', 'LIDCSegmentationDataset', 'LIDCFullDataset']) or \
         (str(stage) in ['2', '3', '4']):
        # 使用新版 LIDC 数据集（从 YAML 配置）
        logger.debug(f"Detected dataset_type: {data_args.dataset_type}")
        logger.debug(f"Loading LIDC dataset from config...")
        from data.builder import build_dataloaders_from_config
        
        train_dataloader, val_dataloader, test_dataloader = build_dataloaders_from_config(
            config, tokenizer=None, stage_name=config.get('training', {}).get('stage_name') if config and 'training' in config else None
        )
        logger.debug(f"Using {data_args.dataset_type} with patient-wise split")
        logger.debug(f"  - Train samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            logger.debug(f"  - Val samples: {len(val_dataloader.dataset)}")
        if test_dataloader:
            logger.debug(f"  - Test samples: {len(test_dataloader.dataset)}")
    else:
        # 使用旧版数据集（保持向后兼容）
        logger.debug(f"Falling back to legacy BTB3D dataset")
        logger.debug(f"  config is not None: {config is not None}")
        logger.debug(f"  dataset_type: {data_args.dataset_type}")
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
        logger.debug(f"Using legacy BTB3D dataset") 
        logger.debug(f"  - Train samples: {len(train_dataloader.dataset)}")
    

    # 初始化 Trainer
    if stage == 1:
        # Stage 1: moved to train_utils.stage1_manual.run_stage1_manual_alignment()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        run_stage1_manual_alignment(model=model, model_args=model_args, training_args=training_args, train_dataloader=train_dataloader, logger=logger, device=device, config=config)
        return

    # ==================== End of Stage 1 Loop ====================
    
    # Stage 2-4：使用完整 Trainer
    
    # 【CRITICAL FIX】禁用 remove_unused_columns，防止 Trainer 过滤掉 'image' 等自定义 keys
    if training_args.remove_unused_columns:
        logger.warning("Forcing remove_unused_columns=False to preserve custom keys ('image', 'mask', etc.) for DataCollator")
        training_args.remove_unused_columns = False
        
    data_collator = Med3DDataCollator(tokenizer)
    trainer = Med3DLISATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    if training_args.do_train:
        logger.debug("*** Training ***")
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
        logger.debug("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
