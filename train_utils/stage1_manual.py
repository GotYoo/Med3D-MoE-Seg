"""Stage 1 manual alignment training loop (extracted).

Refactor-only move of the original Stage-1 block from train_net.py.
Logic is preserved; only relocated.
"""

from __future__ import annotations

import os

import torch
import transformers
from transformers import AutoTokenizer


def run_stage1_manual_alignment(*, model, model_args, training_args, train_dataloader, logger, device, config=None):
    """Run the original Stage-1 manual alignment loop."""
    print("Starting Stage 1: Manual Alignment Training Loop...")

    # 1. 准备 Tokenizer
    # 【Fix】从 config 字典中获取，而不是从 model_args 获取
    tokenizer_name = "dmis-lab/biobert-v1.1" # 默认值
    if config is not None:
        # 尝试从 config['model']['text_encoder']['model_name'] 获取
        tokenizer_name = config.get('model', {}).get('text_encoder', {}).get('model_name', tokenizer_name)
    
    logger.info(f"Loading tokenizer from: {tokenizer_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.debug(f"Loaded tokenizer: {tokenizer.name_or_path}")

    # 2. 准备优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )

    # 3. 准备 Scheduler
    num_training_steps = len(train_dataloader) * training_args.num_train_epochs
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.03 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # 4. 获取模型的数据类型 (用于 BF16 转换)
    base_model = model.module if hasattr(model, 'module') else model
    model_dtype = next(base_model.parameters()).dtype
    logger.debug(f"Model is running in dtype: {model_dtype}")

    model.train()
    global_step = 0

    for epoch in range(int(training_args.num_train_epochs)):
        print(f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")

        for step, batch in enumerate(train_dataloader):
            # --- A. 数据预处理 (Tokenization) ---
            # 从 batch 中提取原始文本
            reports = batch.pop('report', batch.pop('text_raw', None))

            if reports is None:
                # 如果是 list 的 tuple，解包
                if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], str):
                     reports = batch
                else:
                    logger.warning(f"No 'report' found in batch keys: {batch.keys()}. Using dummy.")
                    reports = ["normal chest ct scan"] * (len(batch['ct_volume']) if 'ct_volume' in batch else 1)

            # 实时分词
            text_inputs = tokenizer(
                reports,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 将 input_ids 和 attention_mask 放入 text_inputs 字典
            batch['text_inputs'] = {
                'input_ids': text_inputs['input_ids'].to(device),
                'attention_mask': text_inputs['attention_mask'].to(device)
            }

            # --- B. 类型转换 (Fix Input Type Mismatch) ---
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # 图像数据 float32 -> bfloat16
                    if v.dtype in [torch.float32, torch.float64]:
                        batch[k] = v.to(device, dtype=model_dtype)
                    else:
                        # 标签/Mask/InputIDs 保持原样
                        batch[k] = v.to(device)

            # --- C. Forward & Backward ---
            optimizer.zero_grad()

            # 显式开启 Autocast
            # 注意: 4090 必须用 dtype=torch.bfloat16
            with torch.cuda.amp.autocast(dtype=model_dtype):
                # 确保只传入 forward 接受的参数，防止多余参数报错
                # model(ct_volume, text_inputs, ...)
                outputs = model(**batch)

            loss = outputs['loss']

            # Backward
            loss.backward()

            # Clip Grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # --- D. Logging ---
            if step % training_args.logging_steps == 0:
                log_msg = f"Step {step}: Loss={loss.item():.4f}"
                if 'loss_distill' in outputs:
                    log_msg += f" | Distill={outputs['loss_distill'].item():.4f}"
                if 'loss_local' in outputs:
                    log_msg += f" | Local={outputs['loss_local'].item():.4f}"
                print(log_msg)

            global_step += 1

        # Epoch Save
        if training_args.save_strategy != "no":
            save_path = os.path.join(training_args.output_dir, f"checkpoint-{epoch}")
            os.makedirs(save_path, exist_ok=True)
        
            logger.info(f"Saving checkpoint to {save_path}")
        
            # 【Fix】手动保存 State Dict，兼容所有模型类型
            student = base_model.student_model
            torch.save(student.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
            # 尝试保存 Config (如果有)
            if hasattr(student, "config"):
                try:
                    student.config.save_pretrained(save_path)
                except:
                    pass # 忽略 config 保存错误
        
            # 保存 Tokenizer
            tokenizer.save_pretrained(save_path)

    print("Stage 1 Training Completed.")
    return  # 结束 Stage 1，不走后面的逻辑
