#!/bin/bash
# DeepSpeed 分布式训练脚本
# Med3D-MoE-Seg Training with DeepSpeed

# ================================
# 环境配置
# ================================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

# 设置 MASTER_PORT（避免端口冲突）
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-localhost}

# ================================
# 模型与数据配置
# ================================
# LLaMA-2 模型路径
MODEL_NAME_OR_PATH="/path/to/llama-2-7b-hf"

# Vision Tower (CT-CLIP) 路径
VISION_TOWER="/path/to/ct_clip_pretrained_weights.pth"

# SAM-Med3D 路径
SAM_CHECKPOINT="/path/to/sam_med3d_checkpoint.pth"

# 数据路径
DATA_ROOT="/path/to/nifti/data"
ANN_FILE="/path/to/annotations.json"

# 输出目录
OUTPUT_DIR="./checkpoints/med3d_lisa_$(date +%Y%m%d_%H%M%S)"

# ================================
# 训练超参数
# ================================
NUM_GPUS=8
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=10
WARMUP_STEPS=500
SAVE_STEPS=500
LOGGING_STEPS=10

# ================================
# DeepSpeed 配置
# ================================
# 使用 DeepSpeed ZeRO Stage 2 配置
DS_CONFIG="./config/ds_config_zero2.json"

# 如果配置文件不存在，创建一个默认的
if [ ! -f "$DS_CONFIG" ]; then
    echo "Creating default DeepSpeed config at $DS_CONFIG"
    mkdir -p ./config
    cat > "$DS_CONFIG" << 'EOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.0
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  "zero_allow_untested_optimizer": true,
  "wall_clock_breakdown": false
}
EOF
fi

# ================================
# MoE 配置
# ================================
NUM_EXPERTS=8
NUM_EXPERTS_PER_TOK=2

# ================================
# 启动训练
# ================================
echo "=========================================="
echo "Med3D-MoE-Seg Training with DeepSpeed"
echo "=========================================="
echo "Number of GPUs: ${NUM_GPUS}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Vision Tower: ${VISION_TOWER}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "DeepSpeed Config: ${DS_CONFIG}"
echo "=========================================="

deepspeed --num_gpus=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train_net.py \
    --deepspeed ${DS_CONFIG} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --vision_tower ${VISION_TOWER} \
    --sam_checkpoint ${SAM_CHECKPOINT} \
    --data_root ${DATA_ROOT} \
    --ann_file ${ANN_FILE} \
    --mm_projector_type mlp2x_gelu \
    --num_experts ${NUM_EXPERTS} \
    --num_experts_per_tok ${NUM_EXPERTS_PER_TOK} \
    --freeze_vision_tower True \
    --freeze_llm_backbone False \
    --tune_mm_projector True \
    --bf16 True \
    --fp16 False \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 3 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.0 \
    --warmup_steps ${WARMUP_STEPS} \
    --lr_scheduler_type "cosine" \
    --logging_steps ${LOGGING_STEPS} \
    --report_to "wandb" \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --gradient_checkpointing True \
    --do_train \
    --overwrite_output_dir

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo "=========================================="
