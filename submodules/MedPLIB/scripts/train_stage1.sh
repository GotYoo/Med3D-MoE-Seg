# 设置显卡（根据您的实际情况修改）
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN

# 实验名称
exp_name="medplib-7b-stage1-pretrain"
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

# --- 关键路径配置 (请修改为您的实际路径) ---
# 1. 基础 LLM (Vicuna-7b-v1.5)
MODEL_PATH="/mnt/disk4t0/publicData/huggingface_models/vicuna-7b-v1.5"

# 2. 视觉塔 (CLIP)
VISION_TOWER="/home/wuhanqing/MedPLIB/huggingface_models/clip-vit-large-patch14-336"

# 3. 数据集 (LLaVA-Med Alignment JSON)
DATA_PATH="/mnt/disk4t0/publicData/LLaVA-Med/llava_med_alignment_500k.json"

# 4. 图像文件夹 (对应数据集中的图片)
IMAGE_FOLDER="/mnt/disk4t0/publicData/LLaVA-Med/images"

# 启动训练
deepspeed --include=localhost:0,1,2,3 --master_port=61001 train_ds_medplib.py \
    --model_name_or_path $MODEL_PATH \
    --version plain \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone True \
    --bf16 True \
    --output_dir runs/$exp_name \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    2>&1 | tee -a runs/$exp_name/$time.log