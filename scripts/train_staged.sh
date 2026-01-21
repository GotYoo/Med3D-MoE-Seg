#!/bin/bash
# Med3D-MoE-Seg 4090 调试专用脚本

set -e  # 遇到错误立即退出

# ==================== 环境变量配置 ====================
# 激活环境 (请确保路径正确)
source ~/anaconda3/bin/activate cv

# 指定单卡
export CUDA_VISIBLE_DEVICES=1

# RTX 4090 专用 NCCL/CUDA 配置
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN 

# 显存碎片整理 (PyTorch 2.x)
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ==================== 配置 ====================
PROJECT_ROOT="/home/wuhanqing/Med3D-MoE-Seg"
cd "$PROJECT_ROOT"

# ==================== 辅助函数 ====================
log_info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
log_error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

# ==================== Stage 1: 调试运行 ====================
train_stage1_debug() {
    log_info "=========================================="
    log_info "Stage 1: Debugging on RTX 4090"
    log_info "Config: config/stage1_debug_4090.yaml"
    log_info "=========================================="
    
    # 确保 log 目录存在
    mkdir -p logs/debug

    # 【关键修改】
    # 1. 使用 debug 配置文件
    # 2. 如果你有 debug.json，请替换下面的 train_json 路径
    # 3. 移除了后台管道，直接在终端输出，方便看报错或 pdb 交互
    
    python train_net.py \
        --do_train \
        --config_file config/stage1_debug_4090.yaml \
        --data_root datasets/LIDC-IDRI/processed/LIDC \
        --train_json datasets/LIDC-IDRI/splits/debug.json \
        --val_json datasets/LIDC-IDRI/splits/val.json \
        --output_dir outputs/debug_4090 \
        --report_to none  # 调试时不上传 wandb
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Debug run finished successfully!"
    else
        log_error "Debug run failed with exit code: $TRAIN_EXIT_CODE"
        exit 1
    fi
}

# 执行
train_stage1_debug

# ==================== Stage 2: 分割器微调 ====================
train_stage2() {
    log_info "=========================================="
    log_info "Stage 2: SAM-Med3D Segmentation Training"
    log_info "=========================================="
    
    if should_skip_stage "2"; then
        log_info "Skipping Stage 2 (as requested)"
        return 0
    fi
    
    # 检查Stage 1 checkpoint
    if [ ! -f "outputs/stage1_alignment/checkpoints/best_model/vision_tower.pt" ]; then
        log_error "Stage 1 checkpoint not found! Please run Stage 1 first."
        exit 1
    fi
    
    log_info "Training SAM-Med3D for segmentation..."
    log_info "Components: SAM (with LoRA)"
    log_info "Duration: ~40 epochs"
    
    python train_net.py \
        --do_train \
        --config_file config/stage2_segmentation.yaml \
        --model_name_or_path gpt2 \
        --data_root ${DATA_ROOT} \
        --output_dir outputs/stage2_segmentation \
        2>&1 | tee logs/stage2_segmentation.log
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Stage 2 completed successfully!"
        log_info "Best checkpoint: outputs/stage2_segmentation/checkpoints/best_model"
    else
        log_error "Stage 2 failed with exit code: $TRAIN_EXIT_CODE"
        log_error "Check logs/stage2_segmentation.log for details"
        exit 1
    fi
}

# ==================== Stage 3: MoE + LLM训练 ====================
train_stage3() {
    log_info "=========================================="
    log_info "Stage 3: MoE + LLM Training"
    log_info "=========================================="
    
    if should_skip_stage "3"; then
        log_info "Skipping Stage 3 (as requested)"
        return 0
    fi
    
    # 检查 Stage 2 checkpoint 或预训练 SAM
    if [ ! -f "outputs/stage2_segmentation/checkpoints/best_model/sam.pt" ]; then
        log_info "Stage 2 checkpoint not found, checking for pretrained SAM..."
        if [ ! -f "checkpoints/sam_med3d_turbo.pth" ]; then
            log_error "Neither Stage 2 checkpoint nor pretrained SAM found!"
            log_error "Please download SAM-Med3D to checkpoints/sam_med3d_turbo.pth"
            exit 1
        fi
        log_info "✓ Using pretrained SAM: checkpoints/sam_med3d_turbo.pth"
    else
        log_info "✓ Using Stage 2 checkpoint: outputs/stage2_segmentation/checkpoints/best_model/sam.pt"
    fi
    
    log_info "Training LLM with MoE for report generation..."
    log_info "Components: LLaMA-2 (LoRA) + MoE Router"
    log_info "Duration: ~35 epochs"
    
    # 移除手动设置的分布式环境变量，交由 torchrun 处理
    # 也不要使用 DeepSpeed 的 Launcher，尽量保持简单
    export MASTER_ADDR=localhost
    export MASTER_PORT=29506

    # 使用 DeepSpeed ZeRO-3 (无 Offload)
    # 使用 torchrun 启动多卡训练
    # 指定 GPU 0 和 2 (根据用户需求)
    export CUDA_VISIBLE_DEVICES=0,1,2
    # 显存优化配置
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
    export DS_SKIP_COMPLIANCE_CHECK=1
    
    # 使用 DeepSpeed ZeRO-2 (带 Offload) 以支持 MoE 并节省显存
    # 同时防止 MoE not supported with Stage 3 错误
    torchrun --nproc_per_node=3 --master_port=29506 train_net.py \
        --do_train \
        --bf16 \
        --deepspeed config/ds_config_zero2_no_offload.json \
        --config_file config/stage3_moe_llm.yaml \
        --data_root ${DATA_ROOT} \
        --report_to none \
        --output_dir outputs/stage3_moe_llm \
        2>&1 | tee logs/stage3_moe_llm.log
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Stage 3 completed successfully!"
        log_info "Best checkpoint: outputs/stage3_moe_llm/checkpoints/best_model"
    else
        log_error "Stage 3 failed with exit code: $TRAIN_EXIT_CODE"
        log_error "Check logs/stage3_moe_llm.log for details"
        exit 1
    fi
}

# ==================== Stage 4: 端到端微调 ====================
train_stage4() {
    log_info "=========================================="
    log_info "Stage 4: End-to-End Fine-tuning"
    log_info "=========================================="
    
    if should_skip_stage "4"; then
        log_info "Skipping Stage 4 (as requested)"
        return 0
    fi
    
    # 检查Stage 3 checkpoint
    if [ ! -f "outputs/stage3_moe_llm/checkpoints/best_model/llm_lora.pt" ]; then
        log_error "Stage 3 checkpoint not found! Please run Stage 3 first."
        exit 1
    fi
    
    log_info "End-to-end fine-tuning with RAG and self-correction..."
    log_info "Components: All (with small learning rates)"
    log_info "Duration: ~20 epochs"
    
    python train_net.py \
        --do_train \
        --config_file config/stage4_end2end.yaml \
        --data_root ${DATA_ROOT} \
        --output_dir outputs/stage4_end2end \
        2>&1 | tee logs/stage4_end2end.log
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Stage 4 completed successfully!"
        log_info "Best checkpoint: outputs/stage4_end2end/checkpoints/best_model"
    else
        log_error "Stage 4 failed with exit code: $TRAIN_EXIT_CODE"
        log_error "Check logs/stage4_end2end.log for details"
        exit 1
    fi
}

# ==================== 主函数 ====================
main() {
    # 创建日志目录
    mkdir -p logs
    
    log_info "============================================"
    log_info "Med3D-MoE-Seg Staged Training Pipeline"
    log_info "============================================"
    log_info "Stage to run: $STAGE"
    if [ -n "$SKIP_STAGES" ]; then
        log_info "Skipping stages: $SKIP_STAGES"
    fi
    echo ""
    
    # 根据参数执行对应阶段
    case $STAGE in
        "1")
            train_stage1
            ;;
        "2")
            train_stage2
            ;;
        "3")
            train_stage3
            ;;
        "4")
            train_stage4
            ;;
        "all")
            train_stage1
            train_stage2
            train_stage3
            train_stage4
            ;;
        "1-2")
            train_stage1
            train_stage2
            ;;
        "2-3")
            train_stage2
            train_stage3
            ;;
        "3-4")
            train_stage3
            train_stage4
            ;;
        *)
            log_error "Invalid stage: $STAGE"
            log_info "Usage: bash scripts/train_staged.sh [STAGE] [SKIP_STAGES]"
            log_info "  STAGE: 1, 2, 3, 4, all, 1-2, 2-3, 3-4"
            log_info "  SKIP_STAGES: comma-separated stages to skip (e.g., '1,2')"
            log_info ""
            log_info "Examples:"
            log_info "  bash scripts/train_staged.sh all          # Train all stages"
            log_info "  bash scripts/train_staged.sh 1             # Train only Stage 1"
            log_info "  bash scripts/train_staged.sh 2-3           # Train Stage 2 and 3"
            log_info "  bash scripts/train_staged.sh all 1,2       # Train all, skip 1&2"
            exit 1
            ;;
    esac
    
    log_success "============================================"
    log_success "Training pipeline completed!"
    log_success "============================================"
    log_info "Final model: outputs/stage4_end2end/checkpoints/best_model"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Evaluate on test set: bash scripts/evaluate.sh"
    log_info "  2. Run inference: bash scripts/inference.sh"
    log_info "  3. View logs: tensorboard --logdir outputs/"
}

# 执行主函数
main
