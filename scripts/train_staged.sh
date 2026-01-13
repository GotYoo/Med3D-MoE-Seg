#!/bin/bash
# Med3D-MoE-Seg 分阶段训练脚本

set -e  # 遇到错误立即退出

# ==================== 环境变量配置 ====================
# 激活cv环境
source ~/anaconda3/bin/activate cv

# 只使用GPU 0（单卡训练，避免DataParallel导致batch_size被分割）
export CUDA_VISIBLE_DEVICES=3

# RTX 4000 系列显卡 NCCL 配置
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN  # 只显示警告和错误，不显示详细INFO

# 内存优化配置（PyTorch 2.x推荐的环境变量）
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

# ==================== 配置 ====================
PROJECT_ROOT="/home/wuhanqing/Med3D-MoE-Seg"
DATA_ROOT="datasets/LIDC-IDRI/processed/LIDC"
cd "$PROJECT_ROOT"

# 解析命令行参数
STAGE=${1:-"all"}  # 默认执行所有阶段
SKIP_STAGES=${2:-""}  # 跳过的阶段（逗号分隔）

# ==================== 辅助函数 ====================
log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

should_skip_stage() {
    local stage=$1
    if [[ ",$SKIP_STAGES," == *",$stage,"* ]]; then
        return 0
    fi
    return 1
}

# ==================== Stage 1: 多模态对齐 ====================
train_stage1() {
    log_info "=========================================="
    log_info "Stage 1: Multi-Modal Alignment Training"
    log_info "=========================================="
    
    if should_skip_stage "1"; then
        log_info "Skipping Stage 1 (as requested)"
        return 0
    fi
    
    log_info "Training vision-text alignment..."
    log_info "Components: CT-CLIP + MedPLIB + BioBERT (No LLM needed)"
    log_info "Duration: ~100 epochs"
    
    python train_net.py \
        --config_file config/stage1_alignment.yaml \
        --data_root datasets/LIDC-IDRI/processed/LIDC \
        --train_json datasets/LIDC-IDRI/splits/train.json \
        --val_json datasets/LIDC-IDRI/splits/val.json \
        --output_dir outputs/stage1_alignment \
        2>&1 | tee logs/stage1_alignment.log
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Stage 1 completed successfully!"
        log_info "Best checkpoint: outputs/stage1_alignment/checkpoints/best_model"
    else
        log_error "Stage 1 failed with exit code: $TRAIN_EXIT_CODE"
        log_error "Check logs/stage1_alignment.log for details"
        exit 1
    fi
}

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
    
    # 检查Stage 2 checkpoint
    if [ ! -f "outputs/stage2_segmentation/checkpoints/best_model/sam.pt" ]; then
        log_error "Stage 2 checkpoint not found! Please run Stage 2 first."
        exit 1
    fi
    
    log_info "Training LLM with MoE for report generation..."
    log_info "Components: LLaMA-2 (LoRA) + MoE Router"
    log_info "Duration: ~35 epochs"
    
    python train_net.py \
        --config_file config/stage3_moe_llm.yaml \
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
        --config_file config/stage4_end2end.yaml \
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
