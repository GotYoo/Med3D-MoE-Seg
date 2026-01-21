#!/bin/bash
# Med3D-MoE-Seg 4090 调试专用脚本

set -e  # 遇到错误立即退出

# ==================== 环境变量配置 ====================
# 激活环境 (请确保路径正确)
source ~/anaconda3/bin/activate cv

# 指定单卡
export CUDA_VISIBLE_DEVICES=0

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