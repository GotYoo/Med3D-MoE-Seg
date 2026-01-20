#!/bin/bash
# 清理损坏的 HuggingFace 缓存并重新下载模型

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

MODEL_NAME="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CACHE_DIR="/home/wuhanqing/.cache/huggingface/hub"
MODEL_DIR="$CACHE_DIR/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

log_info "=========================================="
log_info "HuggingFace Model Cache Repair Tool"
log_info "=========================================="

# 检查缓存目录
if [ -d "$MODEL_DIR" ]; then
    log_info "Found cached model directory: $MODEL_DIR"
    
    # 检查 .no_exist 目录（表示下载失败）
    if [ -d "$MODEL_DIR/.no_exist" ]; then
        log_warning "Found .no_exist directory - download was incomplete"
        log_info "Removing incomplete download..."
        rm -rf "$MODEL_DIR/.no_exist"
        log_success "Incomplete download removed"
    fi
    
    # 检查是否有有效的快照
    SNAPSHOT_COUNT=$(find "$MODEL_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ "$SNAPSHOT_COUNT" -eq 0 ]; then
        log_warning "No valid snapshots found"
        log_info "Removing entire cache directory..."
        rm -rf "$MODEL_DIR"
        log_success "Cache directory removed"
    else
        log_success "Found $SNAPSHOT_COUNT valid snapshot(s)"
    fi
else
    log_info "No cached model found - will download fresh"
fi

# 下载模型
log_info "Downloading/verifying model: $MODEL_NAME"
log_info "This may take a while..."

python3 << 'PYTHON'
import os
os.environ['HF_HOME'] = '/home/wuhanqing/.cache/huggingface'

from transformers import CLIPImageProcessor, CLIPVisionModel
import sys

model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

try:
    print(f"Downloading image processor...")
    processor = CLIPImageProcessor.from_pretrained(model_name)
    print(f"✓ Image processor downloaded successfully")
    
    print(f"Downloading vision model...")
    model = CLIPVisionModel.from_pretrained(model_name, low_cpu_mem_usage=True)
    print(f"✓ Vision model downloaded successfully")
    
    print(f"\nModel info:")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print(f"  - Image size: {model.config.image_size}")
    print(f"  - Patch size: {model.config.patch_size}")
    print(f"  - Num layers: {model.config.num_hidden_layers}")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed to download model: {e}")
    sys.exit(1)
PYTHON

DOWNLOAD_STATUS=$?

if [ $DOWNLOAD_STATUS -eq 0 ]; then
    log_success "Model downloaded and verified successfully!"
    log_info "Cache location: $MODEL_DIR"
    
    # 显示目录结构
    log_info "Cache structure:"
    tree -L 2 "$MODEL_DIR" 2>/dev/null || find "$MODEL_DIR" -maxdepth 2 -type d | head -10
else
    log_error "Failed to download model"
    log_error "Possible solutions:"
    log_error "  1. Check your internet connection"
    log_error "  2. Check if you can access huggingface.co"
    log_error "  3. Try using a VPN or proxy"
    log_error "  4. Manually download the model and place it in: $MODEL_DIR"
    exit 1
fi

log_info "=========================================="
log_success "Model cache repair completed!"
log_info "=========================================="
