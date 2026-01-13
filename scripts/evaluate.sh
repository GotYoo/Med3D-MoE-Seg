#!/bin/bash

###############################################################################
# Med3D-MoE-Seg 评估脚本
# 支持分阶段评估和全面指标计算
###############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 打印分隔线
print_separator() {
    echo "================================================================"
}

###############################################################################
# 参数解析
###############################################################################

STAGE="${1:-all}"  # 默认评估所有阶段
TEST_JSON="${2:-}"  # 可选：自定义测试数据路径
DEVICE="${DEVICE:-cuda}"  # 默认使用 GPU
BATCH_SIZE="${BATCH_SIZE:-2}"  # 默认 batch size
NUM_WORKERS="${NUM_WORKERS:-4}"  # 默认 workers

###############################################################################
# 配置
###############################################################################

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 配置文件路径
CONFIG_FILE="config/multi_dataset_stages.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="config/med3d_lisa_full.yaml"
fi

# 检查点和结果目录
CHECKPOINT_DIR="checkpoints"
RESULT_DIR="eval_results"

# 阶段定义
declare -A STAGE_CONFIGS=(
    ["stage1"]="stage1_alignment"
    ["stage2"]="stage2_rag"
    ["stage3"]="stage3_llm"
    ["stage4"]="stage4_full"
)

###############################################################################
# 环境检查
###############################################################################

check_environment() {
    print_separator
    print_info "检查评估环境..."
    print_separator
    
    # 检查 Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found!"
        exit 1
    fi
    print_success "Python: $(python --version)"
    
    # 检查 CUDA
    if [ "$DEVICE" == "cuda" ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            print_warning "CUDA not available, switching to CPU"
            DEVICE="cpu"
        else
            GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            print_success "GPU: $GPU_NAME"
        fi
    fi
    
    # 检查配置文件
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Config: $CONFIG_FILE"
    
    # 检查评估脚本
    if [ ! -f "eval_net.py" ]; then
        print_error "Evaluation script not found: eval_net.py"
        exit 1
    fi
    print_success "Eval script: eval_net.py"
    
    # 检查依赖
    print_info "Checking Python dependencies..."
    python -c "
import sys
required = ['torch', 'numpy', 'tqdm', 'matplotlib', 'scipy', 'nltk', 'rouge_score']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'Missing packages: {\" \".join(missing)}')
    print(f'Install with: pip install {\" \".join(missing)}')
    sys.exit(1)
" || {
        print_error "Missing dependencies!"
        exit 1
    }
    print_success "All dependencies satisfied"
    
    echo ""
}

###############################################################################
# 检查点验证
###############################################################################

check_checkpoint() {
    local stage=$1
    local checkpoint_path="${CHECKPOINT_DIR}/${stage}_final.pth"
    
    # 尝试多种可能的检查点名称
    if [ ! -f "$checkpoint_path" ]; then
        checkpoint_path="${CHECKPOINT_DIR}/${stage}_best.pth"
    fi
    
    if [ ! -f "$checkpoint_path" ]; then
        checkpoint_path="${CHECKPOINT_DIR}/${stage}.pth"
    fi
    
    if [ ! -f "$checkpoint_path" ]; then
        print_warning "Checkpoint not found for $stage"
        print_info "Tried: ${CHECKPOINT_DIR}/${stage}_final.pth"
        print_info "       ${CHECKPOINT_DIR}/${stage}_best.pth"
        print_info "       ${CHECKPOINT_DIR}/${stage}.pth"
        return 1
    fi
    
    echo "$checkpoint_path"
    return 0
}

###############################################################################
# 评估单个阶段
###############################################################################

evaluate_stage() {
    local stage_num=$1
    local stage_name=$2
    
    print_separator
    print_info "Evaluating ${stage_name}..."
    print_separator
    
    # 检查检查点
    checkpoint_path=$(check_checkpoint "$stage_name")
    if [ $? -ne 0 ]; then
        print_error "Skipping ${stage_name} - checkpoint not found"
        return 1
    fi
    
    print_success "Checkpoint: $checkpoint_path"
    
    # 准备输出目录
    output_dir="${RESULT_DIR}/${stage_name}"
    mkdir -p "$output_dir"
    
    # 构建评估命令
    eval_cmd="python eval_net.py \
        --config $CONFIG_FILE \
        --checkpoint $checkpoint_path \
        --stage $stage_name \
        --output_dir $output_dir \
        --device $DEVICE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS"
    
    # 如果指定了自定义测试数据
    if [ -n "$TEST_JSON" ]; then
        if [ -f "$TEST_JSON" ]; then
            eval_cmd="$eval_cmd --test_json $TEST_JSON"
            print_info "Using custom test data: $TEST_JSON"
        else
            print_warning "Custom test JSON not found: $TEST_JSON"
        fi
    fi
    
    # 执行评估
    print_info "Running evaluation..."
    if $eval_cmd; then
        print_success "${stage_name} evaluation completed!"
        print_info "Results saved to: $output_dir"
        
        # 显示关键指标
        if [ -f "${output_dir}/${stage_name}_metrics.json" ]; then
            print_info "Key metrics:"
            python -c "
import json
with open('${output_dir}/${stage_name}_metrics.json', 'r') as f:
    metrics = json.load(f)
    
if 'dice' in metrics:
    print(f\"  Dice: {metrics['dice']['mean']:.4f} ± {metrics['dice']['std']:.4f}\")
if 'iou' in metrics:
    print(f\"  IoU:  {metrics['iou']['mean']:.4f} ± {metrics['iou']['std']:.4f}\")
if 'bleu4' in metrics:
    print(f\"  BLEU-4: {metrics['bleu4']['mean']:.4f} ± {metrics['bleu4']['std']:.4f}\")
" 2>/dev/null || true
        fi
        
        return 0
    else
        print_error "${stage_name} evaluation failed!"
        return 1
    fi
}

###############################################################################
# 主函数
###############################################################################

main() {
    print_separator
    print_info "Med3D-MoE-Seg Evaluation Script"
    print_separator
    echo ""
    
    # 显示配置
    print_info "Configuration:"
    echo "  Stage:       $STAGE"
    echo "  Device:      $DEVICE"
    echo "  Batch Size:  $BATCH_SIZE"
    echo "  Workers:     $NUM_WORKERS"
    if [ -n "$TEST_JSON" ]; then
        echo "  Test Data:   $TEST_JSON"
    fi
    echo ""
    
    # 环境检查
    check_environment
    
    # 执行评估
    if [ "$STAGE" == "all" ]; then
        print_info "Evaluating all stages..."
        echo ""
        
        success_count=0
        fail_count=0
        
        for stage_num in 1 2 3 4; do
            stage_name="${STAGE_CONFIGS[stage${stage_num}]}"
            
            if evaluate_stage "$stage_num" "$stage_name"; then
                ((success_count++))
            else
                ((fail_count++))
            fi
            
            echo ""
        done
        
        # 最终总结
        print_separator
        print_info "Evaluation Summary"
        print_separator
        print_success "Successful: $success_count stages"
        if [ $fail_count -gt 0 ]; then
            print_warning "Failed: $fail_count stages"
        fi
        
        # 生成汇总报告
        if [ $success_count -gt 0 ]; then
            print_info "Generating overall summary..."
            python -c "
import json
from pathlib import Path

result_dir = Path('$RESULT_DIR')
all_metrics = {}

for stage in ['stage1_alignment', 'stage2_rag', 'stage3_llm', 'stage4_full']:
    metrics_file = result_dir / stage / f'{stage}_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics[stage] = json.load(f)

if all_metrics:
    summary_file = result_dir / 'overall_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'✓ Overall summary saved to: {summary_file}')
" 2>/dev/null || true
        fi
        
    else
        # 评估单个阶段
        if [[ "$STAGE" =~ ^stage[1-4]$ ]]; then
            stage_name="${STAGE_CONFIGS[$STAGE]}"
            evaluate_stage "${STAGE: -1}" "$stage_name"
        elif [[ "$STAGE" =~ ^[1-4]$ ]]; then
            stage_name="${STAGE_CONFIGS[stage${STAGE}]}"
            evaluate_stage "$STAGE" "$stage_name"
        else
            print_error "Invalid stage: $STAGE"
            print_info "Valid options: 1, 2, 3, 4, stage1, stage2, stage3, stage4, all"
            exit 1
        fi
    fi
    
    echo ""
    print_separator
    print_success "Evaluation script completed!"
    print_separator
}

# 使用说明
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    cat << EOF
Usage: bash scripts/evaluate.sh [STAGE] [TEST_JSON]

Arguments:
  STAGE       Stage to evaluate (1-4, stage1-stage4, or 'all') [default: all]
  TEST_JSON   Optional path to custom test data JSON file

Environment Variables:
  DEVICE       Device to use (cuda/cpu) [default: cuda]
  BATCH_SIZE   Batch size for evaluation [default: 2]
  NUM_WORKERS  Number of dataloader workers [default: 4]

Examples:
  # Evaluate all stages
  bash scripts/evaluate.sh all

  # Evaluate stage 2 only
  bash scripts/evaluate.sh 2

  # Evaluate with custom test data
  bash scripts/evaluate.sh stage3 data/custom_test.json

  # Evaluate on CPU with larger batch
  DEVICE=cpu BATCH_SIZE=4 bash scripts/evaluate.sh all

EOF
    exit 0
fi

# 运行主函数
main
