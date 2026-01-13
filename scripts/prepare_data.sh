#!/bin/bash
# 数据准备示例脚本
# 用于将原始医学影像数据划分为训练/验证/测试集

# ============================================================
# 配置参数
# ============================================================

# LIDC数据集JSON文件路径
INPUT_JSON="/home/wuhanqing/Med3D-MoE-Seg/datasets/LIDC-IDRI/processed/LIDC/lidc_dataset.json"

# 基础目录（用于解析JSON中的相对路径，可选）
BASE_DIR="/home/wuhanqing/Med3D-MoE-Seg/datasets/LIDC-IDRI/processed/LIDC"

# 输出目录（存放 train.json, val.json, test.json）
OUTPUT_DIR="/home/wuhanqing/Med3D-MoE-Seg/datasets/LIDC-IDRI/splits"

# 划分比例
TRAIN_RATIO=0.7
VAL_RATIO=0.1
TEST_RATIO=0.2

# 随机种子（保证可重复性）
RANDOM_SEED=42

# ============================================================
# 运行数据准备脚本
# ============================================================

echo "==========================================="
echo "Med3D-MoE-Seg Data Preparation (LIDC)"
echo "==========================================="
echo "Input JSON: $INPUT_JSON"
echo "Output directory: $OUTPUT_DIR"
echo "Split ratios: Train=$TRAIN_RATIO, Val=$VAL_RATIO, Test=$TEST_RATIO"
echo ""

# 检查输入JSON文件是否存在
if [ ! -f "$INPUT_JSON" ]; then
    echo "❌ Error: Input JSON file not found: $INPUT_JSON"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行 Python 脚本
python scripts/prepare_data_split.py \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --base_dir "$BASE_DIR"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✅ Data preparation completed successfully!"
    echo "==========================================="
    echo "Generated files:"
    echo "  - $OUTPUT_DIR/train.json"
    echo "  - $OUTPUT_DIR/val.json"
    echo "  - $OUTPUT_DIR/test.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review the generated JSON files"
    echo "  2. Update config/med3d_lisa_full.yaml with data paths"
    echo "  3. Run training: bash scripts/train_ds.sh"
else
    echo ""
    echo "❌ Data preparation failed. Please check the error messages above."
    exit 1
fi

# ============================================================
# 可选：打印数据集统计信息
# ============================================================

echo ""
echo "==========================================="
echo "Dataset Statistics"
echo "==========================================="

for split in train val test; do
    json_file="$OUTPUT_DIR/${split}.json"
    if [ -f "$json_file" ]; then
        n_samples=$(python -c "import json; print(len(json.load(open('$json_file'))))")
        n_patients=$(python -c "import json; data=json.load(open('$json_file')); print(len(set(item['patient_id'] for item in data)))")
        echo "${split^^} set: $n_samples samples from $n_patients patients"
    fi
done

echo "==========================================="
