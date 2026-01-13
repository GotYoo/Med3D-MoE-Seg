#!/bin/bash
# 评估脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 评估参数
CHECKPOINT_PATH="./outputs/checkpoint_best.pth"
TEST_DATA_CONFIG="config/dataset_config.json"

# 启动评估
python train_net.py \
    --mode eval \
    --config config/config.json \
    --dataset_config ${TEST_DATA_CONFIG} \
    --checkpoint ${CHECKPOINT_PATH} \
    --output_dir ./eval_results \
    --save_visualizations
