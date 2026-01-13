# Med3D-MoE-Seg 评估系统使用指南

## 📋 快速开始

### 评估第一阶段（对齐训练）

```bash
cd /home/wuhanqing/Med3D-MoE-Seg

# 方法1: 使用 Shell 脚本
bash scripts/evaluate.sh stage1

# 方法2: 直接使用 Python
python eval_net.py \
  --config config/multi_dataset_stages.yaml \
  --checkpoint outputs/stage1_alignment/checkpoints/best_model/alignment_model.pt \
  --stage stage1_alignment \
  --output_dir eval_results/stage1 \
  --device cuda \
  --batch_size 2
```

## 📊 评估指标说明

### 分割指标
- **Dice Coefficient**: 预测和真实掩码的重叠度 [0-1]，越高越好
- **IoU**: 交并比 [0-1]，越高越好  
- **HD95**: 95% Hausdorff距离（mm），越小越好
- **ASD**: 平均表面距离（mm），越小越好

### 文本生成指标
- **BLEU-1/2/4**: N-gram匹配度 [0-1]
- **ROUGE-L**: 最长公共子序列 [0-1]
- **METEOR**: 考虑同义词的匹配度 [0-1]

## 📁 输出文件

评估完成后在 `eval_results/stage1/` 生成：
- `stage1_alignment_metrics.json` - 聚合指标
- `stage1_alignment_predictions.json` - 每个样本的预测
- `stage1_alignment_report.md` - Markdown报告
- `stage1_alignment_metrics_comparison.png` - 指标对比图
- `visualizations/*.png` - 每个样本的可视化
