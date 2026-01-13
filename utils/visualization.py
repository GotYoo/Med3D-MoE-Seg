"""
Visualization utilities for evaluation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Tuple
import torch


def visualize_segmentation_3d(image: np.ndarray,
                              pred_mask: np.ndarray,
                              gt_mask: np.ndarray,
                              save_path: str,
                              slice_idx: Optional[int] = None,
                              title: str = "Segmentation Result"):
    """
    可视化 3D 分割结果（选择中间切片）
    
    Args:
        image: CT 图像 [D, H, W]
        pred_mask: 预测掩码 [D, H, W]
        gt_mask: 真实掩码 [D, H, W]
        save_path: 保存路径
        slice_idx: 切片索引（None 则自动选择中间切片）
        title: 标题
    """
    # 选择切片
    if slice_idx is None:
        # 找到 GT 非零的中间切片
        nonzero_slices = np.where(gt_mask.sum(axis=(1, 2)) > 0)[0]
        if len(nonzero_slices) > 0:
            slice_idx = nonzero_slices[len(nonzero_slices) // 2]
        else:
            slice_idx = image.shape[0] // 2
    
    # 提取切片
    img_slice = image[slice_idx]
    pred_slice = pred_mask[slice_idx]
    gt_slice = gt_mask[slice_idx]
    
    # 归一化图像
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：原图 + 真实 mask + 预测 mask
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('CT Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_slice, cmap='gray')
    axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.5 * (gt_slice > 0))
    axes[0, 1].set_title('Ground Truth', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_slice, cmap='gray')
    axes[0, 2].imshow(pred_slice, cmap='Blues', alpha=0.5 * (pred_slice > 0))
    axes[0, 2].set_title('Prediction', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行：对比图
    # True Positive (TP) - 绿色
    # False Positive (FP) - 蓝色
    # False Negative (FN) - 红色
    tp = (pred_slice > 0) & (gt_slice > 0)
    fp = (pred_slice > 0) & (gt_slice == 0)
    fn = (pred_slice == 0) & (gt_slice > 0)
    
    # 创建彩色对比图
    overlay = np.zeros((*img_slice.shape, 3))
    overlay[tp] = [0, 1, 0]  # 绿色: TP
    overlay[fp] = [0, 0, 1]  # 蓝色: FP
    overlay[fn] = [1, 0, 0]  # 红色: FN
    
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].imshow(overlay, alpha=0.5)
    axes[1, 0].set_title('TP/FP/FN Overlay', fontsize=12)
    axes[1, 0].axis('off')
    
    # 添加图例
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='blue', label='False Positive')
    fn_patch = mpatches.Patch(color='red', label='False Negative')
    axes[1, 0].legend(handles=[tp_patch, fp_patch, fn_patch], 
                     loc='upper right', fontsize=8)
    
    # 计算指标并显示
    dice = 2 * tp.sum() / (pred_slice.sum() + gt_slice.sum() + 1e-5)
    iou = tp.sum() / ((pred_slice > 0) | (gt_slice > 0)).sum()
    
    axes[1, 1].axis('off')
    metrics_text = f"""
    Slice Metrics (#{slice_idx}):
    
    Dice: {dice:.4f}
    IoU:  {iou:.4f}
    
    TP pixels: {tp.sum()}
    FP pixels: {fp.sum()}
    FN pixels: {fn.sum()}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                   verticalalignment='center', family='monospace')
    
    # 3D 投影可视化
    axes[1, 2].imshow(pred_mask.max(axis=0), cmap='Blues', alpha=0.7)
    axes[1, 2].imshow(gt_mask.max(axis=0), cmap='Reds', alpha=0.3)
    axes[1, 2].set_title('3D Maximum Projection', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_dict: dict, save_path: str):
    """
    绘制指标对比图
    
    Args:
        metrics_dict: 指标字典 {metric_name: {mean, std, ...}}
        save_path: 保存路径
    """
    # 提取分割指标和文本指标
    seg_metrics = {}
    text_metrics = {}
    
    for name, values in metrics_dict.items():
        if name in ['dice', 'iou', 'precision', 'recall']:
            seg_metrics[name] = values
        elif name in ['bleu1', 'bleu2', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'meteor']:
            text_metrics[name] = values
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 分割指标柱状图
    if seg_metrics:
        names = list(seg_metrics.keys())
        means = [seg_metrics[n]['mean'] for n in names]
        stds = [seg_metrics[n]['std'] for n in names]
        
        x = np.arange(len(names))
        axes[0].bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([n.upper() for n in names], rotation=45)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Segmentation Metrics', fontweight='bold')
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (m, s) in enumerate(zip(means, stds)):
            axes[0].text(i, m + s + 0.02, f'{m:.3f}', 
                        ha='center', va='bottom', fontsize=9)
    
    # 文本生成指标柱状图
    if text_metrics:
        names = list(text_metrics.keys())
        means = [text_metrics[n]['mean'] for n in names]
        stds = [text_metrics[n]['std'] for n in names]
        
        x = np.arange(len(names))
        axes[1].bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([n.upper() for n in names], rotation=45)
        axes[1].set_ylabel('Score')
        axes[1].set_title('Report Generation Metrics', fontweight='bold')
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (m, s) in enumerate(zip(means, stds)):
            axes[1].text(i, m + s + 0.02, f'{m:.3f}', 
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_evaluation_report(metrics: dict, 
                            predictions: list,
                            save_path: str,
                            stage_name: str = "Evaluation"):
    """
    创建 Markdown 评估报告
    
    Args:
        metrics: 聚合后的指标字典
        predictions: 预测结果列表
        save_path: 保存路径
        stage_name: 阶段名称
    """
    lines = []
    
    # 标题
    lines.append(f"# {stage_name} - Evaluation Report\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Total Samples**: {len(predictions)}\n")
    lines.append("\n---\n")
    
    # 分割指标
    lines.append("## 📊 Segmentation Metrics\n")
    lines.append("| Metric | Mean | Std | Min | Max | Median |")
    lines.append("|--------|------|-----|-----|-----|--------|")
    
    seg_metric_names = ['dice', 'iou', 'precision', 'recall', 'hd95', 'asd']
    for name in seg_metric_names:
        if name in metrics:
            m = metrics[name]
            lines.append(f"| {name.upper()} | {m['mean']:.4f} | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} | {m['median']:.4f} |")
    
    lines.append("\n")
    
    # 文本生成指标
    lines.append("## 📝 Report Generation Metrics\n")
    lines.append("| Metric | Mean | Std | Min | Max | Median |")
    lines.append("|--------|------|-----|-----|-----|--------|")
    
    text_metric_names = ['bleu1', 'bleu2', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'meteor']
    for name in text_metric_names:
        if name in metrics:
            m = metrics[name]
            lines.append(f"| {name.upper()} | {m['mean']:.4f} | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} | {m['median']:.4f} |")
    
    lines.append("\n")
    
    # Top 5 最好的样本
    if predictions:
        lines.append("## 🏆 Top 5 Best Predictions (by Dice)\n")
        sorted_preds = sorted(predictions, key=lambda x: x.get('dice', 0), reverse=True)[:5]
        
        for i, pred in enumerate(sorted_preds, 1):
            lines.append(f"### {i}. Sample {pred.get('sample_id', 'unknown')}")
            lines.append(f"- **Dice**: {pred.get('dice', 0):.4f}")
            lines.append(f"- **IoU**: {pred.get('iou', 0):.4f}")
            if 'generated_text' in pred:
                lines.append(f"- **Generated**: {pred['generated_text'][:100]}...")
            lines.append("")
        
        # Top 5 最差的样本
        lines.append("## ⚠️ Top 5 Worst Predictions (by Dice)\n")
        sorted_preds = sorted(predictions, key=lambda x: x.get('dice', 0))[:5]
        
        for i, pred in enumerate(sorted_preds, 1):
            lines.append(f"### {i}. Sample {pred.get('sample_id', 'unknown')}")
            lines.append(f"- **Dice**: {pred.get('dice', 0):.4f}")
            lines.append(f"- **IoU**: {pred.get('iou', 0):.4f}")
            lines.append("")
    
    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# 导入 pandas（用于报告生成）
try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed, some visualization features may not work")
