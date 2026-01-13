"""
Evaluation Metrics for Medical Image Segmentation and Report Generation
"""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class SegmentationMetrics:
    """医学图像分割指标计算"""
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
        """
        Dice Coefficient (DSC)
        
        Args:
            pred: 预测 mask [D, H, W]
            target: 真实 mask [D, H, W]
            smooth: 平滑项
            
        Returns:
            dice: Dice 系数 [0, 1]，越高越好
        """
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return float(dice)
    
    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
        """
        Intersection over Union (IoU) / Jaccard Index
        
        Args:
            pred: 预测 mask
            target: 真实 mask
            smooth: 平滑项
            
        Returns:
            iou: IoU 分数 [0, 1]，越高越好
        """
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)
    
    @staticmethod
    def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray, 
                              voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        95% Hausdorff Distance (HD95)
        
        Args:
            pred: 预测 mask [D, H, W]
            target: 真实 mask [D, H, W]
            voxel_spacing: 体素间距 (mm)
            
        Returns:
            hd95: 95% Hausdorff 距离 (mm)，越小越好
        """
        # 提取边界点
        pred_surface = SegmentationMetrics._get_surface_points(pred)
        target_surface = SegmentationMetrics._get_surface_points(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')
        
        # 应用体素间距
        pred_surface = pred_surface * np.array(voxel_spacing)
        target_surface = target_surface * np.array(voxel_spacing)
        
        # 计算双向距离
        distances_pred_to_target = np.sqrt(((pred_surface[:, None, :] - target_surface[None, :, :]) ** 2).sum(axis=2))
        distances_target_to_pred = distances_pred_to_target.T
        
        # 取最小距离
        min_distances_pred = distances_pred_to_target.min(axis=1)
        min_distances_target = distances_target_to_pred.min(axis=1)
        
        # 计算 95% 百分位数
        hd95_pred = np.percentile(min_distances_pred, 95)
        hd95_target = np.percentile(min_distances_target, 95)
        
        hd95 = max(hd95_pred, hd95_target)
        return float(hd95)
    
    @staticmethod
    def average_surface_distance(pred: np.ndarray, target: np.ndarray,
                                 voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        Average Surface Distance (ASD)
        
        Args:
            pred: 预测 mask
            target: 真实 mask
            voxel_spacing: 体素间距
            
        Returns:
            asd: 平均表面距离 (mm)，越小越好
        """
        pred_surface = SegmentationMetrics._get_surface_points(pred)
        target_surface = SegmentationMetrics._get_surface_points(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')
        
        # 应用体素间距
        pred_surface = pred_surface * np.array(voxel_spacing)
        target_surface = target_surface * np.array(voxel_spacing)
        
        # 计算双向距离
        distances_pred_to_target = np.sqrt(((pred_surface[:, None, :] - target_surface[None, :, :]) ** 2).sum(axis=2))
        distances_target_to_pred = distances_pred_to_target.T
        
        # 平均距离
        mean_distance_pred = distances_pred_to_target.min(axis=1).mean()
        mean_distance_target = distances_target_to_pred.min(axis=1).mean()
        
        asd = (mean_distance_pred + mean_distance_target) / 2.0
        return float(asd)
    
    @staticmethod
    def _get_surface_points(mask: np.ndarray) -> np.ndarray:
        """提取表面点"""
        from scipy.ndimage import binary_erosion
        
        # 腐蚀操作找边界
        eroded = binary_erosion(mask, iterations=1)
        surface = mask & ~eroded
        
        # 获取表面点坐标
        surface_points = np.argwhere(surface)
        return surface_points
    
    @staticmethod
    def precision_recall(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """
        Precision and Recall
        
        Returns:
            precision, recall
        """
        pred = pred.flatten()
        target = target.flatten()
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        
        return float(precision), float(recall)
    
    @staticmethod
    def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                           voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
        """
        计算所有分割指标
        
        Args:
            pred: 预测 mask (binary)
            target: 真实 mask (binary)
            voxel_spacing: 体素间距
            
        Returns:
            metrics: 包含所有指标的字典
        """
        metrics = {}
        
        # Dice 和 IoU
        metrics['dice'] = SegmentationMetrics.dice_coefficient(pred, target)
        metrics['iou'] = SegmentationMetrics.iou_score(pred, target)
        
        # Precision 和 Recall
        precision, recall = SegmentationMetrics.precision_recall(pred, target)
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Hausdorff Distance 和 ASD
        try:
            metrics['hd95'] = SegmentationMetrics.hausdorff_distance_95(pred, target, voxel_spacing)
            metrics['asd'] = SegmentationMetrics.average_surface_distance(pred, target, voxel_spacing)
        except Exception as e:
            print(f"Warning: Failed to compute HD95/ASD: {e}")
            metrics['hd95'] = float('inf')
            metrics['asd'] = float('inf')
        
        return metrics


class ReportGenerationMetrics:
    """医学报告生成指标计算"""
    
    @staticmethod
    def bleu_score(references: List[str], hypothesis: str, n: int = 4) -> float:
        """
        BLEU Score
        
        Args:
            references: 参考报告列表
            hypothesis: 生成的报告
            n: n-gram
            
        Returns:
            bleu: BLEU 分数
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Tokenize
            ref_tokens = [ref.split() for ref in references]
            hyp_tokens = hypothesis.split()
            
            # 计算 BLEU
            weights = tuple([1.0 / n] * n)
            smoothing = SmoothingFunction().method1
            
            bleu = sentence_bleu(ref_tokens, hyp_tokens, weights=weights, 
                                smoothing_function=smoothing)
            return float(bleu)
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def rouge_score(references: List[str], hypothesis: str) -> Dict[str, float]:
        """
        ROUGE Score
        
        Args:
            references: 参考报告列表
            hypothesis: 生成的报告
            
        Returns:
            rouge_scores: ROUGE-1, ROUGE-2, ROUGE-L
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            # 对所有参考计算 ROUGE，取最大值
            max_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
            for ref in references:
                scores = scorer.score(ref, hypothesis)
                for key in max_scores:
                    max_scores[key] = max(max_scores[key], scores[key].fmeasure)
            
            return {
                'rouge1': max_scores['rouge1'],
                'rouge2': max_scores['rouge2'],
                'rougeL': max_scores['rougeL']
            }
        except Exception as e:
            print(f"Warning: ROUGE calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    @staticmethod
    def meteor_score(references: List[str], hypothesis: str) -> float:
        """
        METEOR Score
        
        Args:
            references: 参考报告列表
            hypothesis: 生成的报告
            
        Returns:
            meteor: METEOR 分数
        """
        try:
            from nltk.translate.meteor_score import meteor_score as nltk_meteor
            
            # Tokenize
            ref_tokens = [ref.split() for ref in references]
            hyp_tokens = hypothesis.split()
            
            # 对所有参考计算 METEOR，取最大值
            scores = [nltk_meteor(ref, hyp_tokens) for ref in ref_tokens]
            return max(scores) if scores else 0.0
        except Exception as e:
            print(f"Warning: METEOR calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def cider_score(references_list: List[List[str]], hypotheses: List[str]) -> float:
        """
        CIDEr Score (需要多个样本计算)
        
        Args:
            references_list: 所有样本的参考报告 [[ref1_1, ref1_2], [ref2_1], ...]
            hypotheses: 所有样本的生成报告
            
        Returns:
            cider: CIDEr 分数
        """
        try:
            from pycocoevalcap.cider.cider import Cider
            
            # 转换为 COCO 格式
            gts = {i: refs for i, refs in enumerate(references_list)}
            res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
            
            scorer = Cider()
            score, _ = scorer.compute_score(gts, res)
            return float(score)
        except Exception as e:
            print(f"Warning: CIDEr calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def compute_all_metrics(references: List[str], hypothesis: str) -> Dict[str, float]:
        """
        计算所有文本生成指标
        
        Args:
            references: 参考报告列表
            hypothesis: 生成的报告
            
        Returns:
            metrics: 包含所有指标的字典
        """
        metrics = {}
        
        # BLEU scores
        metrics['bleu1'] = ReportGenerationMetrics.bleu_score(references, hypothesis, n=1)
        metrics['bleu2'] = ReportGenerationMetrics.bleu_score(references, hypothesis, n=2)
        metrics['bleu4'] = ReportGenerationMetrics.bleu_score(references, hypothesis, n=4)
        
        # ROUGE scores
        rouge_scores = ReportGenerationMetrics.rouge_score(references, hypothesis)
        metrics.update(rouge_scores)
        
        # METEOR
        metrics['meteor'] = ReportGenerationMetrics.meteor_score(references, hypothesis)
        
        return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    聚合多个样本的指标
    
    Args:
        metrics_list: 每个样本的指标列表
        
    Returns:
        aggregated: 包含 mean, std, min, max 的字典
    """
    if not metrics_list:
        return {}
    
    # 收集所有指标名称
    metric_names = set()
    for metrics in metrics_list:
        metric_names.update(metrics.keys())
    
    aggregated = {}
    
    for name in metric_names:
        values = [m[name] for m in metrics_list if name in m and np.isfinite(m[name])]
        
        if values:
            aggregated[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        else:
            aggregated[name] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
    
    return aggregated
