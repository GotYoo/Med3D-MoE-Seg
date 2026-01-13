"""
Stage 1 对齐数据集
加载 CT Volume、CT Slices 和医学报告，用于多模态对齐训练
"""

import os
import json
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional
import torchvision.transforms as transforms


class LIDCAlignmentDataset(Dataset):
    """LIDC-IDRI 对齐数据集（Stage 1）"""
    
    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        image_size: tuple = (128, 128, 128),
        num_slices: int = 16,
        transform=None,
    ):
        """
        Args:
            data_root: NIfTI 文件根目录（用于相对路径，如果 JSON 中是绝对路径则忽略）
            annotation_file: JSON 标注文件路径（train.json/val.json）
            image_size: 3D 图像尺寸 (D, H, W)
            num_slices: 采样的 2D 切片数量
            transform: 数据增强
        """
        self.data_root = Path(data_root) if data_root else None
        self.image_size = image_size
        self.num_slices = num_slices
        self.transform = transform
        
        # 加载标注
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        print(f"Loaded {len(self.annotations)} samples from {annotation_file}")
    
    def __len__(self):
        return len(self.annotations)
    
    def load_nifti(self, nifti_path: str) -> np.ndarray:
        """加载 NIfTI 文件"""
        # 如果是绝对路径，直接使用
        if os.path.isabs(nifti_path):
            full_path = Path(nifti_path)
        else:
            # 相对路径，拼接 data_root
            full_path = self.data_root / nifti_path
        
        # 检查文件是否存在
        if not full_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {full_path}")
        
        # print(f"Loading {full_path}...")  # 调试日志
        nii = nib.load(str(full_path))
        data = nii.get_fdata()
        return data
    
    def preprocess_volume(self, volume: np.ndarray) -> torch.Tensor:
        """
        预处理 3D Volume
        - Resize 到目标尺寸
        - 归一化
        """
        from scipy.ndimage import zoom
        
        # 计算缩放因子
        current_shape = volume.shape
        zoom_factors = [
            self.image_size[0] / current_shape[0],
            self.image_size[1] / current_shape[1],
            self.image_size[2] / current_shape[2],
        ]
        
        # Resize
        volume = zoom(volume, zoom_factors, order=1)
        
        # 归一化到 [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # 转换为 Tensor [C, D, H, W]
        volume = torch.from_numpy(volume).float().unsqueeze(0)
        
        return volume
    
    def sample_slices(self, volume: np.ndarray, num_slices: int) -> torch.Tensor:
        """
        从 3D Volume 中均匀采样 2D 切片
        
        Returns:
            slices: [N, C, H, W]
        """
        D, H, W = volume.shape
        
        # 均匀采样索引
        indices = np.linspace(0, D - 1, num_slices, dtype=int)
        
        slices = []
        for idx in indices:
            slice_2d = volume[idx, :, :]  # [H, W]
            
            # 归一化
            slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            
            # 转换为 3 通道（模拟 RGB）
            slice_2d = np.stack([slice_2d, slice_2d, slice_2d], axis=0)  # [3, H, W]
            
            # Resize 到 336x336（CLIP要求的输入尺寸）
            from scipy.ndimage import zoom
            slice_2d = zoom(slice_2d, (1, 336 / H, 336 / W), order=1)
            
            slices.append(torch.from_numpy(slice_2d).float())
        
        return torch.stack(slices, dim=0)  # [N, 3, 336, 336]
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回一个样本
        
        Returns:
            {
                'ct_volume': [C, D, H, W],
                'ct_slices': [N, C, H, W],
                'text_report': str,
                'scan_id': str,
            }
        """
        ann = self.annotations[idx]
        
        # 1. 加载 3D CT Volume
        volume = self.load_nifti(ann['image_path'])
        ct_volume = self.preprocess_volume(volume)
        
        # 2. 采样 2D Slices
        ct_slices = self.sample_slices(volume, self.num_slices)
        
        # 3. 获取文本报告
        text_report = ann.get('text_report', '')
        
        return {
            'ct_volume': ct_volume,
            'ct_slices': ct_slices,
            'text_report': text_report,
            'scan_id': ann.get('scan_id', f'sample_{idx}'),
        }


def alignment_collate_fn(batch, tokenizer):
    """
    Stage 1 对齐的 collate 函数
    
    Args:
        batch: List[Dict]
        tokenizer: BioBERT tokenizer
    
    Returns:
        {
            'ct_volume': [B, C, D, H, W],
            'ct_slices': [B, N, C, H, W],
            'text_inputs': {'input_ids': [B, L], 'attention_mask': [B, L]},
            'scan_ids': List[str],
        }
    """
    ct_volumes = []
    ct_slices_list = []
    text_reports = []
    scan_ids = []
    
    for item in batch:
        ct_volumes.append(item['ct_volume'])
        ct_slices_list.append(item['ct_slices'])
        text_reports.append(item['text_report'])
        scan_ids.append(item['scan_id'])
    
    # Stack volumes and slices
    ct_volumes = torch.stack(ct_volumes, dim=0)  # [B, C, D, H, W]
    ct_slices = torch.stack(ct_slices_list, dim=0)  # [B, N, C, H, W]
    
    # Tokenize text
    text_inputs = tokenizer(
        text_reports,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )
    
    return {
        'ct_volume': ct_volumes,
        'ct_slices': ct_slices,
        'text_inputs': text_inputs,
        'scan_ids': scan_ids,
    }
