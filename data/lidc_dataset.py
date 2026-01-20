"""
LIDC-IDRI Dataset for lung nodule segmentation
"""

import json
import os
from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .base_dataset import BaseMedicalDataset, DatasetRegistry


@DatasetRegistry.register('lidc')
@DatasetRegistry.register('lidcfulldataset')
class LIDCDataset(BaseMedicalDataset):
    """
    LIDC-IDRI 数据集加载器
    
    Args:
        data_json: JSON 文件路径（train.json/val.json/test.json）
        data_root: 数据根目录
        image_size: 图像大小 (D, H, W)
        normalize: 是否归一化
        augmentation: 是否使用数据增强
        prompt_template: 提示模板（可选）
        seg_token: 分割 token（可选）
    """
    def __init__(
        self, 
        data_source: Optional[str] = None,
        data_json: Optional[str] = None,
        data_root: str = '.',
        image_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        augmentation: bool = False,
        prompt_template: Optional[str] = None,
        seg_token: Optional[str] = "[SEG]",
        **kwargs
    ):
        # 兼容 data_json 和 data_source
        data_source = data_source or data_json
        if data_source is None:
            raise ValueError("Either data_source or data_json must be provided")

        super().__init__(
            data_source=data_source,
            image_size=image_size,
            normalize=normalize,
            augmentation=augmentation
        )
        
        self.data_root = data_root
        self.prompt_template = prompt_template or "Segment the lung nodule in this CT scan. {text_report}"
        self.seg_token = seg_token
        
        # 加载 JSON 数据
        with open(data_source, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
        
        print(f"Loaded {len(self._data)} samples from {data_source}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """
        返回一个样本
        
        Returns:
            dict: {
                'image': torch.Tensor [C, D, H, W],
                'mask': torch.Tensor [1, D, H, W],
                'text': str,
                'scan_id': str,
                'metadata': dict
            }
        """
        sample = self._data[idx]
        
        # 加载图像
        image_path = sample['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.data_root, image_path)
        
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata().astype(np.float32)  # [H, W, D]
        image = np.transpose(image, (2, 0, 1))  # [D, H, W]
        
        # 加载掩码
        mask_path = sample['mask_path']
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(self.data_root, mask_path)
        
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata().astype(np.float32)  # [H, W, D]
        mask = np.transpose(mask, (2, 0, 1))  # [D, H, W]
        
        # 归一化图像
        if self.normalize:
            # CT Hounsfield 单位归一化
            image = np.clip(image, -1000, 1000)
            image = (image + 1000) / 2000  # 归一化到 [0, 1]
        
        # 调整大小
        image = torch.from_numpy(image).unsqueeze(0)  # [1, D, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, D, H, W]
        
        if image.shape[1:] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0), 
                size=self.image_size, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
        
        if mask.shape[1:] != self.image_size:
            mask = F.interpolate(
                mask.unsqueeze(0), 
                size=self.image_size, 
                mode='nearest'
            ).squeeze(0)
        
        # 数据增强（如果启用）
        if self.augmentation:
            # 随机翻转
            if torch.rand(1) > 0.5:
                image = torch.flip(image, [1])  # 沿 D 轴翻转
                mask = torch.flip(mask, [1])
            if torch.rand(1) > 0.5:
                image = torch.flip(image, [2])  # 沿 H 轴翻转
                mask = torch.flip(mask, [2])
        
        # 构建文本提示
        text_report = sample.get('text_report', 'CT scan of lung nodule.')
        text = self.prompt_template.format(text_report=text_report)
        
        if self.seg_token:
            text = text + " " + self.seg_token
        
        return {
            'image': image,
            'mask': mask,
            'text': text,
            'scan_id': sample['scan_id'],
            'metadata': {
                'patient_id': sample.get('patient_id'),
                'cluster_stats': sample.get('cluster_stats', [])
            }
        }

    def get_dataset_info(self) -> Dict[str, any]:
        return {
            'name': 'LIDC-IDRI',
            'num_samples': len(self._data),
            'modality': 'CT',
            'task': 'segmentation',
            'classes': ['background', 'nodule']
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Collate 函数，用于批量处理
        """
        images = torch.stack([item['image'] for item in batch], dim=0)
        masks = torch.stack([item['mask'] for item in batch], dim=0)
        texts = [item['text'] for item in batch]
        scan_ids = [item['scan_id'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        return {
            'images': images,
            'masks': masks,
            'texts': texts,
            'scan_ids': scan_ids,
            'metadata': metadata
        }
