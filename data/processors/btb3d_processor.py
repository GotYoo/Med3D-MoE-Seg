"""
BTB3D Dataset Processor
用于处理 3D 医学图像和文本指令
"""

import json
import os
import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensityRange, 
    Resize, ToTensor, ScaleIntensity
)
import nibabel as nib


class BTB3DDataset(Dataset):
    """
    BTB3D Dataset 类，用于加载 3D 医学图像和对应的文本指令
    """
    
    def __init__(self, data_root, ann_file, image_size=(128, 128, 128), tokenizer=None):
        """
        初始化数据集
        
        Args:
            data_root: NIfTI 文件目录
            ann_file: 标注文件路径 (JSON 格式)
            image_size: 目标图像尺寸 (D, H, W)
            tokenizer: 文本 tokenizer
        """
        self.data_root = data_root
        self.image_size = image_size
        self.tokenizer = tokenizer
        
        # 加载标注文件
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 定义 MONAI 变换
        self.transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            # 医疗图像关键：HU值截断 (根据不同部位调整，如腹部 -175 到 250)
            ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            # 调整到 BTB3D/SAM-Med3D 需要的输入尺寸
            Resize(image_size),
            ToTensor()
        ])
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 索引
        
        Returns:
            dict: 包含 'image', 'input_ids', 'labels' 的字典
        """
        ann = self.annotations[idx]
        
        # 读取 3D 图像 (.nii.gz)
        image_path = os.path.join(self.data_root, ann['image_path'])
        image = self.transform(image_path)  # 返回 [C, D, H, W]
        
        # 读取文本指令
        text_instruction = ann['text']
        
        # Tokenize 文本
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                text_instruction,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=512
            )
            input_ids = tokenized['input_ids'].squeeze(0)  # [seq_len]
            # 对于语言模型训练，labels 通常就是 input_ids
            labels = input_ids.clone()
        else:
            # 如果没有 tokenizer，返回原始文本
            input_ids = torch.tensor([0])
            labels = torch.tensor([0])
        
        return {
            'image': image,  # [C, D, H, W]
            'input_ids': input_ids,  # [seq_len]
            'labels': labels  # [seq_len]
        }