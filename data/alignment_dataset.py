import torch
import json
import os
import numpy as np
import logging
from typing import Dict, List, Union

# 引入你刚才提供的基类
from .base_dataset import BaseMedicalDataset, DatasetRegistry

logger = logging.getLogger(__name__)

@DatasetRegistry.register('lidc_alignment')
class LIDCAlignmentDataset(BaseMedicalDataset):
    """
    Dataset for Stage 1 Alignment (适配 BaseMedicalDataset 接口).
    Supports both LIDC-IDRI (Weak Supervision) and RadGenome (Strong Supervision).
    """
    def __init__(self, 
                 data_source: str, # 对应之前的 annotation_file/json_path
                 data_root: str = None,
                 image_size: tuple = (128, 128, 128),
                 num_slices: int = 32,
                 require_mask: bool = False,
                 normalize: bool = True,
                 augmentation: bool = False,
                 **kwargs):
        
        super().__init__(
            data_source=data_source,
            image_size=image_size,
            normalize=normalize,
            augmentation=augmentation,
            **kwargs
        )
        
        self.data_root = data_root
        self.num_slices = num_slices
        self.require_mask = require_mask
        
        # 加载 JSON 数据列表
        if data_source and os.path.exists(data_source):
            with open(data_source, 'r') as f:
                self.data_list = json.load(f)
        else:
            logger.warning(f"Data source {data_source} not found or empty.")
            self.data_list = []

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        item = self.data_list[idx]
        
        # --- 1. Load CT Volume ---
        try:
            ct_path = item.get('image_path')
            # 处理相对路径
            if self.data_root and not os.path.isabs(ct_path):
                ct_path = os.path.join(self.data_root, ct_path)
            
            # 加载逻辑 (根据后缀)
            if ct_path.endswith('.npy'):
                ct_volume = np.load(ct_path)
            elif ct_path.endswith('.pt') or ct_path.endswith('.pth'):
                ct_volume = torch.load(ct_path).numpy()
            else:
                # 调试/后备：随机生成
                # logger.debug(f"Generating dummy CT for {ct_path}")
                ct_volume = np.random.randn(*self.image_size).astype(np.float32)

            # 归一化 (调用基类方法)
            if self.normalize:
                ct_volume = self.normalize_image(ct_volume)

            # 转 Tensor [C, D, H, W]
            ct_tensor = torch.from_numpy(ct_volume).float()
            if ct_tensor.ndim == 3:
                ct_tensor = ct_tensor.unsqueeze(0) # Add channel dim

            # 调整切片数 (Interpolate)
            if ct_tensor.shape[1] != self.num_slices:
                ct_tensor = torch.nn.functional.interpolate(
                    ct_tensor.unsqueeze(0), 
                    size=(self.num_slices, self.image_size[1], self.image_size[2]),
                    mode='trilinear', align_corners=False
                ).squeeze(0)
                
        except Exception as e:
            logger.error(f"Error loading CT {item.get('id', idx)}: {e}")
            # 返回随机数据防止 Crash
            ct_tensor = torch.randn(1, self.num_slices, self.image_size[1], self.image_size[2])

        # --- 2. Load Text (Report) ---
        report = item.get('report', "No report available.")

        # --- 3. Load Masks (Optional) ---
        region_masks = None
        if self.require_mask:
            # TODO: 实现 Mask 加载逻辑
            # 这里先返回 None 或 Dummy
            pass

        # 返回符合 BaseMedicalDataset 接口的字典
        # 注意：为了兼容 train_net.py 的逻辑，保留原始 key
        return {
            "id": item.get('id'),
            "ct_volume": ct_tensor, # [C, D, H, W]
            "report": report,       # Raw Text
            "text": report,         # Alias for BaseMedicalDataset standard
            "region_masks": region_masks
        }

    def get_dataset_info(self) -> Dict[str, any]:
        return {
            "name": "LIDCAlignmentDataset",
            "num_samples": len(self),
            "modality": "CT",
            "task": "alignment"
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        简单的 Collate，不包含 Tokenizer (在 train_net.py 中处理)
        """
        if not batch:
            return {}
            
        ids = [x['id'] for x in batch]
        ct_volumes = torch.stack([x['ct_volume'] for x in batch])
        reports = [x['report'] for x in batch]
        
        # Mask 处理 (如果有)
        region_masks = None
        if batch[0].get('region_masks') is not None:
            # stack masks if they exist
            pass

        return {
            "id": ids,
            "ct_volume": ct_volumes,
            "report": reports,
            "text_raw": reports, # 兼容 train_net.py
            "region_masks": region_masks
        }

# 为了兼容旧代码的导入方式，保留这个别名
def alignment_collate_fn(batch, tokenizer=None):
    # 忽略 tokenizer 参数，因为我们在 train_net.py 外部处理了
    return LIDCAlignmentDataset.collate_fn(batch)