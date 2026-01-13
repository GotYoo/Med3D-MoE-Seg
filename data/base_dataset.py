"""
Base Medical Dataset Interface
支持多数据集和分阶段训练的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class BaseMedicalDataset(Dataset, ABC):
    """
    医学数据集基类
    
    所有自定义数据集都应该继承此类并实现必要的方法
    这样可以支持：
    1. 多种数据集（LIDC, LungCT, MSD, 自定义数据集）
    2. 不同训练阶段使用不同数据集
    3. 统一的数据接口
    """
    
    def __init__(self,
                 data_source: str,
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 normalize: bool = True,
                 augmentation: bool = False,
                 **kwargs):
        """
        初始化基类
        
        Args:
            data_source: 数据源路径（可以是目录、JSON文件等）
            image_size: 目标图像大小 (D, H, W)
            normalize: 是否归一化
            augmentation: 是否使用数据增强
            **kwargs: 其他子类特定参数
        """
        super().__init__()
        self.data_source = data_source
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """
        获取一个样本
        
        Returns:
            sample: 必须包含以下键的字典
                - image: CT 图像 [C, D, H, W] 或 [D, H, W]
                - mask: 分割 mask [D, H, W] (可选，取决于任务)
                - text: 文本描述 (可选)
                - prompt: 格式化的 prompt (用于 LLM)
                - metadata: 元数据字典
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, any]:
        """
        返回数据集信息
        
        Returns:
            info: 包含数据集元信息的字典
                - name: 数据集名称
                - num_samples: 样本数量
                - modality: 模态（CT, MRI, etc.）
                - task: 任务类型（segmentation, classification, etc.）
                - classes: 类别列表（如果适用）
        """
        pass
    
    @staticmethod
    @abstractmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Collate function for DataLoader
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            batched: Dictionary with batched data
        """
        pass
    
    def normalize_image(self, image, **kwargs):
        """
        归一化图像（可被子类重写）
        
        默认使用 CT window normalization
        """
        import numpy as np
        
        window_min = kwargs.get('window_min', -1000)
        window_max = kwargs.get('window_max', 400)
        
        image = np.clip(image, window_min, window_max)
        image = (image - window_min) / (window_max - window_min)  # [0, 1]
        image = image * 2 - 1  # [-1, 1]
        
        return image
    
    def apply_augmentation(self, image, mask=None, **kwargs):
        """
        应用数据增强（可被子类重写）
        
        默认实现基础的 3D 增强
        """
        import numpy as np
        
        # 随机翻转
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=2).copy()
        
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()
        
        # 随机旋转
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        
        return (image, mask) if mask is not None else image


class DatasetRegistry:
    """
    数据集注册器
    
    用于注册和管理多个数据集类
    """
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """
        装饰器：注册数据集类
        
        用法:
            @DatasetRegistry.register('lidc')
            class LIDCDataset(BaseMedicalDataset):
                ...
        """
        def decorator(dataset_class):
            cls._registry[name.lower()] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def get(cls, name: str):
        """获取注册的数据集类"""
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(
                f"Dataset '{name}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """列出所有注册的数据集"""
        return list(cls._registry.keys())
    
    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> BaseMedicalDataset:
        """
        创建数据集实例
        
        Args:
            name: 数据集名称
            **kwargs: 传递给数据集构造函数的参数
        
        Returns:
            dataset: 数据集实例
        """
        dataset_class = cls.get(name)
        return dataset_class(**kwargs)


def create_multi_stage_datasets(config: Dict) -> Dict[str, Dict[str, BaseMedicalDataset]]:
    """
    为多阶段训练创建数据集
    
    支持不同训练阶段使用不同的数据集
    
    Args:
        config: 配置字典，格式如下:
            {
                'stage1_alignment': {
                    'dataset': 'lidc',
                    'train_source': 'data/lidc/train.json',
                    'val_source': 'data/lidc/val.json',
                    'data_root': 'processed_lidc_data',
                    ...
                },
                'stage2_rag': {
                    'dataset': 'lungct',
                    'train_source': 'data/lungct/train.json',
                    ...
                },
                'stage3_llm': {
                    'dataset': 'msd',
                    'train_source': 'data/msd/train.json',
                    ...
                },
                'stage4_full': {
                    'datasets': ['lidc', 'lungct', 'msd'],  # 混合多个数据集
                    ...
                }
            }
    
    Returns:
        stage_datasets: 字典，键为阶段名，值为 {'train': dataset, 'val': dataset, 'test': dataset}
    """
    stage_datasets = {}
    
    for stage_name, stage_config in config.items():
        if not stage_name.startswith('stage'):
            continue
        
        # 单数据集
        if 'dataset' in stage_config:
            dataset_name = stage_config['dataset']
            
            train_ds = DatasetRegistry.create_dataset(
                dataset_name,
                data_source=stage_config.get('train_source'),
                **stage_config.get('dataset_params', {})
            )
            
            val_ds = None
            if 'val_source' in stage_config:
                val_ds = DatasetRegistry.create_dataset(
                    dataset_name,
                    data_source=stage_config.get('val_source'),
                    augmentation=False,
                    **stage_config.get('dataset_params', {})
                )
            
            test_ds = None
            if 'test_source' in stage_config:
                test_ds = DatasetRegistry.create_dataset(
                    dataset_name,
                    data_source=stage_config.get('test_source'),
                    augmentation=False,
                    **stage_config.get('dataset_params', {})
                )
            
            stage_datasets[stage_name] = {
                'train': train_ds,
                'val': val_ds,
                'test': test_ds
            }
        
        # 混合多数据集
        elif 'datasets' in stage_config:
            from torch.utils.data import ConcatDataset
            
            train_datasets = []
            val_datasets = []
            test_datasets = []
            
            for ds_name in stage_config['datasets']:
                ds_config = stage_config.get(f'{ds_name}_config', {})
                
                if 'train_source' in ds_config:
                    train_ds = DatasetRegistry.create_dataset(
                        ds_name,
                        data_source=ds_config['train_source'],
                        **ds_config.get('dataset_params', {})
                    )
                    train_datasets.append(train_ds)
                
                if 'val_source' in ds_config:
                    val_ds = DatasetRegistry.create_dataset(
                        ds_name,
                        data_source=ds_config['val_source'],
                        augmentation=False,
                        **ds_config.get('dataset_params', {})
                    )
                    val_datasets.append(val_ds)
                
                if 'test_source' in ds_config:
                    test_ds = DatasetRegistry.create_dataset(
                        ds_name,
                        data_source=ds_config['test_source'],
                        augmentation=False,
                        **ds_config.get('dataset_params', {})
                    )
                    test_datasets.append(test_ds)
            
            stage_datasets[stage_name] = {
                'train': ConcatDataset(train_datasets) if train_datasets else None,
                'val': ConcatDataset(val_datasets) if val_datasets else None,
                'test': ConcatDataset(test_datasets) if test_datasets else None
            }
    
    return stage_datasets
