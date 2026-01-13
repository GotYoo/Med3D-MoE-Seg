"""
Data loader builder
支持多种数据集类型和分阶段训练
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from .processors.btb3d_processor import BTB3DDataset
from .base_dataset import DatasetRegistry, create_multi_stage_datasets
from .lidc_dataset import LIDCDataset  # 自动注册到 registry


def collate_fn(batch):
    """
    自定义 collate 函数，用于处理变长的文本 input_ids 并将 3D 图像堆叠为 Tensor
    
    Args:
        batch: 列表，每个元素是一个字典 {'image': tensor, 'input_ids': tensor, 'labels': tensor}
    
    Returns:
        dict: 批量数据
    """
    images = []
    input_ids_list = []
    labels_list = []
    
    for item in batch:
        images.append(item['image'])
        input_ids_list.append(item['input_ids'])
        labels_list.append(item['labels'])
    
    # 堆叠 3D 图像为 Tensor [B, C, D, H, W]
    images = torch.stack(images, dim=0)
    
    # Pad 变长的 input_ids 和 labels
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    return {
        'image': images,
        'input_ids': input_ids,
        'labels': labels
    }


def build_dataloader(dataset_config, tokenizer, batch_size=4, shuffle=True, num_workers=4):
    """
    构建数据加载器（旧版 BTB3D 接口，保持向后兼容）
    
    Args:
        dataset_config: 数据集配置字典，包含 data_root, ann_file, image_size
        tokenizer: 用于文本 tokenization 的 tokenizer
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
    
    Returns:
        DataLoader
    """
    # 实例化 BTB3DDataset
    dataset = BTB3DDataset(
        data_root=dataset_config['data_root'],
        ann_file=dataset_config['ann_file'],
        image_size=dataset_config['image_size'],
        tokenizer=tokenizer
    )
    
    # 使用自定义 collate_fn 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


def build_dataloaders_from_config(config, tokenizer=None, stage_name=None):
    """
    从配置文件构建数据加载器（支持多数据集和分阶段训练）
    
    Args:
        config: 完整配置字典（YAML 格式）
        tokenizer: Tokenizer（可选）
        stage_name: 训练阶段名称（如 'stage1_alignment', 'stage4_full'）
                   如果为 None，使用简化模式或默认数据配置
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 检查是否使用分阶段训练
    if stage_name and 'training_stages' in config:
        print(f"Building datasets for {stage_name}...")
        
        # 为指定阶段创建数据集
        stage_datasets = create_multi_stage_datasets({
            stage_name: config['training_stages'][stage_name]
        })
        
        if stage_name not in stage_datasets:
            raise ValueError(f"Stage '{stage_name}' not found in config")
        
        datasets = stage_datasets[stage_name]
        train_ds = datasets['train']
        val_ds = datasets['val']
        test_ds = datasets['test']
        
        # 获取 collate_fn（如果是 ConcatDataset，使用第一个数据集的 collate_fn）
        if isinstance(train_ds, ConcatDataset):
            collate_fn = train_ds.datasets[0].collate_fn
        else:
            collate_fn = train_ds.collate_fn if hasattr(train_ds, 'collate_fn') else None
        
    # 简化模式（单数据集）
    elif 'simple_mode' in config and config['simple_mode'].get('enabled', False):
        print("Using simple mode (single dataset)...")
        
        simple_config = config['simple_mode']
        dataset_name = simple_config['dataset']
        
        train_ds = DatasetRegistry.create_dataset(
            dataset_name,
            data_source=simple_config['train_json'],
            data_root=simple_config.get('data_root', '.'),
            image_size=tuple(simple_config.get('image_size', [128, 128, 128])),
            normalize=True,
            augmentation=True
        )
        
        val_ds = DatasetRegistry.create_dataset(
            dataset_name,
            data_source=simple_config['val_json'],
            data_root=simple_config.get('data_root', '.'),
            image_size=tuple(simple_config.get('image_size', [128, 128, 128])),
            normalize=True,
            augmentation=False
        ) if 'val_json' in simple_config else None
        
        test_ds = DatasetRegistry.create_dataset(
            dataset_name,
            data_source=simple_config['test_json'],
            data_root=simple_config.get('data_root', '.'),
            image_size=tuple(simple_config.get('image_size', [128, 128, 128])),
            normalize=True,
            augmentation=False
        ) if 'test_json' in simple_config else None
        
        collate_fn = train_ds.collate_fn if hasattr(train_ds, 'collate_fn') else None
    
    # 旧版兼容模式
    elif 'data' in config:
        print("Using legacy data config...")
        data_config = config['data']
        dataset_type = data_config.get('dataset_type', 'LIDCDataset')
        
        if dataset_type in ['LIDCDataset', 'lidc']:
            # 向后兼容旧的 LIDC 配置
            train_ds = LIDCDataset(
                data_json=data_config['train_json'],
                data_root=data_config['data_root'],
                image_size=tuple(data_config['image_size']),
                normalize=data_config.get('normalize', True),
                augmentation=data_config.get('augmentation', {}).get('enabled', True),
                prompt_template=data_config.get('prompt_template'),
                seg_token=data_config.get('seg_token'),
            )
            
            val_ds = LIDCDataset(
                data_json=data_config['val_json'],
                data_root=data_config['data_root'],
                image_size=tuple(data_config['image_size']),
                normalize=data_config.get('normalize', True),
                augmentation=False,
                prompt_template=data_config.get('prompt_template'),
                seg_token=data_config.get('seg_token'),
            )
            
            test_ds = LIDCDataset(
                data_json=data_config['test_json'],
                data_root=data_config['data_root'],
                image_size=tuple(data_config['image_size']),
                normalize=data_config.get('normalize', True),
                augmentation=False,
                prompt_template=data_config.get('prompt_template'),
                seg_token=data_config.get('seg_token'),
            )
            
            collate_fn = LIDCDataset.collate_fn
            
        elif dataset_type == 'BTB3D':
            # 旧版 BTB3D 数据集
            if tokenizer is None:
                raise ValueError("Tokenizer is required for BTB3D dataset")
            
            train_loader = build_dataloader(
                dataset_config=data_config,
                tokenizer=tokenizer,
                batch_size=config.get('training', {}).get('per_device_train_batch_size', 4),
                shuffle=True,
                num_workers=data_config.get('num_workers', 4)
            )
            return train_loader, None, None
        
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    else:
        raise ValueError("No valid data configuration found in config file")
    
    # 创建 DataLoader
    training_config = config.get('training', {})
    dataloader_config = config.get('dataloader', {})
    
    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.get('per_device_train_batch_size', 2),
        shuffle=True,
        num_workers=dataloader_config.get('num_workers', 8),
        collate_fn=collate_fn,
        pin_memory=dataloader_config.get('pin_memory', True),
        prefetch_factor=dataloader_config.get('prefetch_factor', 2),
    ) if train_ds is not None else None
    
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.get('per_device_eval_batch_size', 2),
        shuffle=False,
        num_workers=dataloader_config.get('num_workers', 8),
        collate_fn=collate_fn,
        pin_memory=dataloader_config.get('pin_memory', True),
    ) if val_ds is not None else None
    
    test_loader = DataLoader(
        test_ds,
        batch_size=training_config.get('per_device_eval_batch_size', 2),
        shuffle=False,
        num_workers=dataloader_config.get('num_workers', 8),
        collate_fn=collate_fn,
        pin_memory=dataloader_config.get('pin_memory', True),
    ) if test_ds is not None else None
    
    return train_loader, val_loader, test_loader

