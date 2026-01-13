# 多数据集与分阶段训练指南

## 📋 概述

Med3D-MoE-Seg 现在支持：
1. **多数据集训练** - 在同一阶段混合使用多个数据集
2. **分阶段训练** - 不同训练阶段使用不同的数据集
3. **灵活配置** - 通过 YAML 配置文件轻松切换数据集
4. **可扩展架构** - 轻松添加新的数据集类型

---

## 🏗️ 架构设计

### 1. 基类设计

所有数据集都继承自 `BaseMedicalDataset`:

```python
from data.base_dataset import BaseMedicalDataset, DatasetRegistry

@DatasetRegistry.register('your_dataset_name')
class YourDataset(BaseMedicalDataset):
    def __init__(self, data_source, **kwargs):
        super().__init__(data_source=data_source, **kwargs)
        # 你的初始化代码
    
    def __getitem__(self, idx):
        # 返回标准格式的数据
        return {
            'image': image_tensor,  # [D, H, W] or [C, D, H, W]
            'mask': mask_tensor,    # [D, H, W]
            'text': text_string,    # 文本描述
            'prompt': prompt_string,# 格式化的 prompt
            'metadata': {...}       # 元数据
        }
    
    def get_dataset_info(self):
        # 返回数据集信息
        return {
            'name': 'YourDataset',
            'num_samples': len(self),
            'modality': 'CT',
            'task': 'segmentation',
            ...
        }
```

### 2. 数据集注册机制

使用 `@DatasetRegistry.register()` 装饰器注册数据集：

```python
# data/your_dataset.py
from data.base_dataset import BaseMedicalDataset, DatasetRegistry

@DatasetRegistry.register('your_dataset')
class YourDataset(BaseMedicalDataset):
    ...

# 使用时
dataset = DatasetRegistry.create_dataset(
    'your_dataset',
    data_source='path/to/data.json',
    **other_params
)
```

---

## 📚 支持的数据集

### 当前已注册的数据集

| 数据集名称 | 注册名 | 任务 | 模态 | 状态 |
|-----------|--------|------|------|------|
| LIDC-IDRI | `lidc` | 肺结节分割 | CT | ✅ 已实现 |
| LungCT | `lungct` | 肺部分割 | CT | 🔄 待实现 |
| MSD Lung | `msd` | 肺部分割 | CT | 🔄 待实现 |
| 自定义 | `custom` | 自定义 | 自定义 | 📝 模板可用 |

### 查看已注册的数据集

```python
from data.base_dataset import DatasetRegistry

# 列出所有已注册的数据集
print(DatasetRegistry.list_datasets())
# 输出: ['lidc', 'lungct', 'msd', ...]
```

---

## 🎯 使用场景

### 场景 1: 单数据集训练（最简单）

适用于：初步实验、单任务训练

**配置文件** (`config/single_dataset.yaml`):
```yaml
simple_mode:
  enabled: true
  dataset: "lidc"
  train_json: "data_splits/lidc/train.json"
  val_json: "data_splits/lidc/val.json"
  test_json: "data_splits/lidc/test.json"
  data_root: "processed_lidc_data"
  image_size: [128, 128, 128]
```

**训练命令**:
```bash
python train_net.py --config_file config/single_dataset.yaml
```

---

### 场景 2: 多数据集混合训练

适用于：增加数据多样性、提升泛化能力

**配置文件** (`config/mixed_datasets.yaml`):
```yaml
training_stages:
  stage_full:
    enabled: true
    datasets: ["lidc", "lungct", "msd"]  # 混合多个数据集
    
    # 为每个数据集配置数据源
    lidc_config:
      train_source: "data_splits/lidc/train.json"
      val_source: "data_splits/lidc/val.json"
      dataset_params:
        data_root: "processed_lidc_data"
        image_size: [128, 128, 128]
    
    lungct_config:
      train_source: "data_splits/lungct/train.json"
      val_source: "data_splits/lungct/val.json"
      dataset_params:
        data_root: "processed_lungct_data"
        image_size: [128, 128, 128]
    
    msd_config:
      train_source: "data_splits/msd/train.json"
      dataset_params:
        data_root: "MSD_Lung_data"
        image_size: [128, 128, 128]
```

**训练命令**:
```bash
python train_net.py \
    --config_file config/mixed_datasets.yaml \
    --stage_name stage_full
```

---

### 场景 3: 分阶段渐进训练

适用于：复杂模型训练、逐步提升能力

**配置文件** (`config/multi_dataset_stages.yaml`):
```yaml
training_stages:
  # 阶段 1: 使用 LIDC 进行对齐训练
  stage1_alignment:
    enabled: true
    dataset: "lidc"
    train_source: "data_splits/lidc/train.json"
    training:
      num_epochs: 20
      loss_weights:
        alignment_loss: 1.0
        seg_loss: 0.0
  
  # 阶段 2: 混合 LIDC 和 LungCT 训练 RAG
  stage2_rag:
    enabled: true
    datasets: ["lidc", "lungct"]
    training:
      num_epochs: 30
      loss_weights:
        rag_retrieval_loss: 1.0
  
  # 阶段 3: 使用 MSD 大数据集微调 LLM
  stage3_llm:
    enabled: true
    dataset: "msd"
    training:
      num_epochs: 40
      loss_weights:
        llm_loss: 1.0
  
  # 阶段 4: 混合所有数据集端到端训练
  stage4_full:
    enabled: true
    datasets: ["lidc", "lungct", "msd"]
    training:
      num_epochs: 50
      loss_weights:
        seg_loss: 1.0
        llm_loss: 0.5
        alignment_loss: 0.1
```

**训练命令**:
```bash
# 阶段 1
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage1_alignment \
    --output_dir outputs/stage1

# 阶段 2（加载阶段 1 的权重）
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage2_rag \
    --output_dir outputs/stage2 \
    --resume_from outputs/stage1/checkpoint-best

# 阶段 3
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage3_llm \
    --output_dir outputs/stage3 \
    --resume_from outputs/stage2/checkpoint-best

# 阶段 4
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage4_full \
    --output_dir outputs/stage4 \
    --resume_from outputs/stage3/checkpoint-best
```

---

## 🔧 添加新数据集

### 步骤 1: 创建数据集类

```python
# data/your_dataset.py
"""
Your Custom Dataset
"""

import torch
from pathlib import Path
from typing import Dict, List, Tuple
from .base_dataset import BaseMedicalDataset, DatasetRegistry


@DatasetRegistry.register('your_dataset')
class YourDataset(BaseMedicalDataset):
    """
    你的自定义数据集描述
    """
    
    def __init__(self,
                 data_source: str,
                 data_root: str = ".",
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 normalize: bool = True,
                 augmentation: bool = False,
                 **kwargs):
        super().__init__(
            data_source=data_source,
            image_size=image_size,
            normalize=normalize,
            augmentation=augmentation
        )
        
        self.data_root = Path(data_root)
        
        # 加载你的数据列表
        self.data_list = self.load_data_list(data_source)
        
        print(f"Loaded {len(self.data_list)} samples from YourDataset")
    
    def load_data_list(self, data_source):
        """加载数据列表"""
        import json
        with open(data_source, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 加载图像和 mask
        image = self.load_image(item['image_path'])
        mask = self.load_mask(item['mask_path'])
        
        # 归一化
        if self.normalize:
            image = self.normalize_image(image)
        
        # 增强
        if self.augmentation:
            image, mask = self.apply_augmentation(image, mask)
        
        # 转换为 tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'text': item.get('text', ''),
            'prompt': self.build_prompt(item),
            'metadata': item.get('metadata', {})
        }
    
    def get_dataset_info(self):
        return {
            'name': 'YourDataset',
            'num_samples': len(self),
            'modality': 'CT',  # 或 'MRI', 'X-ray', etc.
            'task': 'segmentation',
            'classes': ['background', 'foreground'],
            'image_size': self.image_size,
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function"""
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        texts = [item['text'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        return {
            'images': images,
            'masks': masks,
            'texts': texts,
            'prompts': prompts,
            'metadata': metadata,
        }
    
    def load_image(self, path):
        """实现你的图像加载逻辑"""
        pass
    
    def load_mask(self, path):
        """实现你的 mask 加载逻辑"""
        pass
    
    def build_prompt(self, item):
        """构建 prompt"""
        return f"USER: <image>\nSegment the target in this image.\nASSISTANT:"
```

### 步骤 2: 导入到 builder

```python
# data/builder.py
from .your_dataset import YourDataset  # 自动注册
```

### 步骤 3: 在配置文件中使用

```yaml
training_stages:
  stage_custom:
    enabled: true
    dataset: "your_dataset"  # 使用注册名
    train_source: "path/to/your/train.json"
    val_source: "path/to/your/val.json"
    dataset_params:
      data_root: "path/to/data"
      image_size: [128, 128, 128]
```

---

## 💡 最佳实践

### 1. 数据集选择策略

| 训练阶段 | 推荐数据集 | 原因 |
|---------|-----------|------|
| Stage 1 (Alignment) | 小规模标注数据集 | 快速验证对齐效果 |
| Stage 2 (RAG) | 多领域混合数据 | 提升检索多样性 |
| Stage 3 (LLM) | 大规模数据集 | 充分训练语言能力 |
| Stage 4 (Full) | 所有可用数据集 | 最大化泛化能力 |

### 2. 数据格式统一

确保所有数据集的 JSON 文件格式一致：

```json
[
  {
    "id": "sample_001",
    "image_path": "relative/path/to/image.nii.gz",
    "mask_path": "relative/path/to/mask.nii.gz",
    "text": "Optional text description",
    "metadata": {
      "patient_id": "P001",
      "modality": "CT",
      "additional_info": "..."
    }
  },
  ...
]
```

### 3. 数据增强策略

- **训练集**: 启用增强（翻转、旋转、缩放）
- **验证集**: 禁用增强
- **测试集**: 禁用增强

### 4. 批次大小调整

混合多个数据集时，考虑调整批次大小：

```yaml
training:
  per_device_train_batch_size: 2  # 多数据集时减小
  gradient_accumulation_steps: 8  # 增加累积步数保持有效 batch size
```

---

## 🧪 测试与验证

### 测试数据集加载

```python
from data.base_dataset import DatasetRegistry

# 创建数据集
dataset = DatasetRegistry.create_dataset(
    'lidc',
    data_source='data_splits/lidc/train.json',
    data_root='processed_lidc_data',
    image_size=(128, 128, 128)
)

# 测试加载
print(f"Dataset size: {len(dataset)}")
print(f"Dataset info: {dataset.get_dataset_info()}")

# 加载一个样本
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape}")
```

### 测试多数据集混合

```python
from data.base_dataset import create_multi_stage_datasets
import yaml

# 加载配置
with open('config/multi_dataset_stages.yaml') as f:
    config = yaml.safe_load(f)

# 创建阶段数据集
stage_datasets = create_multi_stage_datasets(config['training_stages'])

# 检查各阶段数据集
for stage_name, datasets in stage_datasets.items():
    print(f"\n{stage_name}:")
    print(f"  Train: {len(datasets['train'])} samples")
    print(f"  Val: {len(datasets['val']) if datasets['val'] else 'None'} samples")
```

---

## 📖 完整示例

### 示例：混合 3 个数据集训练

```yaml
# config/my_training.yaml
training_stages:
  stage_full:
    enabled: true
    datasets: ["lidc", "lungct", "msd"]
    
    lidc_config:
      train_source: "data/lidc/train.json"
      val_source: "data/lidc/val.json"
      dataset_params:
        data_root: "datasets/lidc"
        image_size: [128, 128, 128]
        normalize: true
        augmentation: true
    
    lungct_config:
      train_source: "data/lungct/train.json"
      val_source: "data/lungct/val.json"
      dataset_params:
        data_root: "datasets/lungct"
        image_size: [128, 128, 128]
    
    msd_config:
      train_source: "data/msd/train.json"
      dataset_params:
        data_root: "datasets/msd"
        image_size: [128, 128, 128]
    
    training:
      num_epochs: 50
      batch_size: 2
      learning_rate: 2.0e-5
```

```bash
# 训练命令
python train_net.py \
    --config_file config/my_training.yaml \
    --stage_name stage_full \
    --output_dir outputs/mixed_training
```

---

## 🆘 常见问题

### Q1: 如何查看已注册的数据集？

```python
from data.base_dataset import DatasetRegistry
print(DatasetRegistry.list_datasets())
```

### Q2: 不同数据集的图像尺寸不同怎么办？

所有数据集都会 resize 到配置的 `image_size`，确保统一。

### Q3: 如何添加自己的数据集？

参考"添加新数据集"章节，继承 `BaseMedicalDataset` 并注册。

### Q4: 可以在同一阶段使用不同的损失权重吗？

可以，在配置文件的 `training.loss_weights` 中设置。

### Q5: 如何只使用某个数据集的子集？

在数据划分 JSON 文件中筛选样本即可。

---

**参考文档**:
- [data/base_dataset.py](data/base_dataset.py) - 基类定义
- [data/lidc_dataset.py](data/lidc_dataset.py) - LIDC 实现示例
- [config/multi_dataset_stages.yaml](config/multi_dataset_stages.yaml) - 配置示例
