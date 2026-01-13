# 代码更新说明

## 📋 更新概览

根据数据划分和 RAG 知识库构建，已完成以下代码更新：

---

## ✅ 1. MedicalKnowledgeRetriever 更新

**文件**: [model/rag/retriever.py](model/rag/retriever.py)

**更新内容**:
- ✅ 添加 `knowledge_texts_path` 参数支持加载知识文本
- ✅ 在 `__init__` 中加载 `knowledge_texts.json`
- ✅ 在 `forward` 中返回检索到的知识文本（`retrieved_texts`）
- ✅ 支持同时加载 embeddings 和文本元数据

**新增功能**:
```python
# 创建时指定知识库路径
retriever = MedicalKnowledgeRetriever(
    knowledge_embed_path='assets/rag_db/knowledge_embeddings.pt',
    knowledge_texts_path='assets/rag_db/knowledge_texts.json',
    knowledge_dim=768,
    llm_hidden_size=4096,
    top_k=3
)

# 检索时返回文本
outputs = retriever(query_embed, return_details=True)
retrieved_texts = outputs['retrieved_texts']  # List[List[Dict]]
```

---

## ✅ 2. 配置文件创建

**文件**: [config/med3d_lisa_full.yaml](config/med3d_lisa_full.yaml)

**包含配置**:
- ✅ **模型配置**: LLM, Vision, MoE, SAM, BioBERT, Alignment, RAG, Self-Correction
- ✅ **数据配置**: LIDC-IDRI 数据集路径、数据划分 JSON、数据增强
- ✅ **训练配置**: 优化器、学习率、损失权重、评估策略
- ✅ **DeepSpeed 配置**: ZeRO-2, 混合精度 (BF16)
- ✅ **推理配置**: 生成参数、可视化、后处理

**关键配置项**:
```yaml
# RAG 配置
model:
  rag:
    enabled: true
    knowledge_embeddings: "assets/rag_db/knowledge_embeddings.pt"
    knowledge_texts: "assets/rag_db/knowledge_texts.json"
    knowledge_dim: 768
    top_k: 3

# 数据配置
data:
  dataset_type: "LIDCDataset"
  data_root: "processed_lidc_data"
  train_json: "data_splits/train.json"
  val_json: "data_splits/val.json"
  test_json: "data_splits/test.json"
```

---

## ✅ 3. LIDC Dataset 创建

**文件**: [data/lidc_dataset.py](data/lidc_dataset.py)

**功能**:
- ✅ 加载 `prepare_data_split.py` 生成的 JSON 数据划分
- ✅ 患者级别数据组织（防止数据泄漏）
- ✅ NIfTI 格式 CT 图像和 mask 加载
- ✅ CT 窗口归一化 ([-1000, 400] HU → [-1, 1])
- ✅ 3D 数据增强（翻转、旋转）
- ✅ 动态 prompt 生成
- ✅ 自定义 collate_fn

**使用方式**:
```python
from data.lidc_dataset import create_lidc_dataloaders

# 加载配置
with open('config/med3d_lisa_full.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建数据集
train_ds, val_ds, test_ds = create_lidc_dataloaders(config)

# 数据样本格式
sample = train_ds[0]
# Keys: image, mask, text, prompt, patient_id, nodule_id, metadata
# image: [D, H, W] torch.Tensor
# mask:  [D, H, W] torch.Tensor
```

**支持的数据格式**:
```json
{
  "patient_id": "LIDC-IDRI-0001",
  "nodule_id": "nodule_001",
  "image_path": "LIDC-IDRI-0001/image_001.nii.gz",
  "mask_path": "LIDC-IDRI-0001/mask_001.nii.gz",
  "report_path": "LIDC-IDRI-0001/report_001.txt",
  "metadata": {...}
}
```

---

## ✅ 4. Data Builder 更新

**文件**: [data/builder.py](data/builder.py)

**更新内容**:
- ✅ 添加 `build_dataloaders_from_config()` 函数
- ✅ 支持从 YAML 配置文件构建 DataLoader
- ✅ 自动选择数据集类型（`LIDCDataset` / `BTB3D`）
- ✅ 保持旧版 `build_dataloader()` 向后兼容

**新接口**:
```python
from data.builder import build_dataloaders_from_config

train_loader, val_loader, test_loader = build_dataloaders_from_config(
    config, 
    tokenizer=None  # LIDC 不需要 tokenizer
)
```

---

## ✅ 5. Train Net 更新

**文件**: [train_net.py](train_net.py)

**ModelArguments 新增**:
```python
# RAG 配置
rag_enabled: bool = True
rag_knowledge_embeddings: Optional[str] = None
rag_knowledge_texts: Optional[str] = None
rag_top_k: int = 3
```

**DataArguments 新增**:
```python
# 数据集类型
dataset_type: str = "LIDCDataset"

# 数据划分（患者级别）
train_json: Optional[str] = None
val_json: Optional[str] = None
test_json: Optional[str] = None

# YAML 配置文件
config_file: Optional[str] = None
```

**main() 函数更新**:
- ✅ 支持从 YAML 配置文件加载参数
- ✅ 自动选择数据集类型（LIDC / BTB3D）
- ✅ 打印 RAG 和数据集信息
- ✅ 创建训练/验证/测试集

**使用方式**:
```bash
# 使用 YAML 配置
python train_net.py \
    --config_file config/med3d_lisa_full.yaml \
    --output_dir outputs/med3d_moe_seg_full

# 或使用命令行参数
python train_net.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_json data_splits/train.json \
    --val_json data_splits/val.json \
    --rag_knowledge_embeddings assets/rag_db/knowledge_embeddings.pt \
    --rag_knowledge_texts assets/rag_db/knowledge_texts.json
```

---

## 🔄 数据流程

### 完整训练流程

```
1. 数据准备
   ├─ prepare_data_split.py → 患者级别划分
   └─ 生成: train.json, val.json, test.json

2. 知识库构建
   ├─ build_rag_index.py → BioBERT 编码
   └─ 生成: knowledge_embeddings.pt, knowledge_texts.json

3. 训练
   ├─ train_net.py 加载配置
   ├─ LIDCDataset 加载数据
   ├─ MedicalKnowledgeRetriever 加载知识库
   └─ Med3DLISA_Full 模型训练
```

### 数据加载流程

```
train.json
    ↓
LIDCDataset.__getitem__()
    ├─ 加载 CT 图像 (NIfTI)
    ├─ 加载分割 mask
    ├─ 加载报告文本
    ├─ 归一化 & 增强
    └─ 构建 prompt
    ↓
DataLoader (batch)
    ├─ collate_fn 打包
    └─ 返回批次数据
    ↓
Med3DLISA_Full.forward()
    ├─ CT-CLIP 编码图像
    ├─ BioBERT 对齐
    ├─ RAG 检索知识
    ├─ MoE LLM 生成
    ├─ SAM-Med3D 分割
    └─ Self-Correction 优化
```

---

## 📦 文件结构

```
Med3D-MoE-Seg/
├── config/
│   └── med3d_lisa_full.yaml           # ✅ 新增：完整配置文件
│
├── data/
│   ├── builder.py                      # ✅ 更新：支持新数据集
│   ├── lidc_dataset.py                 # ✅ 新增：LIDC 数据集类
│   └── medical_knowledge_sample.txt    # ✅ 已有：示例知识
│
├── data_splits/                        # 由 prepare_data_split.py 生成
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── split_info.json
│
├── assets/rag_db/                      # 由 build_rag_index.py 生成
│   ├── knowledge_embeddings.pt         # [27, 768]
│   ├── knowledge_texts.json            # 27 条知识
│   └── metadata.json
│
├── model/
│   ├── rag/
│   │   └── retriever.py                # ✅ 更新：加载知识文本
│   └── meta_arch/
│       └── med3d_lisa.py               # ✅ 已有：4-Stage 架构
│
├── scripts/
│   ├── prepare_data_split.py           # ✅ 已有：数据划分
│   ├── prepare_data.sh                 # ✅ 已有：数据划分脚本
│   ├── build_rag_index.py              # ✅ 已有：RAG 构建
│   └── build_rag.sh                    # ✅ 已有：RAG 构建脚本
│
└── train_net.py                        # ✅ 更新：支持新配置和数据集
```

---

## 🚀 快速开始

### Step 1: 数据准备

```bash
# 运行数据划分
bash scripts/prepare_data.sh

# 验证输出
ls data_splits/
# 应看到: train.json, val.json, test.json, split_info.json
```

### Step 2: 构建 RAG 知识库

```bash
# 运行知识库构建
bash scripts/build_rag.sh

# 验证输出
ls assets/rag_db/
# 应看到: knowledge_embeddings.pt, knowledge_texts.json, metadata.json
```

### Step 3: 训练模型

```bash
# 使用配置文件训练
python train_net.py \
    --config_file config/med3d_lisa_full.yaml \
    --output_dir outputs/med3d_moe_seg_full \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 50 \
    --learning_rate 2e-4 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10
```

---

## 🔍 测试代码

### 测试 LIDC Dataset

```bash
cd /home/wuhanqing/Med3D-MoE-Seg
python data/lidc_dataset.py
```

**预期输出**:
```
Loading dataset from: data_splits/train.json
Loaded XXX samples
  - Unique patients: XX

Testing data loading...
Sample keys: dict_keys(['image', 'mask', 'text', 'prompt', 'patient_id', 'nodule_id', 'metadata'])
Image shape: torch.Size([128, 128, 128])
Mask shape: torch.Size([128, 128, 128])
Prompt: USER: <image>...
Patient ID: LIDC-IDRI-XXXX
```

### 测试 RAG Retriever

```python
from model.rag.retriever import MedicalKnowledgeRetriever
import torch

# 创建 retriever（加载知识库）
retriever = MedicalKnowledgeRetriever(
    knowledge_embed_path='assets/rag_db/knowledge_embeddings.pt',
    knowledge_texts_path='assets/rag_db/knowledge_texts.json',
    knowledge_dim=768,
    llm_hidden_size=4096,
    top_k=3
)

# 测试检索
query = torch.randn(1, 768)
outputs = retriever(query, return_details=True)

print(f"Context embed shape: {outputs['context_embed'].shape}")  # [1, 4096]
print(f"Top-3 indices: {outputs['retrieved_indices'][0].tolist()}")
print(f"Top-3 scores: {outputs['relevance_scores'][0].tolist()}")
print(f"Retrieved texts: {outputs['retrieved_texts'][0]}")
```

---

## 📝 配置文件说明

### 关键配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model.rag.enabled` | `true` | 是否启用 RAG |
| `model.rag.knowledge_embeddings` | `assets/rag_db/...` | 知识库向量路径 |
| `model.rag.knowledge_texts` | `assets/rag_db/...` | 知识库文本路径 |
| `model.rag.top_k` | `3` | 检索 Top-K |
| `data.dataset_type` | `"LIDCDataset"` | 数据集类型 |
| `data.train_json` | `"data_splits/train.json"` | 训练集路径 |
| `data.image_size` | `[128, 128, 128]` | 图像大小 |
| `training.per_device_train_batch_size` | `2` | 批次大小 |
| `training.gradient_accumulation_steps` | `8` | 梯度累积 |

---

## 🔧 常见问题

### Q1: 如何使用旧版 BTB3D 数据集？

**答**: 在配置文件中设置:
```yaml
data:
  dataset_type: "BTB3D"
  ann_file: "path/to/annotations.json"
```

或命令行:
```bash
python train_net.py \
    --dataset_type BTB3D \
    --ann_file path/to/annotations.json
```

### Q2: 如何禁用 RAG？

**答**: 在配置文件中设置:
```yaml
model:
  rag:
    enabled: false
```

或命令行:
```bash
python train_net.py \
    --rag_enabled false
```

### Q3: 数据划分后如何重新加载？

**答**: 直接运行训练，会自动加载 `data_splits/` 目录下的 JSON 文件。

### Q4: 如何添加新的医学知识？

**答**: 
1. 编辑 `data/medical_knowledge.txt`（每行一条知识）
2. 重新运行 `bash scripts/build_rag.sh`
3. 训练时会自动加载新的知识库

---

## 📈 下一步计划

- [ ] 实现完整的训练循环
- [ ] 添加评估指标（Dice, IoU, Precision, Recall）
- [ ] 实现可视化工具
- [ ] 添加推理脚本
- [ ] 支持多 GPU 训练（DeepSpeed）
- [ ] 添加 Checkpoint 恢复
- [ ] 实现 WandB 日志记录

---

**更新日期**: 2026-01-07  
**更新内容**: 完成数据划分和 RAG 知识库集成
