# 📁 项目文件结构说明

## 概览

当前项目包含 **2 个 YAML 配置文件** 和 **5 个 Shell 脚本**，它们各有明确的用途。

---

## 📋 YAML 配置文件（.yaml）

YAML 文件用于**配置训练参数**，避免在命令行中输入大量参数。

### 1️⃣ `config/med3d_lisa_full.yaml`
**用途**: 单数据集完整训练配置（默认配置）

**包含内容**:
- ✅ 模型配置（LLM, Vision, MoE, SAM, BioBERT, RAG）
- ✅ 数据配置（LIDC-IDRI 数据集路径）
- ✅ 训练配置（优化器、学习率、损失权重）
- ✅ DeepSpeed 配置（分布式训练）

**使用场景**: 
- 初次训练
- 使用 LIDC 数据集单独训练
- 快速开始实验

**使用方式**:
```bash
python train_net.py --config_file config/med3d_lisa_full.yaml
```

**关键配置**:
```yaml
# 数据集配置
data:
  dataset_type: "LIDCDataset"
  train_json: "data_splits/train.json"
  val_json: "data_splits/val.json"

# RAG 知识库
model:
  rag:
    knowledge_embeddings: "assets/rag_db/knowledge_embeddings.pt"
    knowledge_texts: "assets/rag_db/knowledge_texts.json"
```

---

### 2️⃣ `config/multi_dataset_stages.yaml`
**用途**: 多数据集分阶段训练配置（高级配置）

**包含内容**:
- ✅ Stage 1: Multi-Modal Alignment（使用 LIDC）
- ✅ Stage 2: RAG Integration（混合 LIDC + LungCT）
- ✅ Stage 3: LLM Fine-tuning（使用 MSD）
- ✅ Stage 4: Full Training（混合所有数据集）

**使用场景**:
- 多数据集训练
- 分阶段渐进训练
- 不同阶段使用不同数据集
- 高级实验设置

**使用方式**:
```bash
# 训练阶段 1
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage1_alignment

# 训练阶段 4（混合多数据集）
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage4_full
```

**特点**:
- 每个阶段可配置不同的数据集
- 每个阶段可配置不同的损失权重
- 每个阶段可配置不同的冻结策略

---

## 🔧 Shell 脚本（.sh）

Shell 脚本用于**自动化执行复杂命令**，封装常用操作。

### 1️⃣ `scripts/prepare_data.sh`
**用途**: 数据划分（Patient-wise split）

**功能**:
- ✅ 将原始数据按患者 ID 划分为 train/val/test
- ✅ 防止数据泄漏（同一患者的所有扫描在同一集合）
- ✅ 生成 `data_splits/train.json`, `val.json`, `test.json`

**何时使用**: 
- **第一次准备数据时必须运行**
- 更改数据划分比例时

**使用方式**:
```bash
bash scripts/prepare_data.sh
```

**输出**:
```
data_splits/
├── train.json      # 训练集（70%患者）
├── val.json        # 验证集（15%患者）
├── test.json       # 测试集（15%患者）
└── split_info.json # 划分统计信息
```

---

### 2️⃣ `scripts/build_rag.sh`
**用途**: 构建 RAG 知识库

**功能**:
- ✅ 使用 BioBERT 编码医学知识文本
- ✅ 生成知识向量和文本索引
- ✅ 可选构建 FAISS 索引（加速检索）

**何时使用**:
- **第一次训练前必须运行**
- 添加新的医学知识时
- 更新知识库时

**使用方式**:
```bash
bash scripts/build_rag.sh
```

**输出**:
```
assets/rag_db/
├── knowledge_embeddings.pt  # 知识向量 [N, 768]
├── knowledge_texts.json     # 知识文本和元数据
├── metadata.json            # 统计信息
└── knowledge_index.faiss    # FAISS 索引（可选）
```

---

### 3️⃣ `scripts/train_ds.sh`
**用途**: DeepSpeed 分布式训练脚本

**功能**:
- ✅ 使用 DeepSpeed 进行多 GPU 训练
- ✅ 自动配置分布式参数
- ✅ 支持混合精度训练（BF16/FP16）

**何时使用**:
- 多 GPU 训练时
- 需要 ZeRO 优化时
- 训练大模型（内存不足时）

**使用方式**:
```bash
bash scripts/train_ds.sh
```

**内部执行**:
```bash
deepspeed train_net.py \
    --deepspeed config/deepspeed_config.json \
    --config_file config/med3d_lisa_full.yaml \
    ...
```

---

### 4️⃣ `scripts/eval.sh`
**用途**: 模型评估脚本

**功能**:
- ✅ 在测试集上评估模型
- ✅ 计算分割指标（Dice, IoU, Precision, Recall）
- ✅ 生成可视化结果

**何时使用**:
- 训练完成后评估模型
- 对比不同 checkpoint 性能
- 生成论文图表

**使用方式**:
```bash
bash scripts/eval.sh
```

---

### 5️⃣ `scripts/test_integration.sh`
**用途**: 集成测试脚本

**功能**:
- ✅ 检查文件完整性
- ✅ 验证数据划分是否完成
- ✅ 验证 RAG 知识库是否构建
- ✅ 测试 Python 模块导入
- ✅ 测试配置文件加载
- ✅ 测试 RAG Retriever 功能

**何时使用**:
- **代码更新后必须运行**
- 部署到新环境时
- 排查环境问题时

**使用方式**:
```bash
bash scripts/test_integration.sh
```

**输出示例**:
```
======================================================================
[Test 1] Checking file integrity...
  ✓ config/med3d_lisa_full.yaml
  ✓ data/lidc_dataset.py
  ...
✅ All required files exist

[Test 2] Checking data split outputs...
  ✓ train.json (Lines: 1450, Size: 234K)
  ...
✅ Data splits ready

[Test 3] Checking RAG knowledge base...
  ✓ knowledge_embeddings.pt (Size: 2.1M)
  ...
✅ RAG knowledge base ready
```

---

## 🎯 典型工作流程

### 场景 1: 首次训练（从零开始）

```bash
# 步骤 1: 数据准备
bash scripts/prepare_data.sh

# 步骤 2: 构建 RAG 知识库
bash scripts/build_rag.sh

# 步骤 3: 测试环境
bash scripts/test_integration.sh

# 步骤 4: 开始训练
bash scripts/train_ds.sh
```

---

### 场景 2: 单数据集训练（LIDC）

```bash
# 使用默认配置训练
python train_net.py --config_file config/med3d_lisa_full.yaml
```

---

### 场景 3: 多数据集分阶段训练

```bash
# 阶段 1: 对齐训练（LIDC）
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage1_alignment \
    --output_dir outputs/stage1

# 阶段 2: RAG 训练（LIDC + LungCT）
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage2_rag \
    --output_dir outputs/stage2 \
    --resume_from outputs/stage1/checkpoint-best

# 阶段 4: 全模型训练（所有数据集）
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage4_full \
    --output_dir outputs/stage4
```

---

### 场景 4: 评估模型

```bash
# 评估最佳 checkpoint
bash scripts/eval.sh outputs/med3d_moe_seg_full/checkpoint-best
```

---

## 📊 文件决策树

```
需要训练模型？
├─ 是 → 数据准备好了吗？
│   ├─ 否 → 运行 prepare_data.sh
│   └─ 是 → RAG 知识库构建了吗？
│       ├─ 否 → 运行 build_rag.sh
│       └─ 是 → 使用什么配置？
│           ├─ 单数据集 → med3d_lisa_full.yaml
│           └─ 多数据集/分阶段 → multi_dataset_stages.yaml
│
├─ 否 → 需要评估模型？
│   └─ 是 → 运行 eval.sh
│
└─ 测试环境？
    └─ 是 → 运行 test_integration.sh
```

---

## 🔍 如何选择使用哪个文件

| 需求 | 使用文件 | 说明 |
|------|---------|------|
| 准备训练数据 | `prepare_data.sh` | 必须先运行 |
| 构建知识库 | `build_rag.sh` | 必须先运行 |
| 单数据集训练 | `med3d_lisa_full.yaml` | 简单直接 |
| 多数据集训练 | `multi_dataset_stages.yaml` + 指定 `stage_name` | 更灵活 |
| 分布式训练 | `train_ds.sh` | 多GPU加速 |
| 评估模型 | `eval.sh` | 测试性能 |
| 测试环境 | `test_integration.sh` | 排查问题 |

---

## 💡 最佳实践

### ✅ 推荐做法

1. **首次使用**: 按顺序运行
   ```bash
   bash scripts/prepare_data.sh
   bash scripts/build_rag.sh
   bash scripts/test_integration.sh
   ```

2. **训练时**: 优先使用 Shell 脚本（参数已预配置）
   ```bash
   bash scripts/train_ds.sh
   ```

3. **实验时**: 直接修改 YAML 文件，而非命令行参数
   ```yaml
   # 修改 config/med3d_lisa_full.yaml
   training:
     learning_rate: 1.0e-4  # 调整学习率
     batch_size: 4          # 调整批次大小
   ```

### ⚠️ 避免做法

1. ❌ 跳过数据准备直接训练
2. ❌ 手动输入大量命令行参数（容易出错）
3. ❌ 不运行 `test_integration.sh` 就部署到新环境
4. ❌ 混用不同版本的配置文件

---

## 📝 快速参考

### 必须运行的脚本（首次使用）
```bash
bash scripts/prepare_data.sh      # 1. 数据划分
bash scripts/build_rag.sh         # 2. 知识库构建
bash scripts/test_integration.sh  # 3. 环境测试
```

### 训练相关
```bash
# 单 GPU 训练
python train_net.py --config_file config/med3d_lisa_full.yaml

# 多 GPU 训练
bash scripts/train_ds.sh

# 分阶段训练
python train_net.py \
    --config_file config/multi_dataset_stages.yaml \
    --stage_name stage1_alignment
```

### 评估相关
```bash
bash scripts/eval.sh path/to/checkpoint
```

---

## 🆘 常见问题

**Q: 为什么有两个 YAML 配置文件？**

A: 
- `med3d_lisa_full.yaml`: 简单场景，单数据集训练
- `multi_dataset_stages.yaml`: 复杂场景，多数据集分阶段训练

**Q: 必须运行所有 Shell 脚本吗？**

A: 不是。必须运行的只有：
1. `prepare_data.sh`（首次准备数据）
2. `build_rag.sh`（首次构建知识库）
3. `test_integration.sh`（推荐，验证环境）

**Q: 可以直接修改 Shell 脚本吗？**

A: 可以。Shell 脚本本质上是封装了常用命令，您可以根据需求修改参数。

**Q: 如何添加自己的配置？**

A: 复制现有的 YAML 文件，修改参数即可：
```bash
cp config/med3d_lisa_full.yaml config/my_config.yaml
# 编辑 my_config.yaml
python train_net.py --config_file config/my_config.yaml
```

---

**总结**: 
- **YAML 文件 = 配置参数**（告诉程序"做什么"）
- **Shell 脚本 = 自动化命令**（告诉程序"怎么做"）

两者配合使用，让训练流程更简单、更可复现。
