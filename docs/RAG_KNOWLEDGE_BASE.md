# RAG 知识库构建指南

## 📋 概述

RAG (Retrieval-Augmented Generation) 知识库用于在推理时检索相关医学知识，增强模型的领域专业性。`build_rag_index.py` 脚本将医学文本编码为向量并构建检索索引。

---

## 🚀 快速开始

### 方法 1: 使用 Bash 脚本（推荐）

```bash
# 1. 准备医学知识文本文件
# 格式：每行一条知识

# 2. 运行构建脚本
bash scripts/build_rag.sh
```

### 方法 2: 直接运行 Python 脚本

```bash
python scripts/build_rag_index.py \
    --input_file data/medical_knowledge.txt \
    --output_dir assets/rag_db \
    --biobert_model dmis-lab/biobert-v1.1 \
    --batch_size 32 \
    --use_faiss
```

---

## 📁 输入格式

### 纯文本格式（推荐）

每行一条医学知识，支持注释：

```
# Lung Nodule Characteristics
Pulmonary nodules are small round or oval-shaped growths in the lungs...
Ground-glass opacity (GGO) refers to hazy increased lung attenuation...
Solid pulmonary nodules appear as homogeneous soft-tissue attenuation...

# CT Imaging Guidelines
The Fleischner Society provides guidelines for management...
Calcification patterns in pulmonary nodules can help determine benignity...
```

### JSON Lines 格式（可选）

每行一个 JSON 对象，包含更多元数据：

```json
{"id": "k001", "text": "Pulmonary nodules are...", "source": "textbook", "category": "nodules"}
{"id": "k002", "text": "Ground-glass opacity...", "source": "guidelines", "category": "imaging"}
{"id": "k003", "text": "Solid pulmonary nodules...", "source": "radiology", "category": "nodules"}
```

支持的字段：
- `text` (必需): 知识文本内容
- `id` (可选): 唯一标识符
- `source` (可选): 知识来源（如 "textbook", "guidelines"）
- `category` (可选): 知识类别（如 "nodules", "imaging"）

---

## 📊 输出文件

构建完成后生成以下文件：

```
assets/rag_db/
├── knowledge_embeddings.pt      # BioBERT 编码的向量 [N, 768]
├── knowledge_texts.json         # 原始知识文本和元数据
├── metadata.json                # 知识库元信息
└── knowledge_index.faiss        # FAISS 索引（可选，需要 faiss 库）
```

### 文件说明

#### 1. knowledge_embeddings.pt
```python
# PyTorch tensor: [num_entries, 768]
embeddings = torch.load('assets/rag_db/knowledge_embeddings.pt')
print(embeddings.shape)  # torch.Size([1000, 768])
```

#### 2. knowledge_texts.json
```json
[
  {
    "id": "k0",
    "text": "Pulmonary nodules are small round...",
    "source": "text_file",
    "category": "general"
  },
  ...
]
```

#### 3. metadata.json
```json
{
  "num_entries": 1000,
  "embedding_dim": 768,
  "biobert_model": "dmis-lab/biobert-v1.1",
  "max_length": 512,
  "created_date": "..."
}
```

---

## ⚙️ 参数配置

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_file` | str | **必需** | 医学知识文本文件路径 |
| `--output_dir` | str | `assets/rag_db` | 输出目录 |
| `--biobert_model` | str | `dmis-lab/biobert-v1.1` | BioBERT 模型 |
| `--batch_size` | int | 32 | 批处理大小 |
| `--max_length` | int | 512 | 最大序列长度 |
| `--use_faiss` | flag | False | 是否构建 FAISS 索引 |
| `--device` | str | auto | 设备 (cuda/cpu) |

### 推荐配置

**小规模知识库** (<1000条):
```bash
--batch_size 32 --max_length 512
```

**中规模知识库** (1000-10000条):
```bash
--batch_size 64 --max_length 256 --use_faiss
```

**大规模知识库** (>10000条):
```bash
--batch_size 128 --max_length 256 --use_faiss
# 建议使用 GPU: --device cuda
```

---

## 🔍 FAISS 索引（可选）

### 安装 FAISS

```bash
# CPU 版本
pip install faiss-cpu

# GPU 版本（需要 CUDA）
pip install faiss-gpu
```

### 使用 FAISS

启用 FAISS 后，检索速度显著提升：

```bash
python scripts/build_rag_index.py \
    --input_file data/medical_knowledge.txt \
    --output_dir assets/rag_db \
    --use_faiss  # 启用 FAISS
```

FAISS 索引使用 **IndexFlatIP** (内积) 进行余弦相似度搜索：
- 自动归一化向量
- O(1) 索引构建
- O(N) 检索复杂度（精确搜索）

---

## 📈 示例运行

### 示例输出

```
======================================================================
RAG Knowledge Base Construction
======================================================================
Input file: data/medical_knowledge_sample.txt
Output directory: assets/rag_db
BioBERT model: dmis-lab/biobert-v1.1
Batch size: 32
Max length: 512
Device: cuda
Use FAISS: True

[Step 1] Loading knowledge texts...
Loaded 27 knowledge entries

[Step 2] Initializing BioBERT encoder...
✓ BioBERT encoder loaded
  Hidden size: 768

[Step 3] Encoding knowledge texts...
Encoding: 100%|██████████| 1/1 [00:01<00:00,  1.61s/it]

✓ Encoded 27 knowledge entries
  Embedding shape: torch.Size([27, 768])

[Step 4] Saving knowledge base...
✓ Saved embeddings to: assets/rag_db/knowledge_embeddings.pt
✓ Saved texts to: assets/rag_db/knowledge_texts.json
✓ Saved metadata to: assets/rag_db/metadata.json

[Step 5] Building FAISS index...
✓ Built and saved FAISS index to: assets/rag_db/knowledge_index.faiss
  Index size: 27 vectors
  Index type: IndexFlatIP (Cosine Similarity)

======================================================================
Knowledge Base Statistics
======================================================================
Total entries: 27
Embedding dimension: 768

Categories:
  - general: 27 (100.0%)

Text lengths:
  - Mean: 141.3 chars
  - Median: 142.0 chars

✅ Knowledge base construction completed!
======================================================================
```

---

## 🔌 集成到训练

### 1. 更新模型配置

编辑配置文件（如 `config/med3d_lisa_full.yaml`）：

```yaml
rag:
  enabled: true
  knowledge_embeddings: assets/rag_db/knowledge_embeddings.pt
  knowledge_texts: assets/rag_db/knowledge_texts.json
  top_k: 3
  knowledge_dim: 768
  llm_hidden_size: 4096
```

### 2. 加载知识库

在模型初始化时加载：

```python
from model.rag.retriever import MedicalKnowledgeRetriever

# 创建 retriever
retriever = MedicalKnowledgeRetriever(
    knowledge_dim=768,
    llm_hidden_size=4096,
    top_k=3
)

# 加载预构建的知识库
retriever.load_knowledge_base('assets/rag_db/knowledge_embeddings.pt')

# 加载文本
import json
with open('assets/rag_db/knowledge_texts.json') as f:
    knowledge_texts = json.load(f)
```

### 3. 训练时使用

```python
# 在训练循环中
image_features = ct_clip(images)
rag_outputs = retriever(image_features, return_details=True)

# 获取检索到的知识
retrieved_texts = [
    knowledge_texts[idx]['text'] 
    for idx in rag_outputs['indices'][0].tolist()
]

print(f"Retrieved knowledge: {retrieved_texts}")
```

---

## 📚 准备医学知识

### 知识来源

1. **医学教科书**
   - 提取重要的概念、定义、诊断标准

2. **临床指南**
   - Fleischner Society guidelines
   - ACR Appropriateness Criteria
   - NCCN Guidelines

3. **放射学手册**
   - 影像特征描述
   - 鉴别诊断要点

4. **医学文献**
   - 综述文章摘要
   - 研究发现总结

### 知识组织建议

**按疾病分类**:
```
# Lung Nodules
...nodule characteristics...
...nodule management...

# Pneumonia
...pneumonia patterns...
...pneumonia diagnosis...
```

**按影像特征分类**:
```
# Ground-Glass Opacity
...GGO definition...
...GGO differential...

# Consolidation
...consolidation patterns...
```

**按解剖结构分类**:
```
# Lung Anatomy
...lobar anatomy...
...bronchial tree...

# Mediastinum
...mediastinal compartments...
```

---

## 🧪 验证知识库

### 检查文件完整性

```bash
python -c "
import torch
import json

# 加载并检查
embeddings = torch.load('assets/rag_db/knowledge_embeddings.pt')
with open('assets/rag_db/knowledge_texts.json') as f:
    texts = json.load(f)

print(f'Embeddings: {embeddings.shape}')
print(f'Texts: {len(texts)}')
assert embeddings.shape[0] == len(texts), 'Size mismatch!'
print('✅ Validation passed')
"
```

### 测试检索功能

```python
import torch
from model.rag.retriever import MedicalKnowledgeRetriever

# 创建 retriever 并加载知识库
retriever = MedicalKnowledgeRetriever(knowledge_dim=768, llm_hidden_size=4096, top_k=3)
retriever.load_knowledge_base('assets/rag_db/knowledge_embeddings.pt')

# 测试检索
query = torch.randn(1, 768)  # 模拟查询向量
results = retriever(query, return_details=True)

print(f"Top-3 indices: {results['indices'][0].tolist()}")
print(f"Top-3 scores: {results['relevance_scores'][0].tolist()}")
```

---

## 🔧 高级用法

### 1. 增量更新知识库

```python
import torch

# 加载现有知识库
old_embeddings = torch.load('assets/rag_db/knowledge_embeddings.pt')

# 编码新知识
new_embeddings = encode_new_knowledge(new_texts)

# 合并
updated_embeddings = torch.cat([old_embeddings, new_embeddings], dim=0)

# 保存
torch.save(updated_embeddings, 'assets/rag_db/knowledge_embeddings.pt')
```

### 2. 过滤低质量知识

```python
# 移除过短或重复的知识
filtered_texts = [
    item for item in knowledge_list 
    if len(item['text']) > 50 and is_unique(item['text'])
]
```

### 3. 多语言支持

使用多语言 BioBERT（如果可用）：

```bash
python scripts/build_rag_index.py \
    --input_file data/medical_knowledge_multilang.txt \
    --biobert_model bert-base-multilingual-cased
```

---

## 🐛 故障排除

### Q1: 内存不足
**解决**: 减小 batch_size 或使用 CPU

```bash
python scripts/build_rag_index.py \
    --input_file data/large_knowledge.txt \
    --batch_size 16 \
    --device cpu
```

### Q2: FAISS 安装失败
**解决**: 使用不带 FAISS 的版本

```bash
bash scripts/build_rag.sh
# 编辑 build_rag.sh，设置 USE_FAISS=false
```

### Q3: BioBERT 下载慢
**解决**: 手动下载模型到本地

```bash
# 下载到本地
git clone https://huggingface.co/dmis-lab/biobert-v1.1

# 使用本地路径
python scripts/build_rag_index.py \
    --biobert_model ./biobert-v1.1
```

### Q4: 编码速度慢
**解决**: 使用 GPU 和更大的 batch_size

```bash
python scripts/build_rag_index.py \
    --batch_size 64 \
    --device cuda
```

---

## 📊 性能基准

| 知识库规模 | 编码时间 | 内存占用 | 检索时间 |
|-----------|----------|----------|----------|
| 100 条 | ~5秒 | ~500MB | <1ms |
| 1,000 条 | ~30秒 | ~2GB | ~5ms |
| 10,000 条 | ~5分钟 | ~8GB | ~50ms |
| 100,000 条 | ~50分钟 | ~60GB | ~500ms |

*测试环境: V100 GPU, batch_size=32*

---

## 📝 下一步

知识库构建完成后：

1. ✅ 验证生成的文件
2. ✅ 更新训练配置
3. ✅ 测试检索功能
4. ✅ 开始训练！

---

**脚本位置**: `scripts/build_rag_index.py`  
**示例脚本**: `scripts/build_rag.sh`  
**示例数据**: `data/medical_knowledge_sample.txt`
