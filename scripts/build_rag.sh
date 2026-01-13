#!/bin/bash
# RAG 知识库构建脚本
# 将医学知识文本编码为向量并保存

# ============================================================
# 配置参数
# ============================================================

# 输入文件（医学知识文本，每行一条）
INPUT_FILE="data/medical_knowledge_sample.txt"

# 输出目录
OUTPUT_DIR="assets/rag_db"

# BioBERT 模型
BIOBERT_MODEL="dmis-lab/biobert-v1.1"

# 批处理大小
BATCH_SIZE=32

# 最大序列长度
MAX_LENGTH=512

# 是否使用 FAISS（需要安装 faiss-cpu 或 faiss-gpu）
USE_FAISS=true

# ============================================================
# 运行知识库构建
# ============================================================

echo "==========================================="
echo "RAG Knowledge Base Construction"
echo "==========================================="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "BioBERT model: $BIOBERT_MODEL"
echo ""

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    echo ""
    echo "Please create a medical knowledge text file with format:"
    echo "  - One knowledge entry per line"
    echo "  - Plain text or JSON format"
    echo ""
    echo "Example:"
    echo "  Pulmonary nodules are small round growths in the lungs..."
    echo "  Ground-glass opacity refers to hazy increased lung attenuation..."
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python scripts/build_rag_index.py \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --biobert_model $BIOBERT_MODEL \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH"

# 添加 FAISS 选项
if [ "$USE_FAISS" = true ]; then
    CMD="$CMD --use_faiss"
fi

# 运行脚本
echo "Running: $CMD"
echo ""
$CMD

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✅ Knowledge base construction completed!"
    echo "==========================================="
    echo "Generated files:"
    echo "  - $OUTPUT_DIR/knowledge_embeddings.pt"
    echo "  - $OUTPUT_DIR/knowledge_texts.json"
    echo "  - $OUTPUT_DIR/metadata.json"
    if [ "$USE_FAISS" = true ]; then
        echo "  - $OUTPUT_DIR/knowledge_index.faiss"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Update model config to use these knowledge base files"
    echo "  2. Load knowledge base in MedicalKnowledgeRetriever"
    echo "  3. Start training with RAG-enhanced model"
else
    echo ""
    echo "❌ Knowledge base construction failed. Please check the error messages above."
    exit 1
fi

# ============================================================
# 可选：验证知识库
# ============================================================

echo ""
echo "==========================================="
echo "Verifying Knowledge Base"
echo "==========================================="

python -c "
import torch
import json
from pathlib import Path

# 加载 embeddings
emb_path = Path('$OUTPUT_DIR') / 'knowledge_embeddings.pt'
embeddings = torch.load(emb_path)
print(f'✓ Embeddings shape: {embeddings.shape}')

# 加载文本
text_path = Path('$OUTPUT_DIR') / 'knowledge_texts.json'
with open(text_path) as f:
    texts = json.load(f)
print(f'✓ Number of texts: {len(texts)}')

# 加载元数据
meta_path = Path('$OUTPUT_DIR') / 'metadata.json'
with open(meta_path) as f:
    metadata = json.load(f)
print(f'✓ Embedding dimension: {metadata[\"embedding_dim\"]}')

# 一致性检查
assert embeddings.shape[0] == len(texts), 'Mismatch between embeddings and texts!'
print('\n✅ Knowledge base verification passed!')
"

echo "==========================================="
