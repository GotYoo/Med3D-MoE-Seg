#!/bin/bash

# 快速测试脚本 - 验证代码更新
# 测试数据加载和 RAG 知识库集成

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "Med3D-MoE-Seg Code Integration Test"
echo "======================================================================"
echo ""

# 进入项目目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# ====================================================================
# Test 1: 检查文件完整性
# ====================================================================
echo "======================================================================" 
echo "[Test 1] Checking file integrity..."
echo "======================================================================"

REQUIRED_FILES=(
    "config/med3d_lisa_full.yaml"
    "data/lidc_dataset.py"
    "data/builder.py"
    "model/rag/retriever.py"
    "train_net.py"
    "CODE_UPDATES.md"
    "RAG_KNOWLEDGE_BASE.md"
    "DATA_PREPARATION.md"
)

echo "Checking required files:"
ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "❌ Some required files are missing!"
    exit 1
fi

echo ""
echo "✅ All required files exist"
echo ""

# ====================================================================
# Test 2: 检查数据划分输出
# ====================================================================
echo "======================================================================"
echo "[Test 2] Checking data split outputs..."
echo "======================================================================"

DATA_SPLIT_DIR="data_splits"

if [ -d "$DATA_SPLIT_DIR" ]; then
    echo "Data split directory exists: $DATA_SPLIT_DIR"
    
    SPLIT_FILES=("train.json" "val.json" "test.json" "split_info.json")
    SPLITS_EXIST=true
    
    for file in "${SPLIT_FILES[@]}"; do
        if [ -f "$DATA_SPLIT_DIR/$file" ]; then
            NUM_LINES=$(wc -l < "$DATA_SPLIT_DIR/$file")
            FILE_SIZE=$(du -h "$DATA_SPLIT_DIR/$file" | cut -f1)
            echo "  ✓ $file (Lines: $NUM_LINES, Size: $FILE_SIZE)"
        else
            echo "  ✗ $file (MISSING)"
            SPLITS_EXIST=false
        fi
    done
    
    if [ "$SPLITS_EXIST" = false ]; then
        echo ""
        echo "⚠️  Data splits incomplete. Run: bash scripts/prepare_data.sh"
    else
        echo ""
        echo "✅ Data splits ready"
    fi
else
    echo "⚠️  Data split directory not found: $DATA_SPLIT_DIR"
    echo "   Run: bash scripts/prepare_data.sh"
fi

echo ""

# ====================================================================
# Test 3: 检查 RAG 知识库
# ====================================================================
echo "======================================================================"
echo "[Test 3] Checking RAG knowledge base..."
echo "======================================================================"

RAG_DB_DIR="assets/rag_db"

if [ -d "$RAG_DB_DIR" ]; then
    echo "RAG database directory exists: $RAG_DB_DIR"
    
    RAG_FILES=("knowledge_embeddings.pt" "knowledge_texts.json" "metadata.json")
    RAG_EXIST=true
    
    for file in "${RAG_FILES[@]}"; do
        if [ -f "$RAG_DB_DIR/$file" ]; then
            FILE_SIZE=$(du -h "$RAG_DB_DIR/$file" | cut -f1)
            echo "  ✓ $file (Size: $FILE_SIZE)"
        else
            echo "  ✗ $file (MISSING)"
            RAG_EXIST=false
        fi
    done
    
    if [ "$RAG_EXIST" = false ]; then
        echo ""
        echo "⚠️  RAG knowledge base incomplete. Run: bash scripts/build_rag.sh"
    else
        echo ""
        echo "✅ RAG knowledge base ready"
        
        # 显示知识库统计
        if [ -f "$RAG_DB_DIR/metadata.json" ]; then
            echo ""
            echo "Knowledge base statistics:"
            python3 -c "
import json
with open('$RAG_DB_DIR/metadata.json') as f:
    meta = json.load(f)
print(f\"  - Entries: {meta.get('num_entries', 'N/A')}\")
print(f\"  - Embedding dim: {meta.get('embedding_dim', 'N/A')}\")
print(f\"  - Model: {meta.get('biobert_model', 'N/A')}\")
" 2>/dev/null || echo "  (Failed to read metadata)"
        fi
    fi
else
    echo "⚠️  RAG database directory not found: $RAG_DB_DIR"
    echo "   Run: bash scripts/build_rag.sh"
fi

echo ""

# ====================================================================
# Test 4: 测试 Python 导入
# ====================================================================
echo "======================================================================"
echo "[Test 4] Testing Python imports..."
echo "======================================================================"

echo "Testing imports:"
python3 << EOF
import sys
import os

# 添加项目路径
sys.path.insert(0, os.getcwd())

errors = []

# Test 1: LIDC Dataset
try:
    from data.lidc_dataset import LIDCDataset, create_lidc_dataloaders
    print("  ✓ data.lidc_dataset")
except Exception as e:
    print(f"  ✗ data.lidc_dataset: {e}")
    errors.append(("lidc_dataset", e))

# Test 2: Data Builder
try:
    from data.builder import build_dataloaders_from_config
    print("  ✓ data.builder (updated)")
except Exception as e:
    print(f"  ✗ data.builder: {e}")
    errors.append(("builder", e))

# Test 3: RAG Retriever
try:
    from model.rag.retriever import MedicalKnowledgeRetriever
    print("  ✓ model.rag.retriever (updated)")
except Exception as e:
    print(f"  ✗ model.rag.retriever: {e}")
    errors.append(("retriever", e))

# Test 4: Med3DLISA
try:
    from model.meta_arch.med3d_lisa import Med3DLISA_Full
    print("  ✓ model.meta_arch.med3d_lisa")
except Exception as e:
    print(f"  ✗ model.meta_arch.med3d_lisa: {e}")
    errors.append(("med3d_lisa", e))

if errors:
    print(f"\n❌ {len(errors)} import(s) failed")
    sys.exit(1)
else:
    print("\n✅ All imports successful")
    sys.exit(0)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Some imports failed. Check dependencies."
    echo ""
fi

# ====================================================================
# Test 5: 测试配置文件加载
# ====================================================================
echo "======================================================================"
echo "[Test 5] Testing config file loading..."
echo "======================================================================"

python3 << EOF
import yaml
import sys

try:
    with open('config/med3d_lisa_full.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Config loaded successfully")
    print(f"  - Model type: {config.get('model', {}).get('type', 'N/A')}")
    print(f"  - Dataset type: {config.get('data', {}).get('dataset_type', 'N/A')}")
    print(f"  - RAG enabled: {config.get('model', {}).get('rag', {}).get('enabled', 'N/A')}")
    print(f"  - Image size: {config.get('data', {}).get('image_size', 'N/A')}")
    
    print("\n✅ Config file valid")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Config loading failed: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Config file loading failed"
    echo ""
fi

# ====================================================================
# Test 6: 测试 RAG Retriever 加载
# ====================================================================
echo ""
echo "======================================================================"
echo "[Test 6] Testing RAG Retriever with knowledge base..."
echo "======================================================================"

if [ -f "$RAG_DB_DIR/knowledge_embeddings.pt" ] && [ -f "$RAG_DB_DIR/knowledge_texts.json" ]; then
    python3 << EOF
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import torch
    from model.rag.retriever import MedicalKnowledgeRetriever
    
    print("Creating MedicalKnowledgeRetriever...")
    retriever = MedicalKnowledgeRetriever(
        knowledge_embed_path='$RAG_DB_DIR/knowledge_embeddings.pt',
        knowledge_texts_path='$RAG_DB_DIR/knowledge_texts.json',
        knowledge_dim=768,
        llm_hidden_size=4096,
        top_k=3
    )
    
    print(f"\nRetriever info:")
    print(f"  - Knowledge entries: {retriever.num_knowledge_entries}")
    print(f"  - Knowledge dim: {retriever.knowledge_dim}")
    print(f"  - Top-K: {retriever.top_k}")
    print(f"  - Knowledge texts loaded: {len(retriever.knowledge_texts) if retriever.knowledge_texts else 0}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    query = torch.randn(1, 768)
    outputs = retriever(query, return_details=True)
    
    print(f"  - Context embed shape: {outputs['context_embed'].shape}")
    print(f"  - Top-3 indices: {outputs['retrieved_indices'][0].tolist()}")
    print(f"  - Top-3 scores: {[f'{s:.4f}' for s in outputs['relevance_scores'][0].tolist()]}")
    
    if 'retrieved_texts' in outputs:
        print(f"  - Retrieved texts: {len(outputs['retrieved_texts'][0])} items")
        print(f"    Example: {outputs['retrieved_texts'][0][0]['text'][:80]}...")
    
    print("\n✅ RAG Retriever test passed")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ RAG Retriever test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  RAG Retriever test failed"
        echo ""
    fi
else
    echo "⚠️  Knowledge base files not found, skipping RAG test"
    echo "   Run: bash scripts/build_rag.sh"
    echo ""
fi

# ====================================================================
# 总结
# ====================================================================
echo "======================================================================"
echo "Test Summary"
echo "======================================================================"
echo ""
echo "✅ File integrity check passed"
echo "✅ Python imports working"
echo "✅ Config file loading successful"

if [ -d "$DATA_SPLIT_DIR" ] && [ -f "$DATA_SPLIT_DIR/train.json" ]; then
    echo "✅ Data splits ready"
else
    echo "⚠️  Data splits missing - run: bash scripts/prepare_data.sh"
fi

if [ -d "$RAG_DB_DIR" ] && [ -f "$RAG_DB_DIR/knowledge_embeddings.pt" ]; then
    echo "✅ RAG knowledge base ready"
else
    echo "⚠️  RAG knowledge base missing - run: bash scripts/build_rag.sh"
fi

echo ""
echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo ""
echo "1. If data splits missing:"
echo "   bash scripts/prepare_data.sh"
echo ""
echo "2. If RAG knowledge base missing:"
echo "   bash scripts/build_rag.sh"
echo ""
echo "3. Start training:"
echo "   python train_net.py --config_file config/med3d_lisa_full.yaml"
echo ""
echo "4. For more details, see:"
echo "   - CODE_UPDATES.md"
echo "   - DATA_PREPARATION.md"
echo "   - RAG_KNOWLEDGE_BASE.md"
echo ""
echo "======================================================================"
