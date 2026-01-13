"""
RAG 知识库构建脚本
将医学知识文本编码为向量并构建检索索引

使用方法:
    python scripts/build_rag_index.py \
        --input_file data/medical_knowledge.txt \
        --output_dir assets/rag_db \
        --biobert_model dmis-lab/biobert-v1.1 \
        --batch_size 32 \
        --use_faiss
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.encoders.biobert_encoder import BioBERTEncoder


def load_knowledge_texts(input_file: Path) -> List[Dict[str, str]]:
    """
    加载医学知识文本
    
    Args:
        input_file: 输入文本文件路径
    
    Returns:
        knowledge_list: 知识条目列表
    """
    knowledge_list = []
    
    print(f"Loading knowledge from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 如果是 JSON 格式（每行一个 JSON 对象）
            if line.startswith('{'):
                try:
                    item = json.loads(line)
                    knowledge_list.append({
                        'id': item.get('id', f'k{idx}'),
                        'text': item.get('text', line),
                        'source': item.get('source', 'unknown'),
                        'category': item.get('category', 'general')
                    })
                except json.JSONDecodeError:
                    # 如果解析失败，当作纯文本处理
                    knowledge_list.append({
                        'id': f'k{idx}',
                        'text': line,
                        'source': 'text_file',
                        'category': 'general'
                    })
            else:
                # 纯文本格式
                knowledge_list.append({
                    'id': f'k{idx}',
                    'text': line,
                    'source': 'text_file',
                    'category': 'general'
                })
    
    print(f"Loaded {len(knowledge_list)} knowledge entries")
    
    return knowledge_list


def encode_knowledge_batch(
    texts: List[str],
    encoder: BioBERTEncoder,
    tokenizer,
    max_length: int = 512,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    批量编码文本为向量
    
    Args:
        texts: 文本列表
        encoder: BioBERT 编码器
        tokenizer: BioBERT tokenizer
        max_length: 最大序列长度
        device: 设备
    
    Returns:
        embeddings: [batch_size, hidden_size] 向量
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 移动到设备
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # 编码
    with torch.no_grad():
        embeddings = encoder(input_ids, attention_mask)
    
    return embeddings.cpu()


def build_knowledge_base(
    input_file: Path,
    output_dir: Path,
    biobert_model: str = 'dmis-lab/biobert-v1.1',
    batch_size: int = 32,
    max_length: int = 512,
    use_faiss: bool = False,
    device: str = None
):
    """
    构建 RAG 知识库
    
    Args:
        input_file: 输入文本文件
        output_dir: 输出目录
        biobert_model: BioBERT 模型名称
        batch_size: 批处理大小
        max_length: 最大序列长度
        use_faiss: 是否使用 FAISS 索引
        device: 设备 (cuda/cpu)
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备选择
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("RAG Knowledge Base Construction")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"BioBERT model: {biobert_model}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"Device: {device}")
    print(f"Use FAISS: {use_faiss}")
    print()
    
    # 1. 加载知识文本
    print("[Step 1] Loading knowledge texts...")
    knowledge_list = load_knowledge_texts(input_file)
    
    if len(knowledge_list) == 0:
        raise ValueError("No knowledge entries found in input file")
    
    # 2. 初始化 BioBERT 编码器
    print("\n[Step 2] Initializing BioBERT encoder...")
    encoder = BioBERTEncoder(
        model_name=biobert_model,
        freeze_layers=12  # 完全冻结用于推理
    )
    encoder.eval()
    encoder.to(device)
    
    # 创建 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(biobert_model)
    
    print(f"✓ BioBERT encoder loaded")
    print(f"  Hidden size: {encoder.hidden_size}")
    
    # 3. 批量编码知识文本
    print("\n[Step 3] Encoding knowledge texts...")
    all_embeddings = []
    
    texts = [item['text'] for item in knowledge_list]
    
    # 使用 tqdm 显示进度
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = encode_knowledge_batch(
            batch_texts,
            encoder,
            tokenizer,
            max_length=max_length,
            device=device
        )
        all_embeddings.append(batch_embeddings)
    
    # 合并所有批次
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    print(f"\n✓ Encoded {all_embeddings.shape[0]} knowledge entries")
    print(f"  Embedding shape: {all_embeddings.shape}")
    
    # 4. 保存 Embeddings
    print("\n[Step 4] Saving knowledge base...")
    
    # 保存 embeddings
    embeddings_path = output_dir / 'knowledge_embeddings.pt'
    torch.save(all_embeddings, embeddings_path)
    print(f"✓ Saved embeddings to: {embeddings_path}")
    
    # 保存文本
    texts_path = output_dir / 'knowledge_texts.json'
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_list, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved texts to: {texts_path}")
    
    # 保存元数据
    metadata = {
        'num_entries': len(knowledge_list),
        'embedding_dim': encoder.hidden_size,
        'biobert_model': biobert_model,
        'max_length': max_length,
        'created_date': str(Path(__file__).stat().st_mtime)
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # 5. 可选：构建 FAISS 索引
    if use_faiss:
        try:
            import faiss
            
            print("\n[Step 5] Building FAISS index...")
            
            # 转换为 numpy
            embeddings_np = all_embeddings.numpy()
            
            # 创建 FAISS 索引 (Inner Product / Cosine Similarity)
            # 归一化向量用于余弦相似度
            faiss.normalize_L2(embeddings_np)
            
            # 使用 IndexFlatIP (Inner Product) 进行余弦相似度搜索
            index = faiss.IndexFlatIP(encoder.hidden_size)
            index.add(embeddings_np)
            
            # 保存 FAISS 索引
            faiss_path = output_dir / 'knowledge_index.faiss'
            faiss.write_index(index, str(faiss_path))
            
            print(f"✓ Built and saved FAISS index to: {faiss_path}")
            print(f"  Index size: {index.ntotal} vectors")
            print(f"  Index type: IndexFlatIP (Cosine Similarity)")
        
        except ImportError:
            print("\n⚠ FAISS not installed. Skipping FAISS index creation.")
            print("  Install with: pip install faiss-cpu  or  pip install faiss-gpu")
    
    # 6. 统计信息
    print("\n" + "=" * 70)
    print("Knowledge Base Statistics")
    print("=" * 70)
    
    # 按类别统计
    categories = {}
    sources = {}
    text_lengths = []
    
    for item in knowledge_list:
        category = item.get('category', 'general')
        source = item.get('source', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        text_lengths.append(len(item['text']))
    
    print(f"Total entries: {len(knowledge_list)}")
    print(f"Embedding dimension: {encoder.hidden_size}")
    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count} ({count/len(knowledge_list)*100:.1f}%)")
    
    print(f"\nSources:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  - {src}: {count}")
    
    import numpy as np
    print(f"\nText lengths:")
    print(f"  - Mean: {np.mean(text_lengths):.1f} chars")
    print(f"  - Median: {np.median(text_lengths):.1f} chars")
    print(f"  - Min: {min(text_lengths)} chars")
    print(f"  - Max: {max(text_lengths)} chars")
    
    print("\n" + "=" * 70)
    print("✅ Knowledge base construction completed!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {embeddings_path}")
    print(f"  - {texts_path}")
    print(f"  - {metadata_path}")
    if use_faiss:
        print(f"  - {output_dir / 'knowledge_index.faiss'}")
    
    print(f"\nTo use in training, update your config:")
    print(f"  rag:")
    print(f"    knowledge_embeddings: {embeddings_path}")
    print(f"    knowledge_texts: {texts_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG knowledge base from medical texts"
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input text file with medical knowledge (one entry per line)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='assets/rag_db',
        help='Output directory for knowledge base (default: assets/rag_db)'
    )
    parser.add_argument(
        '--biobert_model',
        type=str,
        default='dmis-lab/biobert-v1.1',
        help='BioBERT model name or path (default: dmis-lab/biobert-v1.1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--use_faiss',
        action='store_true',
        help='Build FAISS index for fast retrieval'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # 转换为 Path 对象
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # 检查输入文件
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # 构建知识库
    build_knowledge_base(
        input_file=input_file,
        output_dir=output_dir,
        biobert_model=args.biobert_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_faiss=args.use_faiss,
        device=args.device
    )


if __name__ == '__main__':
    main()
