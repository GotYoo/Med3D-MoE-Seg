"""
集成测试：所有新增模块（Stage 1, 2, 4）
"""
import torch
from model.encoders.biobert_encoder import BioBERTEncoder
from model.encoders.uni_alignment import UnifiedAlignmentModule
from model.rag.retriever import MedicalKnowledgeRetriever
from model.correction.consistency import ConsistencyChecker

print("=" * 70)
print("Med3D-MoE-Seg: Complete Module Integration Test")
print("=" * 70)

# 参数配置
batch_size = 2
seq_len = 128
image_dim = 512
text_dim = 768
latent_dim = 512
llm_hidden_size = 4096
mask_size = (32, 64, 64)

print("\n【Stage 1: Multi-Modal Alignment】")
print("-" * 70)

# 1. 模拟输入
print("1. Preparing inputs...")
# 图像特征（来自 CT-CLIP）
image_features = torch.randn(batch_size, image_dim)
# 临床文本（BioBERT 处理）
text_features = torch.randn(batch_size, text_dim)
print(f"   ✓ Image features: {image_features.shape}")
print(f"   ✓ Text features: {text_features.shape}")

# 2. 统一对齐
print("\n2. Unified Alignment...")
alignment_module = UnifiedAlignmentModule(
    image_dim=image_dim,
    text_dim=text_dim,
    latent_dim=latent_dim
)
align_outputs = alignment_module(image_features, text_features, return_loss=True)
print(f"   ✓ Aligned image embeds: {align_outputs['image_embeds'].shape}")
print(f"   ✓ Aligned text embeds: {align_outputs['text_embeds'].shape}")
print(f"   ✓ Contrastive loss: {align_outputs['contrastive_loss'].item():.4f}")

print("\n【Stage 2: RAG Knowledge Retrieval】")
print("-" * 70)

# 3. 知识检索
print("3. Medical Knowledge Retrieval...")
retriever = MedicalKnowledgeRetriever(
    knowledge_dim=latent_dim,
    llm_hidden_size=llm_hidden_size,
    top_k=3,
    num_knowledge_entries=500
)

# 使用对齐后的特征作为查询
query_embed = align_outputs['image_embeds']
rag_outputs = retriever(query_embed, return_details=True)
print(f"   ✓ Retrieved knowledge: {rag_outputs['retrieved_embeds'].shape}")
print(f"   ✓ Context embed (for LLM): {rag_outputs['context_embed'].shape}")
print(f"   ✓ Relevance scores: {rag_outputs['relevance_scores'][0].tolist()}")

print("\n【Stage 3: MoE LLM Reasoning】")
print("-" * 70)
print("4. LLM Processing (simulated)...")
# 模拟 LLM 输出
llm_text_embeds = torch.randn(batch_size, seq_len, llm_hidden_size)
llm_mask_output = torch.randn(batch_size, 1, *mask_size)
print(f"   ✓ LLM text output: {llm_text_embeds.shape}")
print(f"   ✓ Generated mask: {llm_mask_output.shape}")

print("\n【Stage 4: Self-Correction Loop】")
print("-" * 70)

# 5. 一致性检查
print("5. Consistency Check...")
checker = ConsistencyChecker(
    mask_channels=256,
    text_hidden_size=llm_hidden_size,
    embed_dim=512,
    num_heads=8
)

consistency_outputs = checker(llm_mask_output, llm_text_embeds, return_attention=True)
print(f"   ✓ Consistency scores: {consistency_outputs['consistency_score'].squeeze().tolist()}")
print(f"   ✓ Attention weights: {consistency_outputs['attention_weights'].shape}")

# 6. 判断是否需要重新细化
threshold = 0.7
scores = consistency_outputs['consistency_score'].squeeze()
needs_refinement = (scores < threshold).any()
print(f"\n6. Refinement Decision (threshold={threshold})...")
print(f"   Scores: {scores.tolist()}")
print(f"   Needs refinement: {needs_refinement}")

print("\n" + "=" * 70)
print("Pipeline Summary:")
print("=" * 70)
print("Stage 1: Multi-Modal Input → Unified Alignment")
print("         ✓ Image + Text → Contrastive Learning")
print("\nStage 2: RAG → Knowledge Retrieval & Injection")
print("         ✓ Query → Top-K Knowledge → Context for LLM")
print("\nStage 3: MoE LLM → Reasoning & Generation")
print("         ✓ Enhanced Input → Report + Segmentation")
print("\nStage 4: Self-Correction → Consistency Check")
print("         ✓ Text-Mask Matching → Iterative Refinement")

print("\n" + "=" * 70)
print("✓ All new modules integrated successfully!")
print("=" * 70)
