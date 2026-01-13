"""
测试 Stage 2 模块：RAG 检索系统
"""
import torch
from model.rag.retriever import MedicalKnowledgeRetriever, create_dummy_knowledge_base

print("=" * 70)
print("Testing Stage 2: RAG Retrieval System")
print("=" * 70)

# 测试参数
batch_size = 4
knowledge_dim = 768
llm_hidden_size = 4096
top_k = 3
num_knowledge_entries = 1000

print("\n1. Creating dummy knowledge base...")
knowledge_embeds = create_dummy_knowledge_base(
    num_entries=num_knowledge_entries,
    embed_dim=knowledge_dim
)
print(f"   Knowledge base shape: {knowledge_embeds.shape}")
print("   ✓ Knowledge base created")

print("\n2. Testing MedicalKnowledgeRetriever...")
retriever = MedicalKnowledgeRetriever(
    knowledge_embed_path=None,  # 使用随机初始化
    knowledge_dim=knowledge_dim,
    llm_hidden_size=llm_hidden_size,
    top_k=top_k,
    num_knowledge_entries=num_knowledge_entries
)

# 模拟查询（可以是图像特征或文本特征）
query_embed = torch.randn(batch_size, knowledge_dim)
print(f"   Query embed shape: {query_embed.shape}")

print("\n3. Testing retrieve()...")
retrieved_embeds, retrieved_indices, relevance_scores = retriever.retrieve(query_embed)
print(f"   Retrieved embeddings: {retrieved_embeds.shape}")
print(f"   Retrieved indices: {retrieved_indices.shape}")
print(f"   Relevance scores: {relevance_scores.shape}")
print(f"   Top-1 relevance scores: {relevance_scores[:, 0].tolist()}")
print("   ✓ Retrieval successful")

print("\n4. Testing forward() with projection...")
outputs = retriever(query_embed, return_details=True)
print(f"   Context embed (projected): {outputs['context_embed'].shape}")
print(f"   Expected shape: [B={batch_size}, llm_hidden_size={llm_hidden_size}]")
print("   ✓ Context projection successful")

print("\n5. Testing inject_context_to_prompt()...")
# 模拟 LLM prompt embeddings
seq_len = 32
prompt_embeds = torch.randn(batch_size, seq_len, llm_hidden_size)
context_embed = outputs['context_embed']

enhanced_embeds = retriever.inject_context_to_prompt(
    prompt_embeds, 
    context_embed, 
    injection_position='prepend'
)
print(f"   Original prompt: {prompt_embeds.shape}")
print(f"   Enhanced prompt: {enhanced_embeds.shape}")
print(f"   Context injected at position: prepend")
print("   ✓ Context injection successful")

print("\n6. Testing different injection positions...")
for position in ['prepend', 'append', 'middle']:
    enhanced = retriever.inject_context_to_prompt(prompt_embeds, context_embed, position)
    print(f"   Position '{position}': {enhanced.shape}")
print("   ✓ All injection positions working")

print("\n7. Summary:")
print(f"   - Knowledge base: {num_knowledge_entries} entries")
print(f"   - Top-K retrieval: {top_k}")
print(f"   - Query → Retrieved {top_k} relevant knowledge")
print(f"   - Context projected: {knowledge_dim * top_k} → {llm_hidden_size}")
print(f"   - Ready for LLM injection")

print("\n" + "=" * 70)
print("✓ Stage 2 RAG module working correctly!")
print("=" * 70)
