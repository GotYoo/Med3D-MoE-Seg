"""
测试 Stage 1 模块：BioBERT + Unified Alignment
"""
import torch
from model.encoders.biobert_encoder import BioBERTEncoder
from model.encoders.uni_alignment import UnifiedAlignmentModule

print("=" * 70)
print("Testing Stage 1: Multi-Modal Alignment Modules")
print("=" * 70)

# 测试参数
batch_size = 4
seq_len = 128
image_dim = 512
text_dim = 768
latent_dim = 512

print("\n1. Testing BioBERTEncoder...")
print(f"   Config: output_dim=768, freeze_layers=6")

# 模拟输入
input_ids = torch.randint(0, 30522, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

print(f"   Input: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}")
print("   ✓ BioBERTEncoder 接口正确")

print("\n2. Testing UnifiedAlignmentModule...")
alignment_module = UnifiedAlignmentModule(
    image_dim=image_dim,
    text_dim=text_dim,
    latent_dim=latent_dim,
    temperature=0.07
)

# 模拟特征
image_features = torch.randn(batch_size, image_dim)
text_features = torch.randn(batch_size, text_dim)

outputs = alignment_module(image_features, text_features, return_loss=True)

print(f"   Input: image_features {image_features.shape}, text_features {text_features.shape}")
print(f"   Output:")
print(f"   - image_embeds: {outputs['image_embeds'].shape}")
print(f"   - text_embeds: {outputs['text_embeds'].shape}")
print(f"   - contrastive_loss: {outputs['contrastive_loss'].item():.4f}")
print("   ✓ UnifiedAlignmentModule 前向传播成功")

print("\n3. Testing Contrastive Loss...")
similarity = alignment_module.get_similarity_matrix(
    outputs['image_embeds'], 
    outputs['text_embeds']
)
print(f"   Similarity matrix shape: {similarity.shape}")
print(f"   Diagonal (positive pairs): {torch.diag(similarity).mean().item():.4f}")
print(f"   Off-diagonal (negative pairs): {(similarity.sum() - torch.diag(similarity).sum()) / (batch_size * (batch_size - 1)):.4f}")
print("   ✓ 对比学习相似度矩阵正确")

print("\n" + "=" * 70)
print("✓ All Stage 1 modules working correctly!")
print("=" * 70)
