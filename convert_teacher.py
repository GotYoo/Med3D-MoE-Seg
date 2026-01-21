import torch
import os

# 1. 你的源文件路径
source_path = "checkpoints/CT-CLIP_v2.pt"
# 2. 目标文件路径
target_path = "checkpoints/teacher_vit.pt"

print(f"Loading {source_path}...")
state_dict = torch.load(source_path, map_location="cpu")

# 有些 checkpoint 会把权重包在 'state_dict' 或 'model' 里
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
elif "model" in state_dict:
    state_dict = state_dict["model"]

print(f"Total keys: {len(state_dict)}")

# 3. 提取视觉相关的权重
# CT-CLIP 的视觉部分通常包含 'visual' 或 'clip.visual' 前缀
# 我们需要把这些前缀去掉，以匹配 CTViT 的键名
new_state_dict = {}
vision_prefix = "clip_model.visual.transformer." # 可能的前缀，根据实际情况调整

# 先打印前几个 key 看看结构
print("Top 10 keys in checkpoint:", list(state_dict.keys())[:10])

for k, v in state_dict.items():
    # 过滤规则：只保留视觉 Transformer 的部分
    # 注意：这里需要根据打印出来的 key 灵活调整逻辑
    # 假设 key 是 "clip_model.visual.transformer.layers.0..."
    # 我们需要把它变成 "layers.0..."
    
    # 常见的几种前缀情况：
    if k.startswith("clip_model.visual."):
        new_k = k.replace("clip_model.visual.", "")
        new_state_dict[new_k] = v
    elif k.startswith("visual."):
        new_k = k.replace("visual.", "")
        new_state_dict[new_k] = v
    elif "temporal_transformer" in k or "spatial_transformer" in k:
        # 如果没有前缀，直接保留
        new_state_dict[k] = v

print(f"Extracted {len(new_state_dict)} vision keys.")

if len(new_state_dict) > 0:
    torch.save(new_state_dict, target_path)
    print(f"✅ Saved clean teacher weights to {target_path}")
    print("Now update your config to point to this new file!")
else:
    print("❌ Failed to extract any keys. Please check the prefix logic.")