import torch

# 1. 输入文件 (你当前报错的那个文件)
input_path = "checkpoints/teacher_vit_fixed.pt"
# 2. 输出文件 (最终修复版)
output_path = "checkpoints/teacher_vit_final.pt"

print(f"Loading {input_path}...")
try:
    state_dict = torch.load(input_path, map_location="cpu")
except FileNotFoundError:
    # 容错：如果找不到 fixed，就找原始的
    input_path = "checkpoints/teacher_vit.pt"
    print(f"File not found, trying {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu")

new_state_dict = {}
renamed_count = 0

print(f"Original keys sample: {list(state_dict.keys())[:3]}")

for k, v in state_dict.items():
    new_k = k
    # 暴力剥离所有可能的乱七八糟前缀
    if new_k.startswith("visual_transformer."):
        new_k = new_k.replace("visual_transformer.", "")
    elif new_k.startswith("clip_model.visual.transformer."):
        new_k = new_k.replace("clip_model.visual.transformer.", "")
    elif new_k.startswith("clip_model.visual."):
        new_k = new_k.replace("clip_model.visual.", "")
        
    if new_k != k:
        renamed_count += 1
    
    new_state_dict[new_k] = v

print(f"Fixed keys sample: {list(new_state_dict.keys())[:3]}")
print(f"Renamed {renamed_count} keys.")

torch.save(new_state_dict, output_path)
print(f"✅ Saved FINAL weights to {output_path}")