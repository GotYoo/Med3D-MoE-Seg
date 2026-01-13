import torch
import os
import json
import glob

# --- 配置区域 ---
# 您的 LLaVA-v1.5-7b 模型路径
MODEL_PATH = "/mnt/disk4t0/publicData/huggingface_models/llava-v1.5-7b"
# 提取出的 Projector 保存路径
OUTPUT_PATH = "/mnt/disk4t0/publicData/huggingface_models/llava-v1.5-7b/mm_projector.bin"
# ----------------

def extract_projector():
    print(f"📂 模型路径: {MODEL_PATH}")
    
    projector_weights = {}
    found_shards = []

    # 1. 检查索引文件 (通常大模型是分片的)
    index_file = os.path.join(MODEL_PATH, "pytorch_model.bin.index.json")
    
    if os.path.exists(index_file):
        print("🔍 发现分片索引文件，正在解析...")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # 找出包含 'mm_projector' 的分片文件
        target_shards = set()
        weight_map = index_data.get("weight_map", {})
        for key, shard_name in weight_map.items():
            if "mm_projector" in key:
                target_shards.add(shard_name)
        
        found_shards = [os.path.join(MODEL_PATH, s) for s in target_shards]
        print(f"🎯 Projector 权重位于以下分片: {list(target_shards)}")

    else:
        # 2. 如果没有索引，尝试查找所有的 .bin 文件
        print("🔍 未找到索引文件，扫描所有 .bin 文件...")
        found_shards = glob.glob(os.path.join(MODEL_PATH, "*.bin"))

    if not found_shards:
        print("❌ 错误: 未找到任何模型权重文件 (.bin)！请检查路径。")
        return

    # 3. 遍历分片提取权重
    print("🚀 开始提取...")
    for shard_path in found_shards:
        print(f"📖 读取: {os.path.basename(shard_path)} ...")
        try:
            state_dict = torch.load(shard_path, map_location="cpu")
            
            count = 0
            for key, value in state_dict.items():
                # 筛选关键字
                if "mm_projector" in key:
                    projector_weights[key] = value
                    count += 1
            
            if count > 0:
                print(f"   ✅ 提取了 {count} 个相关参数")
        except Exception as e:
            print(f"   ⚠️ 读取失败: {e}")

    # 4. 保存结果
    if len(projector_weights) > 0:
        print(f"💾 正在保存到: {OUTPUT_PATH}")
        torch.save(projector_weights, OUTPUT_PATH)
        print(f"🎉 成功！共保存 {len(projector_weights)} 个张量。")
        print("现在您可以在 Stage 2 脚本中使用这个文件了。")
    else:
        print("❌ 失败: 在模型文件中未找到任何包含 'mm_projector' 的参数。")

if __name__ == "__main__":
    extract_projector()