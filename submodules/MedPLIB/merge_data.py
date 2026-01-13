import json
import os
import random

# --- 配置区域：请修改为您实际的文件名 ---
# 这里的路径假设您已经把 json 放到了 /mnt/disk4t0/publicData/MeCoVQA/ 目录下
BASE_DIR = "/mnt/disk4t0/publicData/MeCoVQA/MeCoVQA/test"

# 您下载下来的源文件（请根据您目录里的实际文件名修改！）
# 只要是用于训练的 json 都加到这个列表里
SOURCE_FILES = [
    "MeCoVQA-Complex.json",  # 示例文件名，请核对您本地的实际名称
    "MeCoVQA-Region.json",   # 示例文件名
    # "MeCoVQA_Public_train.json"  # 如果有 Public 数据就加上，没有就注释掉
]

# 输出的目标文件名 (脚本里用的那个名字)
OUTPUT_TRAIN_FILE = "MeCoVQA_Complex+Region_VQA_train+Public_VQA.json"
OUTPUT_TEST_FILE = "MeCoVQA_Complex_VQA_test_rand200.json"

# 测试集源文件 (用于生成 rand200)
SOURCE_TEST_FILE = "MeCoVQA_Complex_VQA_test.json" # 请核对实际文件名
# ------------------------------------

def merge_json_files():
    merged_data = []
    print(f"🚀 开始合并训练数据...")
    
    for filename in SOURCE_FILES:
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ 警告: 找不到文件 {filepath}，已跳过。")
            continue
            
        print(f"📖 读取: {filename}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                    print(f"   -> 添加了 {len(data)} 条数据")
                else:
                    print(f"   ❌ 格式错误: {filename} 不是列表格式")
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")

    # 保存合并后的训练文件
    save_path = os.path.join(BASE_DIR, OUTPUT_TRAIN_FILE)
    print(f"💾 正在保存合并文件: {save_path}")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f)
    print(f"✅ 训练集合并完成！总数据量: {len(merged_data)}")

def create_random_test_file():
    print(f"\n🚀 开始生成测试集 (Rand200)...")
    src_path = os.path.join(BASE_DIR, SOURCE_TEST_FILE)
    dst_path = os.path.join(BASE_DIR, OUTPUT_TEST_FILE)
    
    if not os.path.exists(src_path):
        print(f"⚠️ 找不到测试源文件 {src_path}，无法生成 rand200 文件。")
        print("提示：您可以直接在训练脚本中使用完整的测试集文件名。")
        return

    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机抽取 200 条（如果不足 200 就全取）
    sample_size = min(200, len(data))
    sampled_data = random.sample(data, sample_size)
    
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f)
    print(f"✅ 已从 {len(data)} 条数据中随机抽取 {sample_size} 条保存至 {OUTPUT_TEST_FILE}")

if __name__ == "__main__":
    merge_json_files()
    create_random_test_file()