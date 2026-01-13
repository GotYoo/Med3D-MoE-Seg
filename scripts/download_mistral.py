import os
from modelscope import snapshot_download

# ================= 配置 =================
# 目标路径: Med3D-MoE-Seg/assets/weights/mistral-7b-v0.2
TARGET_DIR = "./assets/weights/mistral-7b-v0.2"

# ModelScope 上的模型 ID (这是官方镜像，速度极快)
MODEL_ID = "AI-ModelScope/Mistral-7B-Instruct-v0.2"
# =======================================

def download_weights():
    print(f"🚀 开始下载 Mistral-7B-Instruct-v0.2 到 {TARGET_DIR} ...")
    
    # snapshot_download 会自动处理断点续传
    # cache_dir 指定下载缓存位置，revision 指定版本
    model_dir = snapshot_download(
        MODEL_ID, 
        cache_dir='./assets/weights/temp_cache', # 先下载到临时缓存
        revision='master' 
    )
    
    # 移动/软链接到最终目录 (ModelScope 下载下来的目录名是哈希值，我们整理一下)
    if not os.path.exists(TARGET_DIR):
        os.makedirs(os.path.dirname(TARGET_DIR), exist_ok=True)
        # 将下载好的模型文件夹重命名/移动到我们想要的规范路径
        os.rename(model_dir, TARGET_DIR)
        print(f"✅ 下载并整理完成！模型位于: {TARGET_DIR}")
        
        # 清理空缓存文件夹
        try:
            os.rmdir('./assets/weights/temp_cache')
        except:
            pass
    else:
        print(f"⚠️ 目标目录 {TARGET_DIR} 已存在，跳过移动操作。")
        print(f"   原始下载路径: {model_dir}")

if __name__ == "__main__":
    download_weights()