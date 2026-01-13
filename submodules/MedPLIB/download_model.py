import os

# 1. 设置 Hugging Face 镜像源（国内加速下载关键）
# 必须在导入 huggingface_hub 之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

def download_hf_model(repo_id, local_dir):
    print(f"🚀 开始下载: {repo_id}")
    print(f"📂 保存路径: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # False 表示直接下载文件，不使用缓存软链接（推荐）
            resume_download=True,          # 开启断点续传
            max_workers=8,                 # 多线程并发数，可根据网络情况调整
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # (可选) 忽略不需要的文件格式以节省空间
        )
        print(f"✅ {repo_id} 下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 目标模型 1: LLaVA-v1.5-7b (MedPLIB 的基础模型)
    model_name = "liuhaotian/llava-v1.5-7b"
    save_path = "/mnt/disk4t0/publicData/huggingface_models/llava-v1.5-7b"
    download_hf_model(model_name, save_path)

    # 目标模型 2: CLIP Vision Tower (MedPLIB 训练通常也需要这个)
    # clip_name = "openai/clip-vit-large-patch14-336"
    # clip_path = "./huggingface_models/clip-vit-large-patch14-336"
    # download_hf_model(clip_name, clip_path)