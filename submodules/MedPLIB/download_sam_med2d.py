import os

# 1. 设置国内镜像环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download

def main():
    # --- 修正后的配置 ---
    # 正确的 Hugging Face 仓库 ID
    REPO_ID = "schengal1/SAM-Med2D_model"
    
    # 需要下载的具体文件名
    FILENAME = "sam-med2d_b.pth"
    
    # 保存目录
    SAVE_DIR = "/mnt/disk4t0/publicData/huggingface_models"
    # ----------------

    print(f"🚀 开始下载: {FILENAME}")
    print(f"📦 来源仓库: {REPO_ID}")
    print(f"📂 保存目标: {SAVE_DIR}")

    try:
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=SAVE_DIR,
            local_dir_use_symlinks=False,  # 下载真实文件
            resume_download=True,          # 支持断点续传
        )
        print(f"\n✅ 下载成功！文件已保存在:\n{file_path}")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")

if __name__ == "__main__":
    main()