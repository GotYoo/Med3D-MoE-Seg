import os
from modelscope.hub.snapshot_download import snapshot_download

# 1. 您的目标目录
SAVE_DIR = "/mnt/disk4t0/publicData/GMAI___SA-Med2D-20M"

# 2. 指定要下载的文件名 (支持通配符)
# 使用列表形式，精确匹配那个卡住的文件
TARGET_FILES = ['raw/SA-Med2D-16M.z04']

print(f"🚀 正在使用 ModelScope (Snapshot模式) 补全文件: {TARGET_FILES}")
print(f"📂 本地保存路径: {SAVE_DIR}")

try:
    # 使用 snapshot_download + allow_patterns 实现单文件下载
    path = snapshot_download(
        'OpenGVLab/SA-Med2D-20M', 
        repo_type='dataset',       # 明确指定是数据集
        local_dir=SAVE_DIR,        # 指定下载目录
        allow_patterns=TARGET_FILES, # 关键：只允许下载这个文件，忽略其他
        # ignore_patterns=["*.zip", "*.z01", "*.z02", "*.z03"] # 双重保险：忽略已有文件(可选)
    )
    print(f"\n✅ 成功！文件已保存到: {path}")
    print("您现在可以继续下载 z05 或者开始解压了。")
    
except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    print("提示：如果下载并未开始，请检查 TARGET_FILES 中的路径是否需要去掉 'raw/' 前缀。")