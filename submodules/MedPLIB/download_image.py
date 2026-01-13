import os
import json
import requests
import tarfile
import concurrent.futures
from tqdm import tqdm
import io

# --- 配置区域 ---
INPUT_JSONL = "/mnt/disk4t0/publicData/LLaVA-Med/llava_med_image_urls.jsonl"
SAVE_DIR = "/mnt/disk4t0/publicData/LLaVA-Med/images"
MAX_WORKERS = 16  # AWS S3 抗压能力强，可以适当调高并发
# ----------------

def download_and_extract_stream(item):
    pair_id = item['pair_id']
    # 目标保存文件名
    save_path = os.path.join(SAVE_DIR, f"{pair_id}.jpg")
    
    # 1. 断点续传检查
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
        return "skipped"

    # 2. 构造 AWS S3 镜像链接 (替换原始 FTP 链接)
    # 原始: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/83/41/PMC6149739.tar.gz
    # S3目标: https://pmc-oa-opendata.s3.amazonaws.com/oa_package/83/41/PMC6149739.tar.gz
    ftp_url = item['pmc_tar_url']
    s3_url = ftp_url.replace("https://ftp.ncbi.nlm.nih.gov/pub/pmc/", "https://pmc-oa-opendata.s3.amazonaws.com/")
    
    target_file = item['image_file_path'] # 压缩包内的目标文件路径

    try:
        # 3. 发起流式请求 (stream=True)
        with requests.get(s3_url, stream=True, timeout=20) as r:
            if r.status_code != 200:
                return "error_http"
            
            # 4. 使用 tarfile 打开网络数据流 (管道模式)
            #这种方式不需要将文件下载到本地，直接在内存/流中解压
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                for member in tar:
                    # 5. 寻找目标图片
                    if member.name == target_file:
                        # 提取文件对象
                        f = tar.extractfile(member)
                        if f:
                            with open(save_path, "wb") as out:
                                out.write(f.read())
                            return "success"
                        # 找到文件并提取后，直接退出循环和函数
                        # 此时 requests 连接会被关闭，剩余数据不再下载 -> 极大节省带宽！
                        return "success"
    except Exception as e:
        # 很多时候是网络超时，可以在外层重试
        return f"error: {str(e)}"
    
    return "not_found"

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"📖 加载任务文件: {INPUT_JSONL}")
    tasks = []
    with open(INPUT_JSONL, "r") as f:
        for line in f:
            tasks.append(json.loads(line))
            
    print(f"📦 任务总数: {len(tasks)}")
    print(f"🚀 启动流式下载 (并发数: {MAX_WORKERS})...")
    print("💡 策略: 使用 AWS S3 镜像 + 找到图片即停止下载")

    success = 0
    skipped = 0
    failed = 0
    
    # 使用线程池并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_and_extract_stream, item): item for item in tasks}
        
        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(tasks))
        for future in pbar:
            res = future.result()
            if res == "success":
                success += 1
            elif res == "skipped":
                skipped += 1
            else:
                failed += 1
                
            pbar.set_description(f"✅{success} ⏭️{skipped} ❌{failed}")

    print("\n" + "="*30)
    print(f"处理完成 Summary:")
    print(f"  成功下载: {success}")
    print(f"  本地已存: {skipped}")
    print(f"  失败/未找: {failed}")
    print("="*30)

if __name__ == "__main__":
    main()