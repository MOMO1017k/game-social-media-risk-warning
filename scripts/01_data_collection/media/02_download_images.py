import gevent.monkey
# 必须在导入 requests 之前执行，否则 gevent 无法使其 I/O 操作非阻塞
gevent.monkey.patch_all()

import gevent.pool
import requests
import queue
import os
import sys
import time
import hashlib
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import random
random.seed(os.getpid())

# --- 配置参数 ---
MAX_CONCURRENCY = 4     # 最大并发协程数
MAX_RETRIES = 5          # 最大重试次数
OUTPUT_DIR = "imgs2" # 图片保存目录

session = requests.Session()
# cookie = requests.cookies.RequestsCookieJar()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_CONCURRENCY, pool_maxsize=MAX_CONCURRENCY)
session.mount("http://", adapter=adapter)
session.mount("https://", adapter=adapter)


# A list of diverse User-Agents
USER_AGENTS = [
    # Windows Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    # macOS Safari
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    # Linux Firefox
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    # iPhone Safari
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
    # Android Chrome
    'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36',
]

def generate_random_headers(target_url: str) -> dict:
    """
    生成一套随机且逼真的请求头。
    """
    # random_ua = random.choice(USER_AGENTS)
    
    # # 从目标 URL 提取域名作为 Host
    # try:
    #     from urllib.parse import urlparse
    #     host = urlparse(target_url).netloc
    # except Exception:
    #     host = "www.example.com" # 失败时使用默认值

    return {
        # 1. 随机 User-Agent (必选)
        'User-Agent':     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        
        # 2. 伪造来源 (绕过防盗链)
        # 'Referer': f'https://{host}/',
        
        # 3. 告诉服务器期望的响应格式
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        
        # 4. 告诉服务器支持的编码方式 (减少传输量)
        'Accept-Encoding': 'gzip, deflate, br',
        
        # 5. 告诉服务器支持的语言 (伪装地理位置)
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7',
        
        # 6. 请求保持连接 (优化性能)
        'Connection': 'keep-alive',
        
        # 7. Host (目标域名)
        'Host': "image.baidu.com",
    }

def md5(text: str) -> str:
    """
    计算给定字符串的 MD5 散列值。

    参数:
        text (str): 要计算 MD5 的输入字符串。

    返回:
        str: 字符串的十六进制 MD5 散列值。
    """
    # 1. 创建 MD5 对象
    m = hashlib.md5()
    
    # 2. 将字符串编码为 bytes 并更新 MD5 对象
    #    .encode('utf-8') 将字符串转换为 bytes
    m.update(text.encode('utf-8'))
    
    # 3. 获取十六进制散列值
    return m.hexdigest()


def download_image(url: str, retries_left: int, download_queue: queue.Queue):
    """
    下载单个图片文件，并在失败时根据重试次数重新加入队列。
    """


    md5sum = md5(url)
    filedir = os.path.join(OUTPUT_DIR, md5sum[0:2])
    
    dst_file_path = os.path.join(filedir, f"{md5sum}.jpg")
    if(os.path.exists(dst_file_path)):
        # print(f"[{gevent.getcurrent()}] ⏭️ 已存在，跳过下载: {dst_file_path}")
        return
    
    
    if md5sum.endswith('00'):
        print(f"[{gevent.getcurrent()}] 尝试下载: {url} (剩余重试次数: {retries_left})")

    gevent.sleep(0.2)

    try:
        # 使用 requests.get (此时它已被 gevent 转换为非阻塞操作)
        
        # response = requests.get(url, timeout=5, headers=generate_random_headers(url))
        response = session.get(url, timeout=5, headers=generate_random_headers(url))

        response.raise_for_status()  # 检查 HTTP 错误状态码 (4xx 或 5xx)
        # 验证内容是否是 jpg
        image = Image.open(BytesIO(response.content))
        os.makedirs(filedir, exist_ok=True)
        if image.format != 'JPEG':
            image.convert("RGB").save(dst_file_path)
        else:
            # 将内容写入文件
            with open(dst_file_path, 'wb') as f:
                f.write(response.content)
            
        # print(f"[{gevent.getcurrent()}] ✅ [{url}] 成功下载: {dst_file_path}")

    except (requests.exceptions.RequestException, UnidentifiedImageError,ValueError) as e:
        print(f"❌ 下载失败: {url}, 错误: {e.__class__.__name__}")
        session.cookies.clear()
        
        if retries_left > 0:
            # 重试次数大于 0，将任务重新加入队列
            new_retries = retries_left - 1
            print(f"🔁 重新加入队列: {url} (新剩余次数: {new_retries})")
            
            # 重新加入队列时，带上新的重试次数
            download_queue.put((url, new_retries))
            
        else:
            print(f"🛑 达到最大重试次数，放弃下载: {url}")

        



def worker_loop(pool: gevent.pool.Pool, download_queue: queue.Queue):
    """
    循环从队列中取出任务，并在协程池中调度下载任务。
    """
    while True:
        try:
            # 非阻塞地从队列中获取任务，timeout 1秒
            url, retries_left = download_queue.get(block=False, timeout=0.1)
            
            # 将下载任务提交给协程池。
            # gevent 的 Pool.spawn 启动一个协程。
            pool.spawn(download_image, url, retries_left, download_queue)
            
            # 通知队列任务完成，即使任务只是被 spawn 出来。
            # 这是为了让 download_image 协程在重试时能重新 put 进队列，
            # 并避免 .join() 卡住。
            download_queue.task_done()

            # print(f"[调度器] 📨 任务分发: {url} (剩余重试次数: {retries_left}) | 队列大小: {download_queue.qsize()} | 空闲协程数: {pool.free_count()}")

            if(download_queue.qsize() // 100 == 0):
                print(f"Remain task: {download_queue.qsize()}")

        except queue.Empty:
            # 队列为空，退出循环
            if pool.free_count() == MAX_CONCURRENCY:
                # 只有当队列为空且所有协程都已完成时，才退出
                break
            else:
                # 队列可能暂时为空，但仍有任务正在池中处理，等待一下
                gevent.sleep(1) # 暂停，避免空转占用 CPU

        except Exception as e:
            print(f"调度器发生意外错误: {e}")
            download_queue.task_done()



def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 包含一些成功的URL和一些可能会失败的URL（例如 404 或连接超时）
    all_urls = set()
    # read all_image_video_urls.txt and download all images
    with open('all_image_video_urls.txt', 'r', encoding='utf-8') as f:
        urls = f.read().splitlines()
        print(f'Total URLs: {len(urls)}')
        # print('Sample URLs:', urls[:5])
        hashes = set([md5(url) for url in urls])
        print(f'Total unique URLs by MD5: {len(hashes)}')
        assert len(hashes) == len(urls), "There are duplicate URLs!"
        # remove all url does not has jpg
        urls = [url for url in urls if ".jpg" in url.lower()]
        print(f'Total URLs after filtering does not contains .jpg: {len(urls)}')
        all_urls.update(urls)


    # 1. 初始化任务队列
    download_queue = queue.Queue()
    all_urls = list(all_urls)
    random.shuffle(all_urls)
    for url in all_urls:
        # 初始任务加入队列，重试次数设为 MAX_RETRIES
        download_queue.put((url, MAX_RETRIES)) 
    
    # 2. 初始化协程池 (限制并发数)
    pool = gevent.pool.Pool(MAX_CONCURRENCY)
    
    start_time = time.time()
    print(f"--- 开始下载，最大并发数: {MAX_CONCURRENCY} ---")

    # 3. 启动调度器
    # 注意：worker_loop 本身也是一个阻塞函数，需要使用 gevent.spawn 启动
    scheduler = gevent.spawn(worker_loop, pool, download_queue)

    # 4. 等待所有初始任务和重试任务完成
    # 阻塞主线程，直到调度器退出。
    scheduler.join()
    
    # 5. 等待所有在池中的协程完成
    pool.join()
    
    end_time = time.time()
    print("--- 所有任务完成 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()




