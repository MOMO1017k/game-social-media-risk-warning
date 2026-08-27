import os
import re
import sys
import glob
import pandas as pd
from tqdm.auto import tqdm
import importlib
import numpy as np
import zlib
import os
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ast

def calculate_compression_ratio(text_series):
    """
    计算 '文本压缩比' (Compression Ratio)
    原理：利用 zlib (LZ77) 算法检测重复模式。
    不管你加什么表情、标点，只要核心“长字符串”重复，压缩比就会骤降。
    
    返回：压缩后大小 / 原始大小
    范围：0.0 (极度重复) ~ 1.0 (完全随机)
    预期：正常对话约 0.4~0.6；刷屏会掉到 0.1 以下。
    """
    # 1. 简单拼接
    # 不需要复杂的清洗，直接把所有评论拼成一个巨大的字符串
    # 甚至保留标点更有利于检测，因为刷屏往往连标点都复制
    valid_texts = [str(t) for t in text_series if str(t).strip()]
    full_text = "\n".join(valid_texts)
    
    if not full_text: return 1.0
    
    # 转换为字节
    original_bytes = full_text.encode('utf-8')
    original_size = len(original_bytes)
    
    if original_size < 500: # 字节数太少，压缩没意义
        return 1.0
        
    # 2. 核心：进行压缩
    compressed_bytes = zlib.compress(original_bytes)
    compressed_size = len(compressed_bytes)
    
    # 3. 计算比率
    ratio = compressed_size / original_size
    return min(1.0, ratio)

def calculate_weighted_intensity(text_series):
    """
    V7 升级版：情绪烈度加权计算
    逻辑：
    1. 统计每条评论中敏感符号的个数。
    2. 设置“饱和阈值” (CAP = 5)。超过5个按5个算，防止单个疯子刷屏破坏数据。
    3. 返回该时间段内的总烈度分。
    """
    if text_series.empty: return 0
    
    # 转换为字符串
    text_series = text_series.astype(str)
    
    # 定义符号正则
    pattern = r'[❗!！\?？🕯️]'
    
    # 内部函数：计算单条文本的得分
    def get_score(text):
        # 找出所有匹配的符号
        matches = re.findall(pattern, text)
        count = len(matches)
        # 截断逻辑：最大只给 5 分 (防止一个评论带100个感叹号)
        return min(count, 5)
    
    # 应用到 Series 并求和
    total_intensity = text_series.apply(get_score).sum()
    
    return total_intensity

# def calculate_gini(user_series):
#     """
#     计算用户发言的基尼系数 (Gini Coefficient)
#     输入: user_id 的 Series (在该时间窗口内的所有发言用户ID)
#     输出: 0.0 ~ 1.0
#     逻辑:
#       - 如果没人发言，返回 NaN
#       - 如果只有1个人发言，Gini = 0 (完全平等? 其实是完全垄断，但公式在n=1时通常归0，这里我们特殊处理)
#         * 修正：对于异常检测，如果只有1个人发了100条，我们希望 Gini 很高。
#         * 但标准 Gini 是衡量分布不平等的。
#         * 让我们用标准公式，结合流量看即可。
#     """
#     if user_series.empty: return np.nan
    
#     # 统计每个用户的发言次数
#     # 例如: User A 发了 10 条, User B 发了 1 条 -> counts = [10, 1]
#     counts = user_series.value_counts().values
    
#     n = len(counts)
#     if n == 0: return np.nan
#     if n == 1: return 0.0 # 只有一个用户，不存在“不平等”（或者说无法计算分布差异）
    
#     # 排序 (从小到大)
#     counts = np.sort(counts)
    
#     # Gini 计算公式
#     # G = (2 * sum(i * y_i) / (n * sum(y_i))) - (n + 1) / n
#     # 这里的 i 是 1 到 n
#     index = np.arange(1, n + 1)
#     gini = (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n
    
#     return gini
def calculate_gini(user_series):
    """
    计算用户发言的基尼系数 (Gini Coefficient) - 鲁棒版
    
    逻辑优化:
      1. 【静音门槛】: 如果参与人数太少(如 <5人)，强制返回 0。防止低流量下的数值震荡。
      2. 【垄断检测】: 如果只有1个人，但他发了很多条(如 >10条)，Gini=1.0 (极度不平等)。
    """
    if user_series.empty: 
        return 0.0  # 返回0比NaN好，方便画图
    
    # 统计每个用户的发言次数
    counts = user_series.value_counts().values
    
    n_users = len(counts)        # 参与人数
    total_posts = np.sum(counts) # 总发帖量
    
    # -------------------------------------------------------------
    # 1. 核心修改：低流量静音 (Silence Noise)
    # 解决右图“钉子”问题的关键。
    # 只有当至少有 5 个不同用户参与，或总帖子数超过 10 条时，才计算分布。
    # -------------------------------------------------------------
    if n_users < 5 and total_posts < 10:
        return 0.0
        
    # -------------------------------------------------------------
    # 2. 核心修改：单人垄断修正 (Monopoly Fix)
    # 解决您注释中的痛点："如果只有1个人发了100条"
    # -------------------------------------------------------------
    if n_users == 1:
        # 如果只有1个人，但他刷屏了(>5条)，视为最大不平等 -> 1.0
        if total_posts > 5:
            return 1.0 
        else:
            return 0.0 # 只是1个人发了1-5条，视为正常闲聊

    # 3. 标准 Gini 计算
    counts = np.sort(counts)
    index = np.arange(1, n_users + 1)
    
    # 公式：G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    gini = (2 * np.sum(index * counts)) / (n_users * total_posts) - (n_users + 1) / n_users
    
    # 4. (可选) 归一化限制
    # 理论上 Gini 范围是 [0, 1 - 1/n]。为了让不同人数的窗口可比，可以除以理论最大值
    # 但在舆情监控中，通常直接用原始 Gini 即可，这里做个兜底防止浮点误差
    return max(0.0, min(1.0, gini))

# def calculate_length_std(text_series):
#     """
#     计算文本长度的标准差 (Standard Deviation)
#     输入: 评论内容 Series
#     输出: 长度的标准差 (数值越大，长短差异越明显；数值越小，越像复制粘贴)
#     """
#     if text_series.empty: return np.nan
    
#     # 1. 计算每条评论的长度
#     # 转换为字符串，处理空值
#     lengths = text_series.astype(str).apply(len)
    
#     # 2. 如果样本太少（比如只有1-2条），标准差没有意义，返回 NaN
#     if len(lengths) < 3: return np.nan
    
#     # 3. 计算标准差
#     return lengths.std()

def categorize_length(text):
    """
    将文本长度分类，用于计算舆论主导：
    0: 短文本 (Short) < 10字 (宣泄/跟风)
    1: 中文本 (Medium) 10-50字 (普通讨论)
    2: 长文本 (Long) > 50字 (小作文/深度维权)
    """
    if pd.isna(text): return None
    l = len(str(text))
    if l < 10: return 'Short'
    elif l > 50: return 'Long'
    else: return 'Medium'

# 计算负面占比（设定阈值 < 0.4 为负面）
def calc_neg_ratio(x):
    if len(x) == 0: return 0
    return ((x < 0.4).sum()+15*0.15) /(len(x)+15) # 贝叶斯平滑 加入信任阻尼，假设未统计前有15条评论，日常负面比率为15%

class VisualImpactDetector:
    """视觉冲击检测器 - 精简版"""
    
    def __init__(self, image_base_dir,
                 cache_path: str = "./phash_cache.pkl",
                 hamming_threshold: int = 10):
        self.image_base_dir = Path(image_base_dir)
        self.cache_path = cache_path
        self.hamming_threshold = hamming_threshold
        self.hash_cache = self._load_cache()
        self.DAMPING_K = 15 # 阻尼系数，防止低流量下集中度过高
    
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.hash_cache, f)
    
    def parse_urls(self, field) -> list:
        if pd.isna(field) or field == '':
            return []
        return str(field).split(', ')
    
    def url_to_path(self, url: str) -> str | None:
        if not url or pd.isna(url):
            return None
        url = str(url).strip()
        if not url:
            return None
        filename = hashlib.md5(url.encode('utf-8')).hexdigest() + '.jpg'
        path = self.image_base_dir / filename[:2] / filename
        return str(path) if path.exists() else None
    
    def compute_phash(self, path: str) -> str | None:
        if path in self.hash_cache:
            return self.hash_cache[path]
        try:
            img = Image.open(path).convert('RGB')
            h = str(imagehash.phash(img, hash_size=16))
            self.hash_cache[path] = h
            return h
        except:
            return None
    
    def attach_hashes(self, df: pd.DataFrame,
                      image_col: str = 'image_urls',
                      video_col: str = 'video_cover_url',
                      time_col: str = 'timestamp',
                      out_hash_col: str = 'img_phashes',
                      out_path_col: str = 'img_paths') -> pd.DataFrame:
        """
        返回一个 df 副本，新增：
          - img_paths: list[str]  每条发言对应的图片本地路径列表
          - img_phashes: list[str] 每条发言对应的pHash列表（与paths同序，过滤None）
          - has_img: int(0/1)
          - img_count: int（hash数量）
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        def get_paths(row):
            paths = []
            for url in self.parse_urls(row.get(image_col, '')):
                p = self.url_to_path(url)
                if p:
                    paths.append(p)
            if video_col in row:
                p = self.url_to_path(row.get(video_col, ''))
                if p:
                    paths.append(p)
            return paths

        df[out_path_col] = df.apply(get_paths, axis=1)

        # 先把所有path收集出来，批量算hash（只算cache里没有的）
        all_paths = set(p for ps in df[out_path_col] for p in ps)
        new_paths = [p for p in all_paths if p not in self.hash_cache]

        if new_paths:
            for p in tqdm(new_paths, desc="计算pHash"):
                self.compute_phash(p)
            self.save_cache()

        def paths_to_hashes(paths):
            hs = []
            for p in paths:
                h = self.hash_cache.get(p)
                if h:
                    hs.append(h)
            return hs

        df[out_hash_col] = df[out_path_col].apply(paths_to_hashes)
        df['has_img'] = df[out_hash_col].apply(lambda x: int(len(x) > 0))
        df['img_count'] = df[out_hash_col].apply(len)

        return df
    
    # def cluster_images(self, hash_dict: dict) -> list:
        # if not hash_dict:
            # return []
        # paths = list(hash_dict.keys())
        # n = len(paths)

    def cluster_hashes(self, hashes: list[str]) -> list[list[int]]:
        """Union-Find聚类，返回按大小降序的簇"""
        n = len(hashes)
        if n == 0:
            return []

        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        
        # hashes = [hash_dict[p] for p in paths]

        # for i in range(n):
        #     for j in range(i + 1, n):
        #         try:
        #             if imagehash.hex_to_hash(hashes[i]) - imagehash.hex_to_hash(hashes[j]) <= self.hamming_threshold:
        #                 union(i, j)
        #         except:
        #             pass
        
        # 预先把hex转成hash对象，避免重复转换
        hh = []
        for h in hashes:
            try:
                hh.append(imagehash.hex_to_hash(h))
            except:
                hh.append(None)
        for i in range(n):
            if hh[i] is None:
                continue
            for j in range(i + 1, n):
                if hh[j] is None:
                    continue
                if (hh[i] - hh[j]) <= self.hamming_threshold:
                    union(i, j)
        clusters = defaultdict(list)
        for i in range(n):
            # clusters[find(i)].append(paths[i])
            clusters[find(i)].append(i)

        return sorted(clusters.values(), key=len, reverse=True)
        

        
    # def calc_metrics(self, hash_dict: dict) -> dict:
        """计算两个核心指标"""
        # total = len(hash_dict)
    def calc_metrics_from_hashes(self, hashes: list[str]) -> dict: #改为使用列表
        total = len(hashes)
           
        if total == 0:
            return {
                'img_count': 0, 
                'unique_groups': 0, 
                'top3_sum': 0,
                'vis_abs_redundancy': 0, 
                'vis_concentration': 0}
        
        # clusters = self.cluster_images(hash_dict)
        clusters = self.cluster_hashes(hashes)
        unique = len(clusters)
        sizes = [len(c) for c in clusters]
        top3_sum = sum(sizes[:3])
        
        return {
            'img_count': total,
            'unique_groups': unique,
            'top3_sum': top3_sum,
            # 核心特征1: 绝对冗余量
            'vis_abs_redundancy': total - unique,
            # 核心特征2: 视觉集中度 (阻尼比例)
            'vis_concentration': top3_sum / (total + self.DAMPING_K)
        }
    # -------------------------
    # 阶段B：只聚合（不再算hash）
    # -------------------------
    def aggregate(self, df_with_hashes: pd.DataFrame,
                  start_date: str | None = None,
                  end_date: str | None = None,
                  time_col: str = 'timestamp',
                  hash_col: str = 'img_phashes',
                  freq: str = '15T') -> pd.DataFrame:
        df = df_with_hashes.copy()
        df['_dt'] = pd.to_datetime(df[time_col], errors='coerce')

        if start_date is not None:
            df = df[df['_dt'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['_dt'] <= pd.to_datetime(end_date)]

        if df.empty:
            return pd.DataFrame()

        df = df.set_index('_dt')
        results = []

        for t, g in df.groupby(pd.Grouper(freq=freq)):
            if g.empty:
                continue

            # flatten：把窗口内每条发言的hash list拼起来
            hashes = []
            for hs in g[hash_col]:
                if hs is None or (isinstance(hs, float) and pd.isna(hs)):
                    continue
                if isinstance(hs, str):
                    try:
                        hs = ast.literal_eval(hs)   # "['a','b']" -> ['a','b']
                    except:
                        hs = []
                if isinstance(hs, (list, tuple)) and hs:
                    hashes.extend(list(hs))

            m = self.calc_metrics_from_hashes(hashes)
            results.append({
                'time': t,
                'total_volume': len(g),
                'with_img_count': int((g.get('has_img', pd.Series([0]*len(g))).sum())),
                **m
            })

        return pd.DataFrame(results)
    
    def analyze(self, df: pd.DataFrame, start_date: str, end_date: str,
                image_col: str = 'image_urls', video_col: str = 'video_cover_url',
                time_col: str = 'timestamp', freq: str = '15T') -> pd.DataFrame:
        
        print(f"🔍 分析: {start_date} ~ {end_date}")
        
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df[(df['_dt'] >= start_date) & (df['_dt'] <= end_date)]
        
        if df.empty:
            print("⚠️ 无数据")
            return pd.DataFrame()
        
        # 解析路径
        print("📁 解析图片...")
        def get_paths(row):
            paths = []
            for url in self.parse_urls(row.get(image_col, '')):
                p = self.url_to_path(url)
                if p: paths.append(p)
            if video_col in row:
                p = self.url_to_path(row[video_col])
                if p: paths.append(p)
            return paths
        
        df['_paths'] = df.apply(get_paths, axis=1)
        df['_has_img'] = df['_paths'].apply(bool).astype(int)
        
        total_found = sum(len(p) for p in df['_paths'])
        print(f"   找到 {total_found} 张图片")
        
        if total_found == 0:
            return pd.DataFrame()
        
        # 计算哈希
        print("🔢 计算pHash...")
        all_paths = set(p for ps in df['_paths'] for p in ps)
        new_paths = [p for p in all_paths if p not in self.hash_cache]
        if new_paths:
            for p in tqdm(new_paths, desc="哈希"):
                self.compute_phash(p)
            self.save_cache()
        
        # 按窗口统计
        print(f"📊 按 {freq} 窗口分析...")
        df.set_index('_dt', inplace=True)
        results = []
        
        for t, g in tqdm(list(df.groupby(pd.Grouper(freq=freq))), desc="分析"):
            if g.empty: continue
            
            paths = [p for ps in g['_paths'] for p in ps]
            hashes = {p: self.hash_cache[p] for p in paths if p in self.hash_cache}
            m = self.calc_metrics(hashes)
            
            results.append({
                'time': t,
                'total_volume': len(g),
                'with_img_count': int(g['_has_img'].sum()),
                **m
            })
        
        result_df = pd.DataFrame(results)
        return result_df
        # self._summary(result_df)
    
    def _summary(self, df):
        if df.empty: return
        print(f"\n{'='*50}")
        print(f"📋 窗口: {len(df)} | 帖子: {df['total_volume'].sum()} | 图片: {df['img_count'].sum()}")
        print(f"   最大绝对冗余: {df['vis_abs_redundancy'].max()}")
        print(f"   最高集中度: {df['vis_concentration'].max():.3f}")
        print('='*50)



