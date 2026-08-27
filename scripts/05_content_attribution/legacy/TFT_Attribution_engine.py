
import pandas as pd
import numpy as np
import ast
import os
import pickle
import hashlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import imagehash  # 需要 pip install imagehash
from TFT_topic_modeling import TopicModeler 
import re


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


from matplotlib.font_manager import FontProperties # [关键] 引入字体管理
# ==========================================
# 1. 视觉冲击检测器 (你的代码集成)
# ==========================================
class VisualImpactDetector:
    def __init__(self, image_base_dir, cache_path="./phash_cache.pkl", hamming_threshold=10):
        self.image_base_dir = Path(image_base_dir)
        self.cache_path = cache_path
        self.hamming_threshold = hamming_threshold
        self.hash_cache = self._load_cache()
        self.DAMPING_K = 15

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except: pass
        return {}

    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.hash_cache, f)

    def parse_urls(self, field) -> list:
        if pd.isna(field) or field == '': return []
        return str(field).split(', ')

    def url_to_path(self, url: str) -> str | None:
        """根据URL找本地文件: hash(url).jpg"""
        if not url or pd.isna(url): return None
        url = str(url).strip()
        if not url: return None
        
        # 逻辑：取 MD5
        filename = hashlib.md5(url.encode('utf-8')).hexdigest() + '.jpg'
        # 逻辑：假设结构是 base/ab/abcdef....jpg
        # path = self.image_base_dir / filename[:2] / filename
        
        sub_folder = filename[:2]
        path = self.image_base_dir / sub_folder / filename
        
        if path.exists():
            return str(path)
        return None
    
        # 如果找不到，尝试不分文件夹直接找 (兼容性)
        # if not path.exists():
        #     path_flat = self.image_base_dir / filename
        #     if path_flat.exists(): return str(path_flat)
        #     return None
            
        return str(path)

    def compute_phash(self, path: str) -> str | None:
        if path in self.hash_cache: return self.hash_cache[path]
        try:
            img = Image.open(path).convert('RGB')
            h = str(imagehash.phash(img, hash_size=16))
            self.hash_cache[path] = h
            return h
        except: return None

    def analyze_window(self, df_window, image_col='image_urls'):
        """分析窗口内的图片，返回核心指标和 Top 重复图"""
        # 1. 提取所有本地路径
        all_paths = []
        for urls in df_window[image_col]:
            for url in self.parse_urls(urls):
                p = self.url_to_path(url)
                if p: all_paths.append(p)
        
        if not all_paths:
            return {"status": "no_local_images", "vis_concentration": 0}

        # 2. 计算 Hash
        path_hash_map = {}
        for p in all_paths:
            if p not in self.hash_cache:
                self.compute_phash(p)
            if p in self.hash_cache:
                path_hash_map[p] = self.hash_cache[p]
        self.save_cache() # 及时保存
        
        valid_paths = list(path_hash_map.keys())
        valid_hashes = list(path_hash_map.values())
        
        if not valid_hashes:
            return {"status": "hash_failed", "vis_concentration": 0}

        # 3. 聚类 (Union-Find)
        clusters = self._cluster_hashes_with_paths(valid_paths, valid_hashes)
        
        # 4. 计算指标
        total = len(all_paths) # 注意分母是总图数
        top3_sum = sum([len(c) for c in clusters[:3]])
        concentration = top3_sum / (total + self.DAMPING_K)
        
        # 5. 提取 Top 3 代表性图片 (取簇里第一张存在的图)
        top_images = []
        for clus in clusters[:3]:
            # clus 是 path 的列表
            top_images.append({
                "path": clus[0], # 取第一张作为代表
                "count": len(clus),
                "hash": path_hash_map[clus[0]]
            })

        return {
            "status": "success",
            "total_images": total,
            "vis_concentration": concentration,
            "unique_groups": len(clusters),
            "top_repeated_images": top_images # 包含路径，用于画图
        }

    def _cluster_hashes_with_paths(self, paths, hashes):
        """聚类，返回 [[path1, path2], [path3], ...]"""
        n = len(hashes)
        parent = list(range(n))
        
        # 预转换
        hash_objs = []
        for h in hashes:
            try: hash_objs.append(imagehash.hex_to_hash(h))
            except: hash_objs.append(None)

        def find(x):
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX, rootY = find(x), find(y)
            if rootX != rootY: parent[rootY] = rootX

        # 两两比较 (O(N^2)，窗口内通常 N 较小)
        for i in range(n):
            if hash_objs[i] is None: continue
            for j in range(i + 1, n):
                if hash_objs[j] is None: continue
                if (hash_objs[i] - hash_objs[j]) <= self.hamming_threshold:
                    union(i, j)
        
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(paths[i])
            
        return sorted(clusters.values(), key=len, reverse=True)

# ==========================================
# 2. 仪表盘生成器
# ==========================================
class DashboardGenerator:
    @staticmethod
    def create_report(text_summary, visual_summary, save_path, title_info):
        """生成拼图报告"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig)
        
        # --- 标题 ---
        fig.suptitle(f"Anomaly Attribution Report: {title_info}", fontsize=20, fontweight='bold')

        # --- A. 词云 (Top Left) ---
        ax_wc = fig.add_subplot(gs[0, 0:2])
        if text_summary and 'visualizations' in text_summary:
            wc_path = text_summary['visualizations'].get('wordcloud')
            if wc_path and os.path.exists(wc_path):
                ax_wc.imshow(Image.open(wc_path))
            else:
                ax_wc.text(0.5, 0.5, "No WordCloud", ha='center')
        else:
            ax_wc.text(0.5, 0.5, "Text Analysis Skipped", ha='center')
        ax_wc.set_title("Semantic Focus (WordCloud)", fontsize=14)
        ax_wc.axis('off')

        # --- B. 聚类图 (Top Right) ---
        ax_clus = fig.add_subplot(gs[0, 2:4])
        if text_summary and 'visualizations' in text_summary:
            clus_path = text_summary['visualizations'].get('cluster_map')
            if clus_path and os.path.exists(clus_path):
                ax_clus.imshow(Image.open(clus_path))
            else:
                ax_clus.text(0.5, 0.5, "No Cluster Map", ha='center')
        ax_clus.set_title("Topic Clusters", fontsize=14)
        ax_clus.axis('off')

        # --- C. 视觉重复图 (Bottom Row) ---
        # 显示 Top 3 图片
        top_imgs = visual_summary.get('top_repeated_images', []) if visual_summary else []
        
        for i in range(3):
            ax_img = fig.add_subplot(gs[1, i]) # 占据左下角 3 个位置
            if i < len(top_imgs):
                img_data = top_imgs[i]
                try:
                    img = Image.open(img_data['path'])
                    ax_img.imshow(img)
                    ax_img.set_title(f"Rank {i+1} (Count: {img_data['count']})", color='red')
                except:
                    ax_img.text(0.5, 0.5, "Image Load Failed", ha='center')
            else:
                ax_img.text(0.5, 0.5, "No Image", ha='center', color='gray')
            ax_img.axis('off')

        # --- D. 文字摘要 (Bottom Right) ---
        ax_text = fig.add_subplot(gs[1, 3])
        ax_text.axis('off')
        
        report_str = "情况总结\n"
        report_str += "-"*25 + "\n"
        
        if text_summary:
            report_str += f"Text Samples: {text_summary.get('total_samples', 0)}\n"
            # 提取前 3 个关键词
            kw = []
            if 'discovered_clusters' in text_summary:
                 for cid, cinfo in list(text_summary['discovered_clusters'].items())[:2]:
                     kw.extend(cinfo.get('keywords', [])[:3])
            report_str += f"Keywords: {', '.join(kw[:5])}\n\n"
        
        if visual_summary:
            report_str += f"Visual Concentration: {visual_summary.get('vis_concentration', 0):.2%}\n"
            report_str += f"Total Images: {visual_summary.get('total_images', 0)}\n"
            report_str += f"Unique Groups: {visual_summary.get('unique_groups', 0)}\n"
        
        ax_text.text(0.05, 0.95, report_str, fontsize=12, va='top', family='monospace')

        # 保存
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close()
        return save_path

# ==========================================
# 3. 归因引擎主类
# ==========================================
class ContentAttributionEngine:
    def __init__(self,vsn_feature_maps=None):
        # [配置] 请修改为你的真实图片路径
        self.IMAGE_BASE_DIR =r"C:\tongji\0 code\00_data\weibo_images"
        
        self.topic_modeler = None
        self.vis_detector = VisualImpactDetector(self.IMAGE_BASE_DIR)
        
        self.vsn_feature_maps = vsn_feature_maps if vsn_feature_maps is not None else {}
        # self.vsn_feature_map = [
        #     'event_code', 'hour', 'dayofweek', 'month', 'day', 'is_weekend', 'is_holiday', 'time_slot', 
        #     'relative_time_idx', 'total_volume_post', 'total_volume_comment', 'gini_post', 'gini_comment', 
        #     'senti_symbol_post', 'senti_symbol_comment', 'comp_ratio_post', 'comp_ratio_comment', 
        #     'total_short_post', 'total_long_post', 'total_short_comment', 'total_long_comment', 
        #     'neg_ratio_post', 'neg_ratio_comment', 'semantic_shift_post', 'semantic_shift_comment', 
        #     'retweet_ratio_post', 'vis_abs_redundancy_post', 'vis_concentration_post'
        # ]
        self.OFFICIAL_AUTHORS = {'无限暖暖', '无限暖暖搬砖工', '无限暖暖小助手美鸭梨'}

    def _get_topic_modeler(self):
        if self.topic_modeler is None:
            self.topic_modeler = TopicModeler()
        return self.topic_modeler

    def _decode_vsn(self, row_data, model_name):
        vsn_idx_col = f"{model_name}_vsn_idx_0"
        if vsn_idx_col in row_data:
            idx = int(row_data[vsn_idx_col])
            # if 0 <= idx < len(self.vsn_feature_map):
            #     return self.vsn_feature_map[idx]

            # --- 核心修改：从字典中按 model_name 获取专属映射表 ---
            if model_name in self.vsn_feature_maps:
                vsn_list = self.vsn_feature_maps[model_name]
                if 0 <= idx < len(vsn_list):
                    return vsn_list[idx]
                
            return f"Unknown_Index_{idx}"
        return "VSN_Info_Missing"
    
    def _decode_feature_name(self, lgbm_feature_name):
        """
        将 LightGBM 特征名 (如 'model_vol_burst_vsn_12') 解析为真实业务指标名
        """
        # --- 1. 解析 VSN 特征 (例如: model_spam_detect_vsn_12) ---
        if '_vsn_' in lgbm_feature_name:
            parts = lgbm_feature_name.split('_vsn_')
            model_name = parts[0]
            
            try:
                # 提取数字索引
                vsn_idx = int(parts[1].split('_')[0]) 
                
                # 从字典中查找该模型对应的真实变量名
                if model_name in self.vsn_feature_maps:
                    vsn_list = self.vsn_feature_maps[model_name]
                    if vsn_idx < len(vsn_list):
                        return vsn_list[vsn_idx]
                        
                return f"Unknown_VSN_{model_name}_{vsn_idx}"
            except ValueError:
                pass # 如果解析数字失败，走到最后直接返回原名
                
        # --- 2. 解析 Attention 特征 (例如: model_sentiment_shift_att_24) ---
        elif '_att_' in lgbm_feature_name:
            parts = lgbm_feature_name.split('_att_')
            model_name = parts[0]
            
            # 处理通过 Top-K 提取的注意力特征 (例如: model_att_val_0)
            if 'val_' in parts[1]:
                rank = parts[1].replace('val_', '')
                return f"Attention_{model_name}_Top{rank}"
            
            # 处理铺平的注意力特征 (例如: model_att_24)
            try:
                time_steps_ago = int(parts[1].split('_')[0])
                return f"Attention_{model_name}(t-{time_steps_ago})"
            except ValueError:
                pass

        # --- 3. 解析标量特征 (如 residual, divergence, pred_p50) ---
        elif '_resid' in lgbm_feature_name:
            return f"Residual_{lgbm_feature_name.split('_resid')[0]}"
        elif '_div' in lgbm_feature_name:
            return f"Divergence_{lgbm_feature_name.split('_div')[0]}"

        # --- 4. 兜底返回 ---
        return lgbm_feature_name

    def route_and_explain(self, df_raw, row_data, top_feature_name):
        """
        :param row_data: Series, 必须包含 'timestamp' 和 'pred_label' (AP/CP)
        """
        current_time = pd.to_datetime(row_data['timestamp'])
        pred_label = row_data.get('pred_label', 'Unknown')
        
        model_prefix = top_feature_name.split('_att_')[0] if '_att_' in top_feature_name else top_feature_name.split('_vsn_')[0]
        
        # 20260220新增 - 解码特征名
        decoded_feature = self._decode_feature_name(top_feature_name)
        # --- 1. 确定 Focus Window ---
        time_offset = 0
        lag_info = "Current Time"
        
        if '_att_val_' in top_feature_name:
            rank_idx = top_feature_name.split('_att_val_')[-1]
            lag_col = f"{model_prefix}_att_lag_{rank_idx}"
            if lag_col in row_data:
                lag_steps = int(row_data[lag_col])
                time_offset = lag_steps * 15
                lag_info = f"Lag {lag_steps} ({time_offset} min)"
            
        focus_time = current_time + pd.Timedelta(minutes=time_offset)
        mask = (df_raw['timestamp'] >= focus_time) & (df_raw['timestamp'] < focus_time + pd.Timedelta(minutes=15))
        df_window = df_raw.loc[mask]
        
        dominant_var = self._decode_vsn(row_data, model_prefix)
        
        explanation = {
            "type": "comprehensive",
            "current_time": str(current_time),
            "focus_time": str(focus_time),
            "label": pred_label,
            "time_logic": lag_info,
            "trigger_feature": top_feature_name,
            "trigger_feature_decoded": decoded_feature,
            "dominant_vsn_variable": dominant_var,
            "sample_count": len(df_window)
        }

        if df_window.empty:
            explanation["status"] = "no_raw_data_in_window"
            return explanation

        # --- 2. 文本分析 ---
        window_id = focus_time.strftime("%Y%m%d%H%M")
        title_str = f"{focus_time} [{pred_label}]" # 传给词云的标题
        
        explanation["text_analysis"] = self.analyze_text(df_window, window_id, title_str)
        
        # --- 3. 视觉分析 (使用新 Detector) ---
        explanation["visual_analysis"] = self.vis_detector.analyze_window(df_window, image_col='image_urls')
        
        # --- 4. 生成大图报告 ---
        output_dir = f"./attribution_results/{window_id}"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        dashboard_path = f"{output_dir}/dashboard_report.png"
        DashboardGenerator.create_report(
            explanation.get("text_analysis"),
            explanation.get("visual_analysis"),
            dashboard_path,
            title_str
        )
        explanation["dashboard_path"] = dashboard_path

        return explanation

    def filter_noise_posts(self, df_window):
        """
        【新增】过滤噪声帖子：
        1. 过滤含图片/视频的帖子 (has_img > 0 或 img_count > 0)
        2. 过滤官方账号发帖
        返回过滤后的 DataFrame
        """
        df = df_window.copy()
        original_len = len(df)
        
        # 过滤条件 1：排除含图片的帖子
        if 'has_img' in df.columns:
            df = df[~(df['has_img'] > 0)]
        elif 'img_count' in df.columns:
            df = df[~(df['img_count'] > 0)]
        
        # 过滤条件 2：排除含视频链接的帖子
        if 'video_url' in df.columns:
            df = df[df['video_url'].isna() | (df['video_url'] == '')]
        
        # 过滤条件 3：排除官方账号
        if 'author_name' in df.columns:
            df = df[~df['author_name'].isin(self.OFFICIAL_AUTHORS)]
        elif 'user_name' in df.columns:
            df = df[~df['user_name'].isin(self.OFFICIAL_AUTHORS)]
        
        filtered_len = len(df)
        if original_len != filtered_len:
            print(f"[Filter] 过滤噪声帖子: {original_len} → {filtered_len} "
                  f"(移除 {original_len - filtered_len} 条含图/视频/官方帖)")
        
        return df

    def extract_top_images(self, df_window, top_k=2, image_col='image_urls'):
        """
        【新增】从窗口数据中提取 Top-K 重复图片的本地路径，
        供看板展示使用。
        """
        result = self.vis_detector.analyze_window(df_window, image_col=image_col)
        
        if result.get('status') != 'success':
            # 如果没有本地图片，尝试通过 img_phashes 列做统计
            return self._fallback_extract_images(df_window, top_k)
        
        top_images = result.get('top_repeated_images', [])[:top_k]
        return top_images

    def _fallback_extract_images(self, df_window, top_k=2):
        """
        【新增】降级方案：当本地无图片文件时，
        从 img_phashes 列统计 Top-K hash 并返回（无实际图片路径）。
        """
        import ast
        all_hashes = []
        
        phash_col = 'img_phashes' if 'img_phashes' in df_window.columns else None
        if phash_col is None:
            return []
        
        for val in df_window[phash_col].dropna():
            if isinstance(val, list):
                all_hashes.extend([str(h) for h in val])
            elif isinstance(val, str) and val.strip():
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        all_hashes.extend([str(h) for h in parsed])
                except:
                    pass
        
        if not all_hashes:
            return []
        
        counter = Counter(all_hashes)
        top_items = counter.most_common(top_k)
        
        results = []
        for phash, count in top_items:
            # 尝试通过 phash 在 df 中找到对应的 image_urls
            img_url = None
            if 'image_urls' in df_window.columns:
                for _, row in df_window.iterrows():
                    row_phashes = row.get('img_phashes', '')
                    if isinstance(row_phashes, str) and phash in row_phashes:
                        urls = str(row.get('image_urls', '')).split(', ')
                        if urls and urls[0].strip():
                            # 尝试找到本地路径
                            local_path = self.vis_detector.url_to_path(urls[0].strip())
                            if local_path:
                                img_url = local_path
                                break
            
            results.append({
                "path": img_url,
                "count": count,
                "hash": phash
            })
        
        return results

    def analyze_text(self, df_window, window_id, title_info):
        modeler = self._get_topic_modeler()
        
        # 1. 决定用于建模的列 (优先使用深度清洗的列)
        model_col = 'text_clean_strict_deep' if 'text_clean_strict_deep' in df_window.columns else 'text_clean_strict'
        if model_col not in df_window.columns:
            model_col = 'text_clean'
            
        # 2. 决定用于展示的列 (使用保留了语义的轻度清洗列，便于阅读)
        display_col = 'text_clean_strict' if 'text_clean_strict' in df_window.columns else 'text'

        if model_col not in df_window.columns: return None
        texts_for_modeling = df_window[model_col].dropna()
        if texts_for_modeling.empty: return None
        
        output_dir = f"./attribution_results/{window_id}"
        summary = modeler.run_topic_modeling(texts_for_modeling, output_dir=output_dir, window_id=str(window_id), title_info=title_info)
        
        # 3. 提取、去重并筛选高质量的人类可读文本
        if summary and 'topic_indices' in summary:
            rep_texts = {}
            for topic_key, idx_list in summary['topic_indices'].items():
                if not idx_list:
                    continue
                
                raw_texts = df_window.loc[idx_list, display_col].dropna().astype(str).tolist()
                
                valid_texts = []
                seen_core = set()
                
                for t in raw_texts:
                    t_disp = t.replace('\n', ' ').strip()
                    t_disp = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？、；：""''（）【】《》\.,!\?]', '', t_disp)
                    
                    if len(t_disp.strip()) > 8:
                        core_text = re.sub(r'[^\w\u4e00-\u9fa5]', '', t_disp)
                        
                        if core_text not in seen_core:
                            seen_core.add(core_text)
                            valid_texts.append(t_disp)
                
                if not valid_texts:
                    for t in raw_texts:
                        t_disp = t.replace('\n', ' ').strip()
                        t_disp = re.sub(r'[^\w\s\u4e00-\u9fa5，。！？、；：""''（）【】《》\.,!\?]', '', t_disp)
                        if len(t_disp.strip()) > 5:
                            core_text = re.sub(r'[^\w\u4e00-\u9fa5]', '', t_disp)
                            if core_text not in seen_core:
                                seen_core.add(core_text)
                                valid_texts.append(t_disp)
                
                valid_texts.sort(key=len, reverse=True)
                rep_texts[topic_key] = valid_texts[:3]
                
            summary['representative_texts'] = rep_texts
            
        return summary
        
        
