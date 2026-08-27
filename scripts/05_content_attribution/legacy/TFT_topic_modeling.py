import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
from scipy.sparse import csr_matrix
import jieba

# 配置中文字体 (解决Matplotlib标题乱码)
# 优先尝试黑体，如果没有则尝试微软雅黑
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 假设配置和工具都在路径下，如果缺失请调整 import
try:
    from version_config import SEED_TOPICS, NIKKI_STOPWORDS
    from utilis_preprocess import tokenize_zh_factory
except ImportError:
    # 兜底：如果没有配置文件，使用空配置防止报错
    SEED_TOPICS = {}
    NIKKI_STOPWORDS = set()
    def tokenize_zh_factory(stopwords):
        return lambda x: jieba.lcut(x)

# ==========================================
# 核心算法类: Class-based TF-IDF
# ==========================================
class ClassTFIDF:
    """
    实现 BERTopic 的核心算法：基于类别的 TF-IDF
    用于从聚类结果中提取最具代表性的关键词
    """
    def __init__(self, stopwords):
        self.vectorizer = None
        self.ctfidf_matrix = None
        self.words = None
        self.stopwords = stopwords

    def fit_transform(self, documents_per_class):
        # [修改点 1] 动态调整 min_df
        # 如果只有极少量文档（比如只有1个类，且该类只有1条合并文本），min_df 必须为 1
        n_samples = len(documents_per_class)
        min_df = 2 if n_samples >= 5 else 1
        
        try:
            # 1. 计算词频
            self.vectorizer = CountVectorizer(
                tokenizer=tokenize_zh_factory(set(self.stopwords)), 
                min_df=min_df,
                # [修改点 2] 允许单个字符 (默认是 (?u)\b\w\w+\b 即2个字符以上)
                token_pattern=r"(?u)\b\w+\b" 
            )
            X = self.vectorizer.fit_transform(documents_per_class)
            self.words = self.vectorizer.get_feature_names_out()
            
            # 2. 计算 c-TF-IDF
            X = csr_matrix(X)
            tf_t = X.sum(axis=0).A1
            
            # 防止除零
            if X.shape[0] == 0 or X.shape[1] == 0:
                return None

            avg_words_per_class = X.sum(axis=1).mean()
            
            # 公式: W_{t,c} = tf_{t,c} * log(1 + A / tf_t)
            idf = np.log(1 + (avg_words_per_class / (tf_t + 1)))
            self.ctfidf_matrix = X.multiply(idf)
            
            # 归一化
            self.ctfidf_matrix = normalize(self.ctfidf_matrix, norm='l1', axis=1)
            return self.ctfidf_matrix
            
        except ValueError as e:
            # [修改点 3] 捕获 "After pruning, no terms remain"
            print(f"[ClassTFIDF] Warning: Feature extraction failed ({e}). Returning None.")
            self.ctfidf_matrix = None
            return None

    # def get_top_words(self, cluster_id, top_n=10):
    #     if self.ctfidf_matrix is None:
    #         return []
    #     if cluster_id >= self.ctfidf_matrix.shape[0]:
    #         return []
            
    #     try:
    #         row = self.ctfidf_matrix.getrow(cluster_id).toarray().flatten()
    #         # 如果某一行全为0，说明提取失败
    #         if row.sum() == 0:
    #             return []
    #         top_indices = row.argsort()[-top_n:][::-1]
    #         return [self.words[idx] for idx in top_indices if row[idx] > 0]
    #     except Exception:
    #         return []

    def get_top_words(self, cluster_id, top_n=10):
        if self.ctfidf_matrix is None:
            return []
        if cluster_id >= self.ctfidf_matrix.shape[0]:
            return []
            
        try:
            row = self.ctfidf_matrix.getrow(cluster_id).toarray().flatten()
            if row.sum() == 0:
                return []
            top_indices = row.argsort()[-top_n:][::-1]
            # 【修改】增加最低得分阈值，过滤掉得分极低的噪声词
            max_score = row[top_indices[0]] if len(top_indices) > 0 else 0
            min_score_threshold = max_score * 0.1  # 得分低于最高分10%的词直接过滤
            return [self.words[idx] for idx in top_indices 
                    if row[idx] > 0 and row[idx] >= min_score_threshold]
        except Exception:
            return []

# ==========================================
# 业务逻辑类: TopicModeler
# ==========================================
class TopicModeler:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.stopwords = NIKKI_STOPWORDS
        self.seed_topics = SEED_TOPICS
        
        # 字体路径 (用于词云和matplotlib)
        self.font_path = "C:/Windows/Fonts/simhei.ttf" 
        if not os.path.exists(self.font_path):
            self.font_path = None 

    def _load_model(self):
        """懒加载模型，避免初始化时占用内存"""
        if self.embedding_model is None:
            print(f"[TopicModeler] Loading SBERT model: {self.model_name}...")
            self.embedding_model = SentenceTransformer(self.model_name)

    def _match_seed_topics(self, texts, embeddings, threshold=0.4):
        """利用预设主题 (SEED_TOPICS) 对文本进行初步分类"""
        if not self.seed_topics:
            return ["Unknown"] * len(texts)

        seed_topic_names = list(self.seed_topics.keys())
        # 将每个主题的关键词拼成一句话进行编码
        seed_texts = [" ".join(self.seed_topics[t]) for t in seed_topic_names]
        seed_embeddings = self.embedding_model.encode(seed_texts)
        
        similarities = cosine_similarity(embeddings, seed_embeddings)
        
        assigned_topics = []
        for i in range(len(texts)):
            max_sim_idx = np.argmax(similarities[i])
            max_sim = similarities[i][max_sim_idx]
            
            if max_sim >= threshold:
                assigned_topics.append(seed_topic_names[max_sim_idx])
            else:
                assigned_topics.append("Unknown")
                
        return assigned_topics

    def run_topic_modeling(self, text_series, output_dir, window_id, title_info=""):
        """主入口函数：对给定的文本序列进行主题建模"""
        
        # 1. 数据清洗
        clean_mask = (
            text_series.notna() & 
            (text_series != 'FILTER_NEUTRAL_LEACHING') & 
            (text_series.str.len() > 1) &
            (~text_series.str.contains(r'^\s*$', regex=True))
        )
        
        df = pd.DataFrame({'text': text_series[clean_mask]})
        
        # [修改] 只有1条也允许跑，只是不聚类
        if len(df) == 0:
            return None

        # 如果样本太少，直接跳过复杂建模，只做词云
        if len(df) < 3:
            print(f"[TopicModeler] Too few samples ({len(df)}) for clustering. Generating WordCloud only.")
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                wc_path = self._plot_wordcloud(df['text'], output_dir, window_id, title_info)
                return {
                    "total_samples": len(df),
                    "visualizations": {"wordcloud": wc_path}
                }
            return {"total_samples": len(df)}

        self._load_model()
        print(f"[TopicModeler] Processing {len(df)} texts for window {window_id}...")

        # 2. 语义嵌入
        embeddings = self.embedding_model.encode(df['text'].tolist(), show_progress_bar=False)
        
        # 3. 预设主题匹配
        df['seed_topic'] = self._match_seed_topics(df['text'].tolist(), embeddings)
        
        # 4. 降维与聚类
        # [修改] 更加鲁棒的参数设置
        # n_neighbors = min(15, len(df) - 1)
        # if n_neighbors < 2: n_neighbors = 2
        
        # min_cluster_size = min(5, len(df))
        # if min_cluster_size < 2: min_cluster_size = 2

        # viz_embeds = np.zeros((len(df), 2)) # 默认值
        # wc_path = None
        # scatter_path = None

        # try:
        #     umap_embeds = umap.UMAP(
        #         n_neighbors=n_neighbors, 
        #         n_components=5, 
        #         min_dist=0.0, 
        #         metric='cosine', 
        #         random_state=42
        #     ).fit_transform(embeddings)

        #     clusterer = hdbscan.HDBSCAN(
        #         min_cluster_size=min_cluster_size,
        #         metric='euclidean',
        #         cluster_selection_method='eom'
        #     )
        #     labels = clusterer.fit_predict(umap_embeds)
        #     df['cluster'] = labels
            
        #     # 2D 嵌入用于可视化
        #     viz_embeds = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42).fit_transform(embeddings)
            
        # except Exception as e:
        #     print(f"[TopicModeler] Clustering failed (sample too small?): {e}")
        #     df['cluster'] = -1
        #     # 如果聚类失败，全当作一个类
        #     if len(df) > 0:
        #         df['cluster'] = 0
        n_neighbors = min(15, len(df) - 1)
        if n_neighbors < 2: n_neighbors = 2
        
        # 【修改1】提高 min_cluster_size，避免碎片化小簇
        min_cluster_size = max(min(10, len(df) // 5), 2)

        viz_embeds = np.zeros((len(df), 2)) # 默认值
        wc_path = None
        scatter_path = None

        try:
            umap_embeds = umap.UMAP(
                n_neighbors=n_neighbors, 
                n_components=5, 
                # 【修改2】增大 min_dist，减少对微小差异的敏感度
                min_dist=0.1, 
                metric='cosine', 
                random_state=42
            ).fit_transform(embeddings)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                cluster_selection_method='eom',
                # 【修改3】通过 min_samples 进一步控制簇的密度要求
                min_samples=max(2, min_cluster_size // 2)
            )
            labels = clusterer.fit_predict(umap_embeds)
            df['cluster'] = labels
            
            # 【新增】簇间语义合并：将 centroid 余弦相似度 > 阈值的簇合并
            df['cluster'] = self._merge_similar_clusters(
                df, embeddings, merge_threshold=0.65
            )
            
            # 2D 嵌入用于可视化
            viz_embeds = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42).fit_transform(embeddings)
            
        except Exception as e:
            print(f"[TopicModeler] Clustering failed (sample too small?): {e}")
            df['cluster'] = -1
            if len(df) > 0:
                df['cluster'] = 0
        # 5. 生成关键词 (c-TF-IDF)
        docs_per_class = df.groupby(['cluster'], as_index=False).agg({'text': ' '.join})
        # 过滤噪声 (-1)，除非全是噪声
        docs_valid = docs_per_class[docs_per_class['cluster'] != -1]
        if docs_valid.empty and not docs_per_class.empty:
            docs_valid = docs_per_class # 降级策略：包含噪声

        topic_keywords = {}
        if not docs_valid.empty:
            ctfidf = ClassTFIDF(self.stopwords)
            # 尝试提取关键词，如果失败则跳过
            ctfidf_res = ctfidf.fit_transform(docs_valid['text'].tolist())
            
            if ctfidf_res is not None:
                for idx, row in docs_valid.iterrows():
                    cluster_id = row['cluster']
                    # 重新定位 index
                    internal_idx = list(docs_valid['cluster']).index(cluster_id)
                    kws = ctfidf.get_top_words(internal_idx, top_n=5)
                    topic_keywords[cluster_id] = kws
        
        # 6. 生成可视化图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # A. 词云图
            wc_path = self._plot_wordcloud(df['text'], output_dir, window_id, title_info)
            
            # B. 聚类散点图
            scatter_path = os.path.join(output_dir, f"cluster_map_{window_id}.png")
            self._plot_clusters(viz_embeds, df['cluster'], df['seed_topic'], scatter_path)

        # 7. 返回结果摘要
        # 【核心修复 1】不再使用 .head(3) 截断，保留所有分类的统计结果
        seed_counts_dict = df['seed_topic'].value_counts().to_dict()
        
        unknown_mask = df['seed_topic'] == 'Unknown'
        unknown_df = df[unknown_mask]
        cluster_counts = unknown_df['cluster'].value_counts()
        
        discovered = {}
        for k, v in cluster_counts.items():
            if k != -1:
                discovered[k] = {
                    "count": v, 
                    "keywords": topic_keywords.get(k, [])
                }
                
        # 【核心修复 2】解决双重计算问题：重置 Unknown 的数量
        # 从总的 Unknown 中，扣除掉已经被聚类成功的数量，剩下的才是纯粹的“散点”
        pure_scatter_count = len(unknown_df[unknown_df['cluster'] == -1])
        seed_counts_dict['Unknown'] = pure_scatter_count

        # 构建用于展示的索引
        topic_indices = {}
        
        # A. 提取已知预设主题的索引
        for topic in df['seed_topic'].unique():
            if topic != 'Unknown':
                topic_indices[topic] = df[df['seed_topic'] == topic].index.tolist()
                
        # B. 提取新发现聚类簇的索引
        for c in unknown_df['cluster'].unique():
            if c != -1:
                cluster_idx = unknown_df[unknown_df['cluster'] == c].index.tolist()
                if cluster_idx:
                    topic_indices[f"Cluster_{c}"] = cluster_idx
                    
        # C. 提取未能聚类的纯散点
        scatter_idx = unknown_df[unknown_df['cluster'] == -1].index.tolist()
        if scatter_idx:
            topic_indices['Unknown_Scatter'] = scatter_idx

        summary = {
            "total_samples": len(df),
            "top_seed_topics": seed_counts_dict, # 传入完整统计字典
            "discovered_clusters": discovered,
            "visualizations": {
                "wordcloud": wc_path,
                "cluster_map": scatter_path
            },
            "topic_indices": topic_indices 
        }
        return summary

    def _merge_similar_clusters(self, df, embeddings, merge_threshold=0.65):
        """
        后处理：计算每个簇的 centroid embedding，
        如果两个簇的余弦相似度超过阈值，则合并为同一个簇。
        """
        labels = df['cluster'].values.copy()
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        if len(unique_labels) <= 1:
            return labels
        
        # 1. 计算每个簇的质心
        centroids = {}
        for cl in unique_labels:
            mask = labels == cl
            centroids[cl] = embeddings[mask].mean(axis=0)
        
        centroid_keys = list(centroids.keys())
        centroid_vecs = np.array([centroids[k] for k in centroid_keys])
        
        # 2. 计算簇间相似度矩阵
        sim_matrix = cosine_similarity(centroid_vecs)
        
        # 3. 贪心合并 (Union-Find)
        parent = {k: k for k in centroid_keys}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        
        for i in range(len(centroid_keys)):
            for j in range(i + 1, len(centroid_keys)):
                if sim_matrix[i][j] >= merge_threshold:
                    union(centroid_keys[i], centroid_keys[j])
        
        # 4. 重新编号
        root_to_new = {}
        new_id = 0
        merged_labels = labels.copy()
        
        for cl in centroid_keys:
            root = find(cl)
            if root not in root_to_new:
                root_to_new[root] = new_id
                new_id += 1
        
        for cl in centroid_keys:
            root = find(cl)
            merged_labels[labels == cl] = root_to_new[root]
        
        n_before = len(unique_labels)
        n_after = len(set(root_to_new.values()))
        if n_before != n_after:
            print(f"[TopicModeler] Merged clusters: {n_before} → {n_after} (threshold={merge_threshold})")
        
        return merged_labels

    def _plot_wordcloud(self, text_series, output_dir, window_id, title_info=""):
        """生成词云"""
        text_joined = " ".join(text_series.tolist())
        words = jieba.cut(text_joined)
        words_filtered = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        if not words_filtered:
            return None

        wc = WordCloud(
            font_path=self.font_path,
            width=800, height=400,
            background_color='white',
            max_words=100
        ).generate(" ".join(words_filtered))
        
        save_path = f"{output_dir}/wordcloud_{window_id}.png"
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        
        if title_info:
            plt.title(f"Topic WordCloud - {title_info}", fontsize=16, color='darkred', pad=20)
            
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path

    def _plot_clusters(self, embeddings_2d, cluster_labels, seed_labels, save_path):
        """生成聚类散点图"""
        plt.figure(figsize=(10, 6))
        
        mask_noise = (cluster_labels == -1)
        if mask_noise.any():
            plt.scatter(embeddings_2d[mask_noise, 0], embeddings_2d[mask_noise, 1], 
                        c='lightgrey', alpha=0.5, s=10, label='Noise')
            
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        if unique_labels:
            scatter = plt.scatter(
                embeddings_2d[~mask_noise, 0], 
                embeddings_2d[~mask_noise, 1], 
                c=cluster_labels[~mask_noise], 
                cmap='tab10', 
                s=20, 
                alpha=0.8
            )
            plt.colorbar(scatter, label='Cluster ID')
            
        plt.title("Topic Clusters (UMAP Projection)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()