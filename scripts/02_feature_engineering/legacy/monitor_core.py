import pandas as pd
import numpy as np
import re
import os
from transformers import pipeline
from tqdm.auto import tqdm
from utilis_preprocess import batch_clean_text, parse_weibo_date, load_stopwords_from_folder,conbine_post_comment
from version_config import ENTITY_CONFIG, GENERIC_TRIGGERS, VERSION_SCHEDULE, SEED_TOPICS, NEUTRAL_KEYWORDS,OFFICIAL, IRRELEVANT, NIKKI_STOPWORDS, GAME_SENTIMENT_DICT
from utilis_visual import calculate_phash, calculate_visual_repetition
from detection_features import calculate_compression_ratio,calculate_weighted_intensity,calculate_gini,categorize_length,calc_neg_ratio,VisualImpactDetector
import zlib  # 引入压缩库
from matplotlib.dates import DateFormatter
from sentence_transformers import SentenceTransformer, util


STOPWORDS_PATH = r'C:\tongji\0 code\02_data_processing\stopwords'
STOPWORDS = NIKKI_STOPWORDS.union(load_stopwords_from_folder(STOPWORDS_PATH))
MODEL_PATH = r'C:\tongji\0 code\02_data_processing\sentiment_model'

class PublicOpinionMonitor:
    def __init__(
            self, 
            seed_topics = SEED_TOPICS, 
            neutral_keywords = NEUTRAL_KEYWORDS,
            official_source = OFFICIAL,
            irrelevant_source = IRRELEVANT, 
            stopwords = STOPWORDS, 
            entity_config=ENTITY_CONFIG, 
            gengric_triggers=GENERIC_TRIGGERS,
            version_schedule=VERSION_SCHEDULE,
            game_sentiment_dict=GAME_SENTIMENT_DICT,
            model_path=MODEL_PATH,
            # topic_model = "paraphrase-multilingual-MiniLM-L12-v2",
            image_base_dir=None,
            RESAMPLE_FREQ = '15min',
            batch_size = 64,
            device_id=0,
            # aggregate_mode=0 # aggregate : 1 else 0 
            ):
        
        # 通用部分
        self.neutral_keywords = neutral_keywords
        self.official_source = official_source
        self.irrelevant_source = irrelevant_source
        self.stopwords = stopwords
        # self.aggregate_mode = aggregate_mode

        real_device = 0 if device_id >= 0 else -1

        
        # 检测部分
        self.resample_freq = RESAMPLE_FREQ
        self.batch_size = batch_size

        # A.情感分析-默认情感词典定义
        self.sentiment_dict = game_sentiment_dict if game_sentiment_dict else {
            "NEGATIVE": ["吃相", "劝退", "恶心", "喂屎", "阴间", "背刺", "暗改", "逼氪", "非酋","难看",'官方说的概率跟实际不一样','不是这对吗','我河呢','狗叠快修' ,'塞爆了'],
            "POSITIVE": ["绝美", "好康", "良心", "欧皇", "丝滑", "期待", "满分","免骂","小蛋糕",'萌死人不偿命','美神']
        }
        # B.情感分析-初始化模型
        print(f"🧠 Loading Sentiment Model on Device: {real_device}...")
        # self.sentiment_pipeline = pipeline(
        #     "text-classification", model=model_name, device=real_device, batch_size=64
        # )
        self.sentiment_pipeline = pipeline(
                "text-classification",
                model=model_path,     # 指向本地路径
                tokenizer = model_path, # 必须显式指定 tokenizer 也指向该路径，否则可能会尝试联网
                device=real_device,         # 确保这是 int (如 0) 或 str (如 "cuda:0")
                batch_size=self.batch_size, # Pipeline 内部批处理大小
                truncation=True,            # 建议在初始化时设置截断
                max_length=512              # 建议在初始化时设置最大长度
            )

        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",device = 'cuda')
        
        if image_base_dir:
            self.visual_detector = VisualImpactDetector(image_base_dir)
        else:
            self.visual_detector = None


        # D.定义官方行为
        self.type_mapping = {
            # 宣发类 (蓝)
            '版本宣发': 'promotion', '其他活动': 'promotion', '抽奖': 'promotion', 'PV': 'promotion',
            
            # 危机/道歉类 (红) - "维护&道歉" 归入此类，因为带有负面承认
            '道歉公告': 'crisis', '维护&道歉': 'crisis', '致歉': 'crisis',
            
            # 维护类 (黄)
            '更新公告': 'maintenance', '维护公告': 'maintenance',
            
            # 攻略/干货类 (绿) - 新增
            '官方攻略': 'guide', '搭配指南': 'guide',
            
            # 日常 (灰)
            '日常互动': 'daily'
        }
        self.official_author_weights = {
            "无限暖暖": 1.0,
            # 其他账号默认使用较低权重
        }
        self.default_official_weight = 0.3

        # 归因部分
        # A.版本实体定义
        # self.entity_list = entity_list
        # if isinstance(entity_config, list):
        #     self.entity_config = {k: [k] for k in entity_config}
        # else:
        #     self.entity_config = entity_config
        
        # B.运营日历及通用实体定义
        # self.version_schedule = version_schedule #新增
        # self.generic_triggers = gengric_triggers #新增

        # # C.预设主题定义
        # self.router = TopicRouter(seed_topics)
        # self.explainer = TopicExplainer(self.stopwords)
        # self.miner = EvidenceMiner(self.router.embedder, self.router)
        
    



    def process_data(self, comment_file=None, post_file=None, official_file=None, combined_file = None, start_date_str = None, end_date_str = None):
        '''数据处理主流程'''
        # 没有完成时间切片部分，暂时在调用时直接切
        # all_features = []

        # def general_clean_process_simple(file_path):
            # df = pd.read_csv(file_path, encoding='utf-8-sig')
        #0: 全部运行，1：只运行数据合并，2：只运行聚合
        if comment_file is not None and post_file is not None:  # 通过分文件输入
            df_total, df_p, df_c = conbine_post_comment(post_file, comment_file)
        else:
            df_total = combined_file

        # 时间空会返回空
        if 'timestamp' in df_total.columns: 
            df_total['timestamp'] = df_total['timestamp'].apply(parse_weibo_date)

        if 'official_timestamp' in df_total.columns: 
            df_total['official_timestamp'] = df_total['official_timestamp'].apply(parse_weibo_date)

        post_condition = df_total['post_type'] != 'comment'
        comment_condition = df_total['post_type'] == 'comment'

        
        # if 'text_clean' not in df_total.columns or 'text_clean_strict' not in df_total.columns:
            # A.数据准备
            # 清洗完成后生成text_clean, text_clean_full
        print("1 正在进行数据清洗")
        df_total['text_clean'] =  df_total['text']
        df_total['text_clean_strict'] =  df_total['text']
        # df_total[post_condition] = batch_clean_text(df_total[post_condition] , self.official_source, self.irrelevant_source, self.neutral_keywords)
        tmp = batch_clean_text(df_total.loc[post_condition, :] , self.official_source, self.irrelevant_source, self.neutral_keywords)
        df_total.loc[post_condition, 'text_clean'] = tmp['text_clean']
        df_total.loc[post_condition, 'text_clean_strict'] = tmp['text_clean_strict']

        # df_total[comment_condition]['text_clean'] =  df_total[comment_condition]['text']
        # df_total[comment_condition]['text_clean_strict'] =  df_total[comment_condition]['text']


        # 简单版mask
        calculate_mask = df_total['text_clean'] != "FILTER_NEUTRAL_LEACHING"
        # 用于情感分析mask
        valid_mask = df_total['text_clean_strict'] != "FILTER_NEUTRAL_LEACHING"
        
        # 标记文本长度--增加列
        df_total['len_type'] = df_total[valid_mask]['text_clean_strict'].apply(categorize_length)

        # 计算情感分数--增加列
        # df.loc[valid_mask, 'text_clean'] = df.loc[valid_mask, 'text_clean'].str.slice(0, 140) # 截断长文本防止显存溢出
        inference_texts = df_total.loc[valid_mask, 'text_clean_strict'].astype(str).tolist()
        
        if inference_texts:
            outputs = []
            # 批量推理
            # for i in range(0, len(inference_texts), self.batch_size):
            for i in tqdm(range(0, len(inference_texts), self.batch_size),
                        total=(len(inference_texts) + self.batch_size - 1) // self.batch_size, # 计算总批次数
                        desc='Sentiment Batch Inference'):
                batch = inference_texts[i:i + self.batch_size]
                outputs.extend(self.sentiment_pipeline(batch, truncation=True, max_length=512))
            
            df_total.loc[valid_mask, 'raw_score'] = [x['score'] for x in outputs]
            df_total.loc[valid_mask, 'label'] = [x['label'] for x in outputs]
        else:
            # 防止全天无有效数据时报错，默认分数为0.5
            df_total['raw_score'] = 0.5
            df_total['label'] = 'neutral'
        
        # 混合打分 (BERT + 规则 + 领域情感极性词典)

        self._apply_sentiment_rules(df_total)     
        # C.简单清洗特征
        # else:
        #     print("1 无需进行数据清洗，跳过此步骤")

    def compute_phashes(self, df_total = None):

        print('6 计算视觉特征')
        post_condition = df_total['post_type'] != 'comment'
        comment_condition = df_total['post_type'] == 'comment'

        if 'img_phashes' not in df_total.columns and self.visual_detector is not None:
            df_post_visual = df_total.loc[post_condition].copy()

            if not df_post_visual.empty:
                # A. 先把hash算好并写回df（这一步可以缓存、也可以落盘）
                df_post_visual_h = self.visual_detector.attach_hashes(
                    df_post_visual,
                    time_col='timestamp',
                    image_col='image_urls',
                    video_col='video_cover_url',
                    out_hash_col='img_phashes'
                )

                df_total.loc[post_condition, 'img_phashes'] = df_post_visual_h['img_phashes']
                df_total.loc[post_condition, 'has_img'] = df_post_visual_h['has_img']
                df_total.loc[post_condition, 'img_count'] = df_post_visual_h['img_count']

        elif self.visual_detector is not None:
            print('已有视觉hash，跳过视觉hash计算')
        else:
            print('无视觉检测器，跳过视觉hash计算')
        
        # if self.aggregate_mode == 0: #只合成
        return df_total

    def aggregate(self, df_total=None):


        print("2 计算简单聚合特征")

        # 设置时间戳索引
        df_total['timestamp'] = pd.to_datetime(df_total['timestamp'])
        df_total.set_index('timestamp',inplace= True)

        # ========== 重新定义mask（索引变更后） ==========
        post_condition = df_total['post_type'] != 'comment'
        comment_condition = df_total['post_type'] == 'comment'
        # 简单版mask
        calculate_mask = df_total['text_clean'] != "FILTER_NEUTRAL_LEACHING"
        # 用于情感分析mask
        valid_mask = df_total['text_clean_strict'] != "FILTER_NEUTRAL_LEACHING"
        
        # ========== 分开计算的特征 ==========
        post_valid = post_condition & valid_mask
        comment_valid = comment_condition & valid_mask
        post_calc = post_condition & calculate_mask
        comment_calc = comment_condition & calculate_mask

        # --- 分别计算 volume ---
        # vol_series = df_total[calculate_mask].groupby(pd.Grouper(key = 'timestamp',freq = self.resample_freq)).count()
        vol_post = df_total[post_calc]['text'].resample(self.resample_freq).count()
        vol_comment = df_total[comment_calc]['text'].resample(self.resample_freq).count()
        
        # 计算参与集中度
        # --- 分别计算 unique_users ---
        unique_post = df_total[post_valid]['author_id'].resample(self.resample_freq).nunique()
        unique_comment = df_total[comment_valid]['author_id'].resample(self.resample_freq).nunique()
        # vol_series_author = df_total[comment_condition]['author_id'].resample(self.resample_freq).count()


        # --- 分别计算基尼系数gini ---
        gini_post = df_total[post_valid]['author_id'].resample(self.resample_freq).apply(calculate_gini)
        gini_comment = df_total[comment_valid]['author_id'].resample(self.resample_freq).apply(calculate_gini)

        # 计算官方微博耦合度

        # --- 计算转发占比（retweet)/(original + retweet) ---
        orig_ratio_series = df_total[post_calc].resample(self.resample_freq).apply(
            lambda x: (x['post_type'] == 'retweet').sum() / (len(x) + 15))
        
        # if 'post_type' in df_total.columns:
            # orig_ratio_series = df_total[calculate_mask].groupby(pd.Grouper(key = 'timestamp',freq = self.resample_freq)).apply(lambda x: (x['post_typr']!='retweet').sum() / (len(x) + 1e-5))
            # orig_ratio_series = df_total[calculate_mask].resample(self.resample_freq).apply(lambda x: (x['post_type']!='retweet').sum() / (len(x) + 1e-5))
        
        # else:
        #     orig_ratio_series = pd.Series(0.5, index=vol_series.index) 

        # --- 分别计算符号密度（加权情感烈度）senti_symbol ---
        # symbol_vol = df_total[calculate_mask]['text_clean'].resample(self.resample_freq).apply(calculate_weighted_intensity)
        symbol_post = df_total[post_calc]['text_clean'].resample(self.resample_freq).apply(calculate_weighted_intensity)
        symbol_comment = df_total[comment_calc]['text_clean'].resample(self.resample_freq).apply(calculate_weighted_intensity)

        
        # ========== 计算文本特征 ==========
        print("3 计算文本特征")

        # --- 分别计算文本压缩比 ---
        comp_post = df_total[post_valid]['text_clean_strict'].resample(self.resample_freq).apply(calculate_compression_ratio)
        comp_comment = df_total[comment_valid]['text_clean_strict'].resample(self.resample_freq).apply(calculate_compression_ratio)
        
        # 不同长度文本数量计算
        
        # 计算总量
        # short_vol = df_total[df_total['len_type'] == 'Short'].groupby(pd.Grouper(key='timestamp', freq=self.resample_freq))['text_clean_strict'].count()
        # medium_vol = df_total[df_total['len_type'] == 'Medium'].groupby(pd.Grouper(key='timestamp', freq=self.resample_freq))['text_clean_strict'].count()
        # long_vol = df_total[df_total['len_type'] == 'Long'].groupby(pd.Grouper(key='timestamp', freq=self.resample_freq))['text_clean_strict'].count()

        # --- 分别计算长度分布 ---
        short_post = df_total[(df_total['len_type'] == 'Short') & post_condition].resample(self.resample_freq)['text_clean_strict'].count()
        medium_post = df_total[(df_total['len_type'] == 'Medium') & post_condition].resample(self.resample_freq)['text_clean_strict'].count()
        long_post = df_total[(df_total['len_type'] == 'Long') & post_condition].resample(self.resample_freq)['text_clean_strict'].count()
        short_comment = df_total[(df_total['len_type'] == 'Short') & comment_condition].resample(self.resample_freq)['text_clean_strict'].count()
        medium_comment = df_total[(df_total['len_type'] == 'Medium') & comment_condition].resample(self.resample_freq)['text_clean_strict'].count()
        long_comment = df_total[(df_total['len_type'] == 'Long') & comment_condition].resample(self.resample_freq)['text_clean_strict'].count()
        
        
        # ========== 计算情感极性 ==========
        print('4 计算情感极性')
        sent_series = df_total.resample(self.resample_freq)['sentiment_score'].mean()
        
        # 计算情感均值
        # sent_series = df_total.groupby(pd.Grouper(key='timestamp', freq=self.resample_freq))['sentiment_score'].mean()
        # sent_series = df_total.resample(self.resample_freq)['sentiment_score'].mean()


    
        neg_ratio_post = df_total[post_condition]['sentiment_score'].resample(self.resample_freq).apply(calc_neg_ratio)
        neg_ratio_comment = df_total[comment_condition]['sentiment_score'].resample(self.resample_freq).apply(calc_neg_ratio)

        # ========== 分别计算语义中心和漂移 ==========
        print('5 计算语义距离')

        df_post_filtered = df_total[post_calc].copy()
        df_comment_filtered = df_total[comment_calc].copy()

        # 获取完整时间线
        full_timeline = pd.date_range(
            start=df_total.index.min().floor(self.resample_freq),
            end=df_total.index.max().ceil(self.resample_freq),
            freq=self.resample_freq
        )
        timeline = list(full_timeline)

        embeddings_post = {t: None for t in timeline}
        embeddings_comment = {t: None for t in timeline}

        # Post语义中心
        if not df_post_filtered.empty:
            for time_bucket, group in tqdm(df_post_filtered.resample(self.resample_freq), 
                                           desc="   - Post语义向量计算"):
                if len(group) > 0:
                    texts_p = group['text_clean'].tolist()
                    vectors_p = self.embedder.encode(texts_p, show_progress_bar=False, batch_size=64)
                    embeddings_post[time_bucket] = np.mean(vectors_p, axis=0)

        # Comment语义中心  
        if not df_comment_filtered.empty:
            for time_bucket, group in tqdm(df_comment_filtered.resample(self.resample_freq), 
                                           desc="   - Comment语义向量计算"):
                if len(group) > 0:
                    texts_c = group['text_clean'].tolist()
                    vectors_c = self.embedder.encode(texts_c, show_progress_bar=False, batch_size=64)
                    embeddings_comment[time_bucket] = np.mean(vectors_c, axis=0)

        # --- 分别计算语义漂移 ---
        def calc_shift_series(embeddings_map, timeline):
            shifts = []
            prev_vec = None
            for t in timeline:
                curr_vec = embeddings_map.get(t)
                if curr_vec is None or prev_vec is None:
                    shifts.append(0.0)
                else:
                    sim = util.cos_sim(curr_vec, prev_vec).item()
                    shifts.append(1 - sim)
                if curr_vec is not None:
                    prev_vec = curr_vec
            return pd.Series(shifts, index=pd.to_datetime(timeline))
        
        shifts_post = calc_shift_series(embeddings_post, timeline)
        shifts_comment = calc_shift_series(embeddings_comment, timeline)
        
        # --- 交叉语义漂移：comment → post ---
        shifts_cross = []
        for t in timeline:
            vec_c = embeddings_comment.get(t)
            vec_p = embeddings_post.get(t)
            if vec_c is None or vec_p is None:
                shifts_cross.append(0.0)
            else:
                sim = util.cos_sim(vec_c, vec_p).item()
                shifts_cross.append(1 - sim)
        shifts_cross_series = pd.Series(shifts_cross, index=pd.to_datetime(timeline))
        
        # ========== 计算视觉特征 ==========
        print('6 计算视觉特征')
        if self.visual_detector is not None:
            # 准备 post 数据（需要 reset_index 因为 timestamp 已成为索引）
            df_post_visual = df_total[post_condition].copy().reset_index()
            print('df_post_visual shape:', df_post_visual.shape)
            if not df_post_visual.empty:
                start_dt = df_post_visual['timestamp'].min()
                end_dt = df_post_visual['timestamp'].max()
                
                # visual_result = self.visual_detector.analyze(
                visual_result = self.visual_detector.aggregate(
                    df_post_visual,
                    start_date=str(start_dt),
                    end_date=str(end_dt),
                    time_col='timestamp',
                    freq=self.resample_freq
                )
                
                if not visual_result.empty:
                    visual_feat = visual_result.set_index('time')[['vis_abs_redundancy', 'vis_concentration']]
                    print('visual_result不为空')
                else:
                    visual_feat = pd.DataFrame(index=timeline, columns=['vis_abs_redundancy', 'vis_concentration'])
                    print('visual_result为空')
            else:
                visual_feat = pd.DataFrame(index=timeline, columns=['vis_abs_redundancy', 'vis_concentration'])
                print('df_post_visual为空')
        else:
            visual_feat = pd.DataFrame(index=timeline, columns=['vis_abs_redundancy', 'vis_concentration'])
            print('无视觉检测器，跳过视觉特征计算')
        
        visual_feat.index = pd.to_datetime(visual_feat.index)
        
        # return vol_series, unique_user_series,gini_series,symbol_vol,comp_series,short_vol, medium_vol,long_vol,neg_ratio_series,shifts
       
        # ========== 合并特征 ==========
        print("7 合并所有特征")
        feature_dict = {
            'total_volume_post': vol_post,
            'total_volume_comment': vol_comment,
            'unique_users_post': unique_post,
            'unique_users_comment': unique_comment,
            'gini_post': gini_post,
            'gini_comment': gini_comment,
            'senti_symbol_post': symbol_post,
            'senti_symbol_comment': symbol_comment,
            'comp_ratio_post': comp_post,
            'comp_ratio_comment': comp_comment,
            'origin_ratio': orig_ratio_series,
            'total_short_post': short_post,
            'total_medium_post': medium_post,
            'total_long_post': long_post,
            'total_short_comment': short_comment,
            'total_medium_comment': medium_comment,
            'total_long_comment': long_comment,
            'neg_ratio_post': neg_ratio_post,
            'neg_ratio_comment': neg_ratio_comment,
            'semantic_shift_post': shifts_post,
            'semantic_shift_comment': shifts_comment,
            'semantic_shift_cross': shifts_cross_series,  # 评论区→广场
            'vis_abs_redundancy_post': visual_feat['vis_abs_redundancy'],
            'vis_concentration_post': visual_feat['vis_concentration'],
        }
        df_feat = pd.concat(feature_dict, axis=1)
        df_feat.index = pd.to_datetime(df_feat.index)

        return df_feat

        # # ========== 官方特征（含权重） ==========
        # df_off = official_file
        # if 'timestamp' in df_off.columns: 
        #     df_off['timestamp'] = df_off['timestamp'].apply(parse_weibo_date)
        # if 'official_timestamp' in df_off.columns: 
        #     df_off['official_timestamp'] = df_off['official_timestamp'].apply(parse_weibo_date)
        # if 'official_post_type' in df_off.columns:
        #     df_off['mapped_type'] = df_off['official_post_type'].map(self.type_mapping)
        
        # feat_off = self._process_official_features(df_off, df_feat.index)

        # df_final = pd.merge(df_feat, feat_off, left_index=True, right_index=True, how='left')
        # df_final = df_final.fillna(0)
            

#------------------------------------------------------检测环节函数定义------------------------------------------------

    def _apply_sentiment_rules(self, df):
        '''
        情感检测--领域词典修正下的分类
        '''
        pattern = '|'.join([re.escape(kw) for kw in self.neutral_keywords]) # 合并中性词
        mask_keyword = df['text_clean'].str.contains(pattern, regex=True, na=False)
        mask_leech = (df['text_clean'] == "FILTER_NEUTRAL_LEACHING")
        mask_pos = df['label'].astype(str).str.contains('positive|4|5', case=False, na=False) # 正面输出
        
        # 领域词典修正
        pat_force_neg = '|'.join(self.sentiment_dict["NEGATIVE"])
        pat_force_pos = '|'.join(self.sentiment_dict["POSITIVE"])
        mask_force_neg = df['text_clean'].str.contains(pat_force_neg, regex=True, na=False)
        mask_force_pos = df['text_clean'].str.contains(pat_force_pos, regex=True, na=False)

        conditions = [
            mask_leech | mask_keyword, # 中性 0.5
            mask_force_neg,            # 强制负面 0.1
            mask_force_pos,            # 强制正面 0.9
            mask_pos,                  # 模型正
            ~mask_pos                  # 模型负
        ]
        # 级联选择
        choices = [0.5, 0.1, 0.9, df['raw_score'], 1 - df['raw_score']]
        
        # 处理可能出现的 NaN (如果 inference_texts 为空)
        default_score = 0.5
        
        df['sentiment_score'] = np.select(conditions, choices, default=default_score)
        df['is_negative'] = df['sentiment_score'] < 0.4


    # --- 计算官方微博相关特征 ---
    def _process_official_features(self, df_off, target_timeline):
        """
        【修正版】处理官方发文 - 加入账号权重
        """
        features_off = pd.DataFrame(index=target_timeline)
        features_off.index.name = 'timestamp'
        
        # ========== 添加账号权重 ==========
        if 'official_author' in df_off.columns:
            df_off['author_weight'] = df_off['official_author'].map(
                lambda x: self.official_author_weights.get(x, self.default_official_weight)
            )
        else:
            df_off['author_weight'] = 1.0
        
        right_df = df_off[['timestamp', 'mapped_type', 'author_weight']].copy()
        right_df['timestamp'] = pd.to_datetime(right_df['timestamp'])
        right_df = right_df.dropna(subset=['timestamp'])
        right_df = right_df.sort_values('timestamp')
        right_df = right_df.rename(columns={'timestamp': 'last_post_time'})

        if right_df.empty:
            merged = features_off.reset_index()
            merged['last_post_time'] = pd.NaT
            merged['mapped_type'] = None
            merged['author_weight'] = 0
        else:
            merged = pd.merge_asof(
                features_off.reset_index(),
                right_df,
                left_on='timestamp',
                right_on='last_post_time',
                direction='backward'
            )
            merged = merged.set_index('timestamp')
            
        if 'last_post_time' in merged.columns and not merged['last_post_time'].isna().all():
            time_diff_minutes = (merged.index - merged['last_post_time']).dt.total_seconds() / 60
            time_diff_minutes = time_diff_minutes.fillna(999999)
        else:
            time_diff_minutes = pd.Series(999999, index=merged.index)

        # ========== 分别计算comment和post的衰减 ==========
        # comment衰减：直接影响，衰减系数120分钟
        decay_comment = np.exp(-time_diff_minutes / 120) * merged['author_weight'].fillna(0)
        
        # post衰减：传递影响，延迟30分钟后开始衰减，衰减系数240分钟（更慢）
        delayed_diff = np.maximum(time_diff_minutes - 30, 0)
        decay_post = np.exp(-delayed_diff / 240) * merged['author_weight'].fillna(0) * 0.6  # 传递损耗
        
        merged['official_impact_decay_comment'] = decay_comment
        merged['official_impact_decay_post'] = decay_post

        active_mask = merged['official_impact_decay_comment'] > 0.01 
        
        merged['is_promotion'] = ((merged['mapped_type'] == 'promotion') & active_mask).astype(int)
        merged['is_crisis'] = ((merged['mapped_type'] == 'crisis') & active_mask).astype(int)
        merged['is_maintenance'] = ((merged['mapped_type'] == 'maintenance') & active_mask).astype(int)
        merged['is_guide'] = ((merged['mapped_type'] == 'guide') & active_mask).astype(int)
        merged['is_daily'] = ((merged['mapped_type'] == 'daily') & active_mask).astype(int)
        
        return merged[['official_impact_decay_comment', 'official_impact_decay_post', 
                       'is_promotion', 'is_crisis', 'is_maintenance', 'is_guide', 'is_daily']]


#------------------------------------------------------归因环节函数定义------------------------------------------------
        def  specific_clean_process(df):
            '''归因阶段清洗过程'''
            # B. 多标签路由 (【修复】解决 ValueError: setting an array element with a sequence)
            # 筛选有效文本 (排除被过滤的噪音数据)
            df = batch_clean_text(df, "full", self.official_source, self.irrelevant_source)

            valid_mask = df['text_clean'] != "FILTER_NEUTRAL_LEACHING"
            valid_texts = df.loc[valid_mask, 'text_clean'].tolist()
            
            # 1. 先初始化全列为 None
            df['assigned_topics'] = None
            
            if valid_texts:
                # 2. 获取路由结果 (List of Lists)--每个文本对应一个主题列表
                routed_data = self.router.route_text_multilabel(valid_texts)
                
                # 3. 【关键修复】显式创建 dtype=object 的 Series 并对齐索引
                # 这样 Pandas 就会把每个 list 当作一个独立的 object，而不是尝试转换为 2D 数组
                assignment_series = pd.Series(routed_data, index=df[valid_mask].index, dtype=object)
                df.loc[valid_mask, 'assigned_topics'] = assignment_series
                
            # 4. 填充剩余空值为 ["Unknown"] (处理被过滤的噪音数据)
            # 同样使用 Series 对齐赋值，防止广播错误
            null_mask = df['assigned_topics'].isnull()
            if null_mask.sum() > 0:
                unknown_series = pd.Series([["Unknown"]] * null_mask.sum(), index=df[null_mask].index, dtype=object)
                df.loc[null_mask, 'assigned_topics'] = unknown_series
            
            #强制中性主题归类
            mask_neutral = df['sentiment_score'] == 0.5
            if mask_neutral.sum() > 0:
                # 创建包含 ["Neutral"] 的 Series
                neutral_topics = pd.Series([["Neutral"]] * mask_neutral.sum(), index=df[mask_neutral].index, dtype=object)
                df.loc[mask_neutral, 'assigned_topics'] = neutral_topics
                # 标记一下，方便后续验证
                df.loc[mask_neutral, 'topic_keywords_ref'] = "Rule_Based_Neutral"
            # E. 生成实体
            df['resolved_entities'] = self._resolve_generic_entities(df)

            # # F. 生成特征矩阵
            # df_features = self._aggregate_features(df)

            # # F. 生成用户侧特征矩阵
            # df_features_user = self._aggregate_features(df)
        
            # # G. 生成并合并官方侧特征
            # target_timeline = df_features_user.index
            # df_features_official = self._process_official_features(official_file_path, target_timeline)
            
            # df_final_features = pd.merge(
            #     df_features_user, 
            #     df_features_official, 
            #     left_index=True, 
            #     right_index=True, 
            #     how='left'
            # ).fillna(0)
            
            # # 【修改点】H. 归因部分被移除
            # # 此时不进行归因，直接返回数据给后续的检测模型使用
            
            # print("✅ 特征工程完成，已生成时序特征矩阵。归因分析将在检测模型报警后进行。")
            
            # # 返回: 特征矩阵, 原始清洗后的明细表 (用于后续查证)
            # return df_final_features, df
            # # G. 归因与实体提取

            # neg_df = df[df['sentiment_score'] < 0.4].copy()
            # keywords = self.explainer.extract_keywords(neg_df)
            # entities = self._extract_top_entities(neg_df)
            
            # # return df_features, keywords, entities, neg_df
            # return df_features, keywords, entities, df #返回全量数据集用于人工验证


    def _extract_top_entities(self, df_neg):
        """
        归因环节 支持别名合并的实体统计
        """
        if not self.entity_config or df_neg.empty: return {}
        
        counts = {}
        
        # 遍历配置字典中的每一个实体
        for canonical_name, aliases in self.entity_config.items():
            # 1. 构建正则模式：(花漾梦萦|精灵|精灵套)
            # re.escape 用于防止别名里包含特殊符号导致正则报错
            pattern = '|'.join([re.escape(alias) for alias in aliases])
            
            # 2. 统计所有别名在负面文本中出现的【总次数】
            # str.count 支持正则，能一次性把所有变体都算进去
            c = df_neg['text_clean'].str.count(pattern).sum()
            
            if c > 0: 
                counts[canonical_name] = c
                
        # 返回 Top 5
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
    

        def _resolve_generic_entities(self, df):
            """
            归因环节 时序实体消歧
            将 "五星"、"新活动" 等泛指词，根据发帖时间映射为具体实体 ID
            通过entity_list映射到entity_config中对应的实体类别
            """
            # 1. 预处理时间表，方便查询
            # 将字符串时间转换为 Timestamp 以便比较
            for period in self.version_schedule:
                period['_start'] = pd.to_datetime(period['start_time'])
                period['_end'] = pd.to_datetime(period['end_time'])

            # 定义单行处理函数
            def resolve_row(row):
                text = row['text_clean']
                post_time = row['timestamp']
                
                detected_entities = []
                
                # A. 先匹配具体的 ENTITY_CONFIG (静态匹配)
                # (这部分逻辑保留您原有的，或者整合进来)
                for entity_key, keywords in self.entity_config.items():
                    for kw in keywords:
                        if kw in text:
                            detected_entities.append(entity_key)
                            break # 一个实体类只匹配一次
                
                # B. 动态泛指匹配 (新增)
                # 找到当前时间所属的版本区间
                current_period = None
                for period in self.version_schedule:
                    if period['_start'] <= post_time <= period['_end']:
                        current_period = period
                        break
                
                if current_period:
                    # 检查是否包含泛指词
                    for category, triggers in self.generic_triggers.items():
                        # 只有当该类别没有被具体实体捕获时，才进行推断
                        # (例如：如果已经识别出了"龙猫套"，就不用把"五星"再映射一遍，防止重复计数)
                        already_has_specific = any(e.startswith(current_period['version']) and category in e for e in detected_entities)
                        
                        if not already_has_specific:
                            for trigger in triggers:
                                if trigger in text:
                                    # 映射到具体实体
                                    target_entity = current_period['mappings'].get(category)
                                    if target_entity:
                                        detected_entities.append(target_entity)
                                    break
                
                return list(set(detected_entities)) # 去重

            # 应用到 DataFrame
            return df.apply(resolve_row, axis=1)