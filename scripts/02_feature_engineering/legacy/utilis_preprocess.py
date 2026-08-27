import re
import pandas as pd
from datetime import datetime
import locale
import jieba
import os
import glob
from collections import Counter
from scipy.stats import entropy


current_dir = os.path.dirname(os.path.abspath(__file__))
dict_path = os.path.join(current_dir, "nikki_dict.txt")
if os.path.exists(dict_path):
    jieba.load_userdict(dict_path)
    print(f"✅ 成功加载自定义词典: {dict_path}")
else:
    print(f"⚠️ 未找到自定义词典: {dict_path}，使用默认词库")



def parse_weibo_date(date_str):
    """
    解析微博多种复杂的时间格式，返回标准 datetime
    """
    if pd.isna(date_str): return pd.NaT
    date_str = str(date_str).strip()
    
    # 格式 1: 2024-2024年12-01 00:49:00 (爬虫乱码)
    match = re.match(r"(\d{4})-(\d{4})年(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})", date_str)
    if match:
        try:
            _, year, month, day, time_str = match.groups()
            return pd.to_datetime(f"{year}-{month}-{day} {time_str}")
        except:
            return pd.NaT

    # 格式 2: Mon Sep 29 13:02:14 +0800 2025
    try:
        # 临时切换 locale 以匹配英文月份
        original_locale = locale.getlocale(locale.LC_TIME)
        try:
            locale.setlocale(locale.LC_TIME, 'en_US.UTF-8') 
        except:
            pass # Windows下可能报错，忽略
            
        dt = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')
        locale.setlocale(locale.LC_TIME, original_locale) 
        return pd.to_datetime(dt).tz_localize(None)
    except:
        pass

    # 格式 3: 标准 YYYY-MM-DD HH:MM:SS
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT

def load_stopwords_from_folder(folder_path):
    """
    读取指定文件夹下所有txt文件的内容，合并为停用词集合
    """
    loaded_words = set()
    
    if not os.path.exists(folder_path):
        print(f"⚠️ 警告: 停用词文件夹不存在: {folder_path}，将仅使用内置停用词。")
        return loaded_words

    print(f"📖 开始从文件夹加载停用词: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            # 尝试用 utf-8 读取，处理换行符
            with open(file_path, 'r', encoding='utf-8') as f:
                words = {line.strip() for line in f if line.strip()}
                loaded_words.update(words)
            print(f"  - 已加载: {filename} ({len(words)} 个词)")
        except UnicodeDecodeError:
            # 如果 utf-8 失败，尝试 gbk (防Windows记事本编码问题)
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    words = {line.strip() for line in f if line.strip()}
                    loaded_words.update(words)
                print(f"  - 已加载(GBK): {filename} ({len(words)} 个词)")
            except Exception as e:
                print(f"  ❌ 读取失败 {filename}: {e}")
 
    return loaded_words







def batch_clean_text(df, official_source, irrelevant_source,neutral_keywords):
    """
    向量化清洗核心逻辑
    mode = simple 在检测环节使用，只有去除官方转发文案、无关来源标记、过短文本标记、
    mode = full 在归因环节使用
    """
    #将微博正文/评论内容修改名字为text
    if 'text' not in df.columns and 'content' in df.columns:
        df['text'] = df['content']
    if 'text' not in df.columns and 'comment_content' in df.columns:
        df['text'] = df['comment_content']
    
    df['text_clean'] = df['text'].astype(str)
    
    # 1. 来源清洗
    if 'source_author_name' in df.columns and 'post_type' in df.columns:
        is_retweet = (df['post_type'] == 'retweet')
        mask_official = is_retweet & df['source_author_name'].isin(official_source)
        mask_irrelevant = is_retweet & df['source_author_name'].isin(irrelevant_source)
        
        # 去除官方转发文案 (保留 //@ 之前的内容)
        df.loc[mask_official, 'text_clean'] = df.loc[mask_official, 'text_clean'].str.split('//@').str[0]
        # 标记无关来源
        df.loc[mask_irrelevant, 'text_clean'] = "FILTER_NEUTRAL_LEACHING"
        # 标记中性词
        pattern = '|'.join([re.escape(kw) for kw in neutral_keywords]) # 合并中性词
        mask_keyword = df['text_clean'].str.contains(pattern, regex=True, na=False)
        df.loc[mask_keyword, 'text_clean'] = "FILTER_NEUTRAL_LEACHING"        
        
        df.loc[df['text_clean'].str.len() < 1, 'text_clean'] = "FILTER_NEUTRAL_LEACHING"

        # 2. 通用正则清洗
        # 去除: 超话, 表情包 [], 话题 ##, 链接
        # regex_pattern = r'无限暖暖超话|\[.*?\]|#.*?#||展开[a-z]|O网页链接|http\S+'
        regex_pattern = r'L.*?微博视频|無限暖暖|无限暖暖|暖暖|转发|转发微博|Yuanese|轉發微博|超话|#无限暖暖全球公测#|\[.*?\]|#.*?#||展开[a-z]|O网页链接|🌸|链接|打卡|http\S+|[，,。？!！…~]{2,}|[@《》]|12月5日|12\.5|12月\d+日|啊{2,}|_那個夏日已然飽和_|Nikki|nikki'
        df['text_clean_strict'] = df['text_clean'].str.replace(regex_pattern, '', regex=True).str.strip()
    
        # 3. 标记过短文本
        df.loc[df['text_clean_strict'].str.len() < 2, 'text_clean'] = "FILTER_NEUTRAL_LEACHING"
        
        # df = df[df['text_clean_strict'].str.len() > 1].copy() #加一行强力去空
    
    return df

def tokenize_zh_factory(stopwords_set):
    """Jieba 分词闭包，专门用于 c-TF-IDF"""
    def tokenize(text):
        # 预清洗：去掉可能影响分词的特殊符号
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1 and w not in stopwords_set]
    return tokenize


# def calculate_text_entropy(text_series):
#     """
#     特征6: 文本信息熵 (Text Entropy)
#     逻辑: 拼接文本 -> 词频分布 -> 香农熵
#     """
#     if text_series.empty:
#         return 10.0 # 默认高熵（表示无重复）
        
#     # 拼接所有文本 (假设已经分词或用空格分隔)
#     full_text = " ".join(text_series.astype(str).tolist())
    
#     # 简单按字/词统计 (这里用字粒度更抗干扰，或者用 split() 按词)
#     # 如果 text_clean 是中文句子，建议 list(full_text) 按字统计
#     tokens = list(full_text) 
    
#     if not tokens:
#         return 0.0
        
#     # 计算概率分布
#     counts = Counter(tokens)
#     total = sum(counts.values())
#     probs = [c / total for c in counts.values()]

#     # 计算熵
#     return entropy(probs, base=2)

# def calculate_symbol_density(text_series):
#     """
#     特征8: 特殊符号密度 (Symbol Density)
#     逻辑: 统计包含 ❗, ?, 🕯️ 的评论条数占比
#     """
#     if text_series.empty:
#         return 0.0
    
#     # 定义敏感符号正则
#     pattern = r'[❗!！\?？🕯️]' 
#     match_count = text_series.str.contains(pattern, regex=True).sum()
    
#     return match_count / len(text_series)

# # --- monitor_core.py 头部新增/修改辅助函数 ---

# def calculate_super_topic_count(series):
#     """
#     统计 user_title 中包含 '无限暖暖超话' 的数量
#     """
#     if series.empty: return 0
#     # 转换为字符串并查找关键词
#     return series.astype(str).str.contains("无限暖暖超话", na=False).sum()

# def calculate_core_fan_count(series):
#     """
#     统计 fan_level 中包含 '铁粉专属'/'金粉专属'/'钻粉专属' 的数量
#     """
#     if series.empty: return 0
#     # 定义核心粉关键词正则
#     pattern = r"铁粉专属|金粉专属|钻粉专属"
#     return series.astype(str).str.contains(pattern, regex=True, na=False).sum()

# def calculate_original_ratio(series):
#     """
#     特征5: 原创微博占比
#     逻辑: 统计 source_post_id 为空 (即原创) 的比例
#     """
#     if series.empty: return 0.0
#     # 假设 NaN 表示原创，有值表示转发
#     original_count = series.isna().sum()
#     return original_count / (len(series) + 1e-5)


def merge_csv_files(path_or_folder,output_path, temp_save_name=None):
    """
    辅助函数：将文件夹下的所有CSV合并为一个DataFrame，或直接读取单文件
    """
    if os.path.isfile(path_or_folder):
        print(f"📖 读取单文件: {path_or_folder}")
        return path_or_folder # 直接返回路径给 monitor 读取
    
    if os.path.isdir(path_or_folder):
        all_files = sorted(glob.glob(os.path.join(path_or_folder, "*.csv")))
        if not all_files:
            print(f"⚠️ 警告: 文件夹为空 {path_or_folder}")
            return None
            
        print(f"🔄 正在合并 {len(all_files)} 个文件用于时序分析...")
        # 我们可以选择合并成一个临时大文件，这样节省内存
        df_list = []
        for f in all_files:
            # try:
                # 只读取必要的列以节省内存 (根据你的实际列名调整)
                # df_list.append(pd.read_csv(f, usecols=['created_at', 'text', ...])) 
            df_list.append(pd.read_csv(f))
            # except Exception as e:
            #     print(f"  ❌ 读取失败 {f}: {e}")
        
        full_df = pd.concat(df_list, ignore_index=True)
        
        # 保存临时大文件
        if temp_save_name:
            temp_path = os.path.join(output_path, temp_save_name)
            full_df.to_csv(temp_path, index=False, encoding='utf-8-sig')
            print(f"✅ 合并完成，暂存为: {temp_path}")
            return temp_path
    print(f"⚠️ 警告: 未能读取文件")

            
    return None
def conbine_post_comment(df_post_original,df_comment_original):
    df_post = df_post_original.copy()
    df_comment = df_comment_original.copy()

    if 'comment_content' in df_comment.columns: df_comment.rename(columns={'comment_content': 'text'}, inplace=True)
    # if 'post_id' in df_comment.columns: df_comment.rename(columns={'post_id': 'official_post_id'}, inplace=True)
    if 'user_id' in df_comment.columns: df_comment.rename(columns={'user_id': 'author_id'}, inplace=True)
    if 'post_date' in df_comment.columns: df_comment.rename(columns={'post_date': 'official_timestamp'}, inplace=True)
    if 'content' in df_post.columns: df_post.rename(columns={'content': 'text'}, inplace=True)

    df_comment['post_type'] = "comment"
    df_comment.drop(['comment_time_label', 'comment_likes', 'comment_replies'], axis=1, inplace = True, errors='ignore')
    df_post.drop(['reposts_count', 'comments_count', 'likes_count'], axis=1, inplace = True, errors='ignore')
    df_total = pd.concat([df_post,df_comment], axis=0, join='outer', ignore_index=True)

    return df_total,df_post,df_comment


# ========================================= 分解合成使用 ========================================
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

def perform_residual_diagnostics(series, label, output_dir=None):
    """
    通用的残差诊断与绘图函数
    用于替代 Step 1_5 的 analyze_garch_results 绘图部分
    以及 Step 1_6 的 analyze_residual_noise 和绘图部分
    """
    # 1. 基础统计
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 20: 
        return None
        
    stats_dict = {
        'mean': np.mean(series),
        'std': np.std(series),
        'skew': stats.skew(series),
        'kurt': stats.kurtosis(series)
    }

    # 2. 检验
    # 正态性
    jb_stat, jb_p = stats.jarque_bera(series)
    shapiro_stat, shapiro_p = stats.shapiro(series[:5000]) # 样本过大时截断
    stats_dict.update({'jb_p': jb_p, 'shapiro_p': shapiro_p})
    
    # 自相关 (Ljung-Box)
    lb_res = acorr_ljungbox(series, lags=[10], return_df=True)
    stats_dict['lb_p_value'] = lb_res['lb_pvalue'].iloc[0]
    
    # 3. 绘图 (满足您的画图需求)
    if output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 图1: 时序图
        axes[0,0].plot(series, alpha=0.7, linewidth=0.5)
        axes[0,0].set_title(f'{label} - 时序波动')
        
        # 图2: 直方图 + 拟合
        axes[0,1].hist(series, bins=50, density=True, alpha=0.6, color='steelblue')
        x = np.linspace(series.min(), series.max(), 100)
        axes[0,1].plot(x, stats.norm.pdf(x, 0, 1), 'r-', label='N(0,1)')
        axes[0,1].set_title(f'分布 (Skew={stats_dict["skew"]:.2f})')
        axes[0,1].legend()
        
        # 图3: Q-Q 图
        stats.probplot(series, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot')
        
        # 图4: ACF
        plot_acf(series, ax=axes[1,1], lags=40, alpha=0.05)
        axes[1,1].set_title('ACF (自相关)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{label}_诊断分析.png', dpi=100)
        plt.close()

    return stats_dict

