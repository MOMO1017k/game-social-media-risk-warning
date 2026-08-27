"""
TFT_utils.py
通用工具函数：IO、日志、节假日判断、评估指标等
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# ============================================================
# 配置与日志
# ============================================================

def load_config(config_path: str = "TFT_config.yaml") -> Dict:
    """
    加载YAML配置文件
    
    Parameters
    ----------
    config_path : str
        配置文件路径
        
    Returns
    -------
    Dict
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "TFT", 
                 level: int = logging.INFO,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志器
    
    Parameters
    ----------
    name : str
        日志器名称
    level : int
        日志级别
    log_file : str, optional
        日志文件路径
        
    Returns
    -------
    logging.Logger
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# 节假日与时间工具
# ============================================================

def get_chinese_holidays(year: int) -> set:
    """
    获取中国法定节假日（简化版本，可根据需要扩展）
    
    Parameters
    ----------
    year : int
        年份
        
    Returns
    -------
    set
        节假日日期集合
    """
    # 尝试使用chinese_calendar库
    try:
        import chinese_calendar as cc
        holidays = set()
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        current = start_date
        while current <= end_date:
            if cc.is_holiday(current):
                holidays.add(current)
            current += pd.Timedelta(days=1)
        return holidays
    except ImportError:
        pass
    
    # 尝试使用holidays库
    try:
        import holidays as hd
        cn_holidays = hd.China(years=year)
        return set(cn_holidays.keys())
    except ImportError:
        pass
    
    # 回退：使用固定日期的简化版本
    holidays = set()
    
    # 元旦
    holidays.add(date(year, 1, 1))
    
    # 春节（假设在1月下旬到2月中旬）
    for day in range(21, 28):
        holidays.add(date(year, 1, day))
    for day in range(1, 7):
        holidays.add(date(year, 2, day))
    
    # 清明节
    for day in range(4, 7):
        holidays.add(date(year, 4, day))
    
    # 劳动节
    for day in range(1, 6):
        holidays.add(date(year, 5, day))
    
    # 端午节
    for day in range(10, 13):
        holidays.add(date(year, 6, day))
    
    # 中秋节
    for day in range(15, 18):
        holidays.add(date(year, 9, day))
    
    # 国庆节
    for day in range(1, 8):
        holidays.add(date(year, 10, day))
    
    return holidays


def is_holiday(timestamp: Union[datetime, pd.Timestamp, date], 
               holidays_cache: Optional[Dict[int, set]] = None) -> bool:
    """
    判断给定日期是否为节假日
    """
    if isinstance(timestamp, pd.Timestamp):
        d = timestamp.date()
    elif isinstance(timestamp, datetime):
        d = timestamp.date()
    elif isinstance(timestamp, date):
        d = timestamp
    else:
        d = pd.to_datetime(timestamp).date()
    
    year = d.year
    
    if holidays_cache is None:
        holidays_cache = {}
    
    if year not in holidays_cache:
        holidays_cache[year] = get_chinese_holidays(year)
    
    return d in holidays_cache[year]


def get_time_slot(hour: int) -> int:
    """
    根据小时获取时段编码
    """
    if 0 <= hour <= 5:
        return 0  # 凌晨
    elif 6 <= hour <= 8:
        return 1  # 早晨
    elif 9 <= hour <= 11:
        return 2  # 上午
    elif 12 <= hour <= 17:
        return 3  # 下午
    elif 18 <= hour <= 19:
        return 4  # 傍晚
    elif 20 <= hour <= 22:
        return 5  # 晚上
    else:
        return 6  # 深夜


# def extract_time_features(df: pd.DataFrame, 
#                           time_column: Optional[str] = None) -> pd.DataFrame:
#     """
#     从时间索引或时间列提取时间特征
#     """
#     result = df.copy()
    
#     if time_column and time_column in df.columns:
#         timestamps = pd.to_datetime(df[time_column])
#     elif isinstance(df.index, pd.DatetimeIndex):
#         timestamps = df.index
#         time_column = None
#     else:
#         raise ValueError("无法找到时间信息")
    
#     years = timestamps.year.unique()
#     holidays_cache = {year: get_chinese_holidays(year) for year in years}
    
#     result['hour'] = timestamps.hour
#     result['dayofweek'] = timestamps.dayofweek
#     result['month'] = timestamps.month
#     result['day'] = timestamps.day
#     result['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
    
#     result['is_holiday'] = [
#         1 if is_holiday(ts, holidays_cache) else 0 
#         for ts in timestamps
#     ]
    
#     result['time_slot'] = result['hour'].apply(get_time_slot)
    
#     if time_column is None:
#         result['timestamp'] = timestamps
    
#     return result

def extract_time_features(df: pd.DataFrame, time_column: str = None) -> pd.DataFrame:
    """
    提取时间特征 (兼容 Series 和 DatetimeIndex)
    """
    df = df.copy()
    
    # 1. 获取时间序列数据
    if time_column and time_column in df.columns:
        timestamps = df[time_column]
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        # 尝试寻找常见的列名
        if 'timestamp' in df.columns:
            timestamps = df['timestamp']
        elif 'date' in df.columns:
            timestamps = df['date']
        else:
            raise ValueError("无法找到时间信息，请指定 time_column 或确保索引为 DatetimeIndex")

    # ================= [核心修复] =================
    # 将其转换为 DatetimeIndex 类型以便访问 .year, .month 等属性
    # 注意：这只是类型转换，完全支持多场景下的重复时间戳
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    # ============================================

    # 2. 提取特征
    result = df.copy()
    
    # 确保 result 中有一列叫 timestamp (方便后续处理)
    if 'timestamp' not in result.columns:
        result['timestamp'] = timestamps

    # 使用 DatetimeIndex 的属性进行提取 (现在安全了)
    years = timestamps.year.unique()
    
    holidays_cache = {year: get_chinese_holidays(year) for year in years}
    
    result['hour'] = timestamps.hour
    result['dayofweek'] = timestamps.dayofweek
    result['month'] = timestamps.month
    result['day'] = timestamps.day
    result['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
    
    result['is_holiday'] = [
        1 if is_holiday(ts, holidays_cache) else 0 
        for ts in timestamps
    ]
    
    result['time_slot'] = result['hour'].apply(get_time_slot)
    
    if time_column is None:
        result['timestamp'] = timestamps
    
    return result
# ============================================================
# 静态特征工具
# ============================================================

def extract_static_suffix(column_name: str, 
                          suffix_mapping: Dict[str, int]) -> Optional[int]:
    """
    从列名中提取静态属性编码
    """
    for suffix, code in suffix_mapping.items():
        if column_name.endswith(f"_{suffix}"):
            return code
    return None


def get_feature_base_name(column_name: str, 
                          suffixes: List[str]) -> str:
    """
    获取特征的基础名称（去掉后缀）
    """
    for suffix in suffixes:
        if column_name.endswith(f"_{suffix}"):
            return column_name[:-len(suffix)-1]
    return column_name


# ============================================================
# IO工具
# ============================================================

def ensure_dir(path: str) -> str:
    """
    确保目录存在，如果不存在则创建
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_pickle(obj, filepath: str):
    """保存pickle文件"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str):
    """加载pickle文件"""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================
# 数据验证工具
# ============================================================

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: Optional[List[str]] = None,
                       check_nan: bool = True,
                       check_inf: bool = True) -> Dict:
    """
    验证数据框的有效性
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            result['valid'] = False
            result['errors'].append(f"缺少必需列: {missing}")
    
    if check_nan:
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isna().sum()
            result['warnings'].append(f"包含NaN的列: {dict(nan_counts)}")
    
    if check_inf:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            result['warnings'].append(f"包含Inf的列: {inf_cols}")
    
    return result


# ============================================================
# Attention分析工具（新增）
# ============================================================

def compute_attention_entropy(attention_weights: np.ndarray) -> np.ndarray:
    """
    计算注意力权重的熵
    
    Parameters
    ----------
    attention_weights : np.ndarray
        注意力权重矩阵，形状为 (batch, heads, seq_len, seq_len) 或 (batch, seq_len, seq_len)
        
    Returns
    -------
    np.ndarray
        每个样本的熵值
    """
    # 确保是numpy数组
    if hasattr(attention_weights, 'cpu'):
        attention_weights = attention_weights.cpu().numpy()
    
    original_shape = attention_weights.shape
    
    # 处理不同维度的输入
    if len(original_shape) == 4:
        # (batch, heads, seq_len, seq_len) -> 对heads维度平均
        attention_weights = attention_weights.mean(axis=1)
    
    # 现在形状是 (batch, seq_len, seq_len)
    batch_size = attention_weights.shape[0]
    entropies = np.zeros(batch_size)
    
    for i in range(batch_size):
        # 对每个时间步的注意力分布计算熵，然后平均
        attn_matrix = attention_weights[i]  # (seq_len, seq_len)
        
        # 确保每行和为1（softmax后应该满足）
        attn_matrix = attn_matrix / (attn_matrix.sum(axis=-1, keepdims=True) + 1e-10)
        
        # 计算每行的熵
        row_entropies = entropy(attn_matrix + 1e-10, axis=-1)
        
        # 取平均熵
        entropies[i] = np.mean(row_entropies)
    
    return entropies


def compute_attention_js_divergence(attention_weights: np.ndarray,
                                     reference_distribution: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算注意力权重与参考分布之间的JS散度
    
    Parameters
    ----------
    attention_weights : np.ndarray
        注意力权重矩阵，形状为 (batch, heads, seq_len, seq_len) 或 (batch, seq_len, seq_len)
    reference_distribution : np.ndarray, optional
        参考分布，如果为None则使用均匀分布
        
    Returns
    -------
    np.ndarray
        每个样本的JS散度
    """
    if hasattr(attention_weights, 'cpu'):
        attention_weights = attention_weights.cpu().numpy()
    
    original_shape = attention_weights.shape
    
    if len(original_shape) == 4:
        attention_weights = attention_weights.mean(axis=1)
    
    batch_size, seq_len, _ = attention_weights.shape
    
    # 如果没有提供参考分布，使用均匀分布
    if reference_distribution is None:
        reference_distribution = np.ones(seq_len) / seq_len
    
    js_divergences = np.zeros(batch_size)
    
    for i in range(batch_size):
        attn_matrix = attention_weights[i]
        attn_matrix = attn_matrix / (attn_matrix.sum(axis=-1, keepdims=True) + 1e-10)
        
        # 计算每行与参考分布的JS散度
        row_js = []
        for row in attn_matrix:
            js = jensenshannon(row + 1e-10, reference_distribution + 1e-10)
            if not np.isnan(js):
                row_js.append(js)
        
        js_divergences[i] = np.mean(row_js) if row_js else 0.0
    
    return js_divergences


def compute_temporal_attention_divergence(attention_weights: np.ndarray,
                                           window_size: int = 10) -> np.ndarray:
    """
    计算时间窗口内注意力分布的变化（用于检测突变）
    
    Parameters
    ----------
    attention_weights : np.ndarray
        注意力权重序列
    window_size : int
        滑动窗口大小
        
    Returns
    -------
    np.ndarray
        注意力分布变化程度
    """
    if hasattr(attention_weights, 'cpu'):
        attention_weights = attention_weights.cpu().numpy()
    
    if len(attention_weights.shape) == 4:
        attention_weights = attention_weights.mean(axis=1)
    
    batch_size = attention_weights.shape[0]
    divergences = np.zeros(batch_size)
    
    if batch_size < window_size + 1:
        return divergences
    
    for i in range(window_size, batch_size):
        # 当前注意力分布（取最后一行，即对历史的关注）
        current_attn = attention_weights[i, -1, :]
        current_attn = current_attn / (current_attn.sum() + 1e-10)
        
        # 窗口内的平均注意力分布
        window_attn = attention_weights[i-window_size:i, -1, :].mean(axis=0)
        window_attn = window_attn / (window_attn.sum() + 1e-10)
        
        # 计算JS散度
        js = jensenshannon(current_attn + 1e-10, window_attn + 1e-10)
        divergences[i] = js if not np.isnan(js) else 0.0
    
    return divergences


# ============================================================
# 残差与预测区间分析工具（新增）
# ============================================================

def compute_prediction_residuals(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    计算预测残差及相关统计量
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值（中位数预测）
        
    Returns
    -------
    Dict[str, np.ndarray]
        包含各种残差指标
    """
    residuals = y_true - y_pred
    
    return {
        'residual': residuals,
        'abs_residual': np.abs(residuals),
        'squared_residual': residuals ** 2,
        'relative_residual': residuals / (np.abs(y_true) + 1e-10)
    }


def compute_interval_divergence(y_true: np.ndarray,
                                 quantile_low: np.ndarray,
                                 quantile_mid: np.ndarray,
                                 quantile_high: np.ndarray) -> Dict[str, np.ndarray]:
    """
    计算预测区间发散度
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    quantile_low : np.ndarray
        下分位数预测（如P10）
    quantile_mid : np.ndarray
        中位数预测（P50）
    quantile_high : np.ndarray
        上分位数预测（如P90）
        
    Returns
    -------
    Dict[str, np.ndarray]
        区间发散度指标
    """
    # 预测区间宽度
    interval_width = quantile_high - quantile_low
    
    # 归一化区间宽度
    normalized_width = interval_width / (np.abs(quantile_mid) + 1e-10)
    
    # 真实值是否在预测区间内
    in_interval = ((y_true >= quantile_low) & (y_true <= quantile_high)).astype(float)
    
    # 真实值偏离区间的程度
    below_low = np.maximum(0, quantile_low - y_true)
    above_high = np.maximum(0, y_true - quantile_high)
    deviation_from_interval = below_low + above_high
    
    # 真实值在区间内的相对位置 (0=下边界, 1=上边界)
    relative_position = (y_true - quantile_low) / (interval_width + 1e-10)
    relative_position = np.clip(relative_position, 0, 1)
    
    # 区间不对称性
    upper_range = quantile_high - quantile_mid
    lower_range = quantile_mid - quantile_low
    asymmetry = (upper_range - lower_range) / (interval_width + 1e-10)
    
    return {
        'interval_width': interval_width,
        'normalized_interval_width': normalized_width,
        'in_interval': in_interval,
        'deviation_from_interval': deviation_from_interval,
        'relative_position_in_interval': relative_position,
        'interval_asymmetry': asymmetry
    }


# ============================================================
# 模型配置生成工具（新增）
# ============================================================

def generate_model_configs(config: Dict) -> List[Dict]:
    """
    根据配置生成多个TFT模型配置
    
    Parameters
    ----------
    config : Dict
        主配置字典
        
    Returns
    -------
    List[Dict]
        模型配置列表
    """
    tft_config = config.get('tft', {})
    
    n_models = tft_config.get('n_models', 6)
    seeds = tft_config.get('random_seeds', list(range(42, 42 + n_models)))
    targets = tft_config.get('target_variables', [])
    encoder_lengths = tft_config.get('encoder_lengths', [96])
    
    model_configs = []
    model_id = 0
    
    # 生成所有组合
    for target in targets:
        for encoder_length in encoder_lengths:
            # 分配种子（循环使用）
            seed = seeds[model_id % len(seeds)]
            
            model_config = {
                'model_id': model_id,
                'model_name': f"tft_{target}_{encoder_length}_{seed}",
                'target_variable': target,
                'encoder_length': encoder_length,
                'decoder_length': tft_config.get('decoder_length', 1),
                'random_seed': seed,
                'quantiles': tft_config.get('quantiles', [0.1, 0.5, 0.9]),
                'hidden_size': tft_config.get('hidden_size', 64),
                'attention_head_size': tft_config.get('attention_head_size', 4),
                'num_attention_heads': tft_config.get('num_attention_heads', 4),
                'dropout': tft_config.get('dropout', 0.1),
                'hidden_continuous_size': tft_config.get('hidden_continuous_size', 32),
                'embedding_sizes': tft_config.get('embedding_sizes', {}),
                'training': tft_config.get('training', {})
            }
            
            model_configs.append(model_config)
            model_id += 1
    
    return model_configs


def set_seed(seed: int):
    """
    设置随机种子以确保可重复性
    
    Parameters
    ----------
    seed : int
        随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass