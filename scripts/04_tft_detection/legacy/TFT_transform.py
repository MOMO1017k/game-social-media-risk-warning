"""
TFT_transform.py
数据预处理、特征工程、格式转换
20260218: 增加时间计算的全局基准，保证训练和推理的一致性
"""

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, boxcox
from scipy.special import logit
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from TFT_utils import (
    load_config, setup_logger, 
    extract_time_features, extract_static_suffix,
    get_feature_base_name, validate_dataframe
)


# ============================================================
# 变换函数
# ============================================================

def safe_log1p(x: np.ndarray) -> np.ndarray:
    """
    安全的log1p变换
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
        
    Returns
    -------
    np.ndarray
        变换后的数组
    """
    x = np.array(x, dtype=float)
    x = np.where(x < 0, 0, x)
    x = np.where(np.isinf(x), np.nan, x)
    return np.log1p(x)


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    """
    安全的平方根变换
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
        
    Returns
    -------
    np.ndarray
        变换后的数组
    """
    x = np.array(x, dtype=float)
    x = np.where(x < 0, 0, x)
    x = np.where(np.isinf(x), np.nan, x)
    return np.sqrt(x)


def safe_logit(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    安全的Logit变换（带边界保护）
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
    epsilon : float
        边界epsilon值
        
    Returns
    -------
    np.ndarray
        变换后的数组
    """
    x = np.array(x, dtype=float)
    x = np.clip(x, epsilon, 1 - epsilon)
    x = np.where(np.isnan(x), 0.5, x)
    x = np.where(np.isinf(x), 0.5, x)
    return logit(x)


def safe_boxcox(x: np.ndarray, 
                shift: float = 1e-6,
                lambda_param: Optional[float] = None) -> Tuple[np.ndarray, Optional[float]]:
    """
    安全的Box-Cox变换
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
    shift : float
        平移量
    lambda_param : float, optional
        Box-Cox的lambda参数，如果提供则使用该值，否则自动拟合
        
    Returns
    -------
    Tuple[np.ndarray, Optional[float]]
        (变换后的数组, lambda参数)
    """
    x = np.array(x, dtype=float)
    valid_mask = ~np.isnan(x) & ~np.isinf(x)
    
    if valid_mask.sum() < 30:
        return x, None
    
    x_valid = x[valid_mask]
    x_positive = x_valid - x_valid.min() + shift
    
    try:
        if lambda_param is None:
            transformed, lambda_opt = boxcox(x_positive)
        else:
            lambda_opt = lambda_param
            if lambda_opt == 0:
                transformed = np.log(x_positive)
            else:
                transformed = (np.power(x_positive, lambda_opt) - 1) / lambda_opt
        
        # 应用到全部数据
        x_all_positive = x - x[valid_mask].min() + shift
        result = np.full_like(x, np.nan)
        
        if lambda_opt == 0:
            result = np.log(x_all_positive)
        else:
            result = (np.power(x_all_positive, lambda_opt) - 1) / lambda_opt
        
        result[~valid_mask] = np.nan
        return result, lambda_opt
    except Exception as e:
        print(f"Box-Cox失败: {e}")
        return x, None


def safe_yeo_johnson(x: np.ndarray,
                     lambda_param: Optional[float] = None) -> Tuple[np.ndarray, Optional[float]]:
    """
    Yeo-Johnson变换（支持负值和零值）
    
    Parameters
    ----------
    x : np.ndarray
        输入数组
    lambda_param : float, optional
        Yeo-Johnson的lambda参数，如果提供则使用该值，否则自动拟合
        
    Returns
    -------
    Tuple[np.ndarray, Optional[float]]
        (变换后的数组, lambda参数)
    """
    x = np.array(x, dtype=float)
    valid_mask = ~np.isnan(x) & ~np.isinf(x)
    
    if valid_mask.sum() < 30:
        return x, None
    
    try:
        x_valid = x[valid_mask]
        
        if lambda_param is None:
            transformed, lambda_opt = yeojohnson(x_valid)
        else:
            lambda_opt = lambda_param
            # 手动应用Yeo-Johnson变换
            transformed = _apply_yeo_johnson(x_valid, lambda_opt)
        
        result = np.full_like(x, np.nan)
        result[valid_mask] = transformed
        
        return result, lambda_opt
    except Exception as e:
        print(f"Yeo-Johnson失败: {e}")
        return x, None


def _apply_yeo_johnson(x: np.ndarray, lmbda: float) -> np.ndarray:
    """
    手动应用Yeo-Johnson变换（用于使用已拟合的lambda）
    """
    result = np.zeros_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    if lmbda == 0:
        result[pos_mask] = np.log1p(x[pos_mask])
    else:
        result[pos_mask] = (np.power(x[pos_mask] + 1, lmbda) - 1) / lmbda
    
    if lmbda == 2:
        result[neg_mask] = -np.log1p(-x[neg_mask])
    else:
        result[neg_mask] = -(np.power(-x[neg_mask] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    
    return result


# ============================================================
# 变换器类
# ============================================================

@dataclass
class TransformParams:
    """存储变换参数"""
    method: str
    lambda_param: Optional[float] = None
    min_value: Optional[float] = None
    shift: float = 1e-6


class FeatureTransformer:
    """
    特征变换器：对观测时变变量进行变换
    
    Attributes
    ----------
    transform_info : pd.DataFrame
        变换信息表，包含'特征'和'变换方法'列
    skip_transforms : List[str]
        跳过的变换方法（如standardize，因为TFT自带标准化）
    fitted_params : Dict[str, TransformParams]
        拟合后的变换参数
    """
    
    def __init__(self, 
                 transform_info: pd.DataFrame,
                 skip_transforms: Optional[List[str]] = None):
        """
        初始化变换器
        
        Parameters
        ----------
        transform_info : pd.DataFrame
            变换信息表
        skip_transforms : List[str], optional
            跳过的变换方法列表
        """
        self.transform_info = transform_info
        self.skip_transforms = skip_transforms or ['standardize']
        self.fitted_params: Dict[str, TransformParams] = {}
        self._is_fitted = False
        
        # 构建变换映射
        self._transform_map = self._build_transform_map()
    
    def _build_transform_map(self) -> Dict[str, str]:
        """构建特征到变换方法的映射"""
        # 假设transform_info有两列：'特征' 和 '变换方法'
        col_feature = self.transform_info.columns[0]
        col_method = self.transform_info.columns[1]
        
        return dict(zip(
            self.transform_info[col_feature],
            self.transform_info[col_method]
        ))
    
    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        """
        拟合变换参数
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        self
        """
        for feature, method in self._transform_map.items():
            if feature not in df.columns:
                continue
            
            if method in self.skip_transforms:
                # 跳过的变换，不存储参数
                self.fitted_params[feature] = TransformParams(method=method)
                continue
            
            x = df[feature].values
            
            if method == 'yeo_johnson':
                _, lambda_param = safe_yeo_johnson(x)
                self.fitted_params[feature] = TransformParams(
                    method=method,
                    lambda_param=lambda_param
                )
            elif method == 'boxcox':
                valid_mask = ~np.isnan(x) & ~np.isinf(x)
                min_value = x[valid_mask].min() if valid_mask.any() else 0
                _, lambda_param = safe_boxcox(x)
                self.fitted_params[feature] = TransformParams(
                    method=method,
                    lambda_param=lambda_param,
                    min_value=min_value
                )
            elif method in ['log1p', 'sqrt', 'logit']:
                self.fitted_params[feature] = TransformParams(method=method)
            else:
                # 未知方法，不做变换
                self.fitted_params[feature] = TransformParams(method='none')
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用变换
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        pd.DataFrame
            变换后的数据
        """
        if not self._is_fitted:
            raise RuntimeError("变换器尚未拟合，请先调用fit()方法")
        
        result = df.copy()
        
        for feature, params in self.fitted_params.items():
            if feature not in result.columns:
                continue
            
            method = params.method
            
            if method in self.skip_transforms or method == 'none':
                continue
            
            x = result[feature].values
            
            if method == 'log1p':
                result[feature] = safe_log1p(x)
            elif method == 'sqrt':
                result[feature] = safe_sqrt(x)
            elif method == 'logit':
                result[feature] = safe_logit(x)
            elif method == 'yeo_johnson':
                transformed, _ = safe_yeo_johnson(x, params.lambda_param)
                result[feature] = transformed
            elif method == 'boxcox':
                transformed, _ = safe_boxcox(x, lambda_param=params.lambda_param)
                result[feature] = transformed
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并变换
        
        Parameters
        ----------
        df : pd.DataFrame
            输入数据
            
        Returns
        -------
        pd.DataFrame
            变换后的数据
        """
        self.fit(df)
        return self.transform(df)
    
    def get_params(self) -> Dict[str, TransformParams]:
        """获取拟合的参数"""
        return self.fitted_params.copy()


# ============================================================
# 运营事件处理
# ============================================================

class EventProcessor:
    """
    运营事件处理器
    
    Attributes
    ----------
    event_mapping : Dict[str, int]
        事件到编码的映射
    """
    
    def __init__(self):
        self.event_mapping: Dict[str, int] = {}
        self._is_fitted = False
    
    def fit(self, df_official: pd.DataFrame) -> 'EventProcessor':
        """
        拟合事件编码
        
        Parameters
        ----------
        df_official : pd.DataFrame
            运营日历数据
            
        Returns
        -------
        self
        """
        # 获取所有唯一事件
        events = df_official['event'].dropna().unique()
        
        # 创建编码映射（0为无事件）
        self.event_mapping = {'none': 0}
        for i, event in enumerate(sorted(events), start=1):
            self.event_mapping[event] = i
        
        self._is_fitted = True
        return self
    
    def transform(self, 
                  timestamps: pd.DatetimeIndex,
                  df_official: pd.DataFrame) -> pd.Series:
        """
        将时间戳映射到事件编码
        
        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            时间戳序列
        df_official : pd.DataFrame
            运营日历数据
            
        Returns
        -------
        pd.Series
            事件编码序列
        """
        if not self._is_fitted:
            raise RuntimeError("处理器尚未拟合，请先调用fit()方法")
        
        # 将df_official的timestamp转换为日期
        df_official = df_official.copy()
        df_official['timestamp'] = pd.to_datetime(df_official['timestamp'])
        df_official['date'] = df_official['timestamp'].dt.date
        
        # 创建日期到事件的映射
        date_to_event = dict(zip(df_official['date'], df_official['event']))
        
        # 映射时间戳到事件编码
        result = []
        for ts in timestamps:
            d = ts.date() if hasattr(ts, 'date') else pd.to_datetime(ts).date()
            event = date_to_event.get(d, 'none')
            code = self.event_mapping.get(event, 0)
            result.append(code)
        
        return pd.Series(result, index=timestamps, name='event_code')
    
    def fit_transform(self,
                      timestamps: pd.DatetimeIndex,
                      df_official: pd.DataFrame) -> pd.Series:
        """拟合并变换"""
        self.fit(df_official)
        return self.transform(timestamps, df_official)
    
    def get_num_events(self) -> int:
        """获取事件类别数"""
        return len(self.event_mapping)
    
    def get_event_mapping(self) -> Dict[str, int]:
        """获取事件映射"""
        return self.event_mapping.copy()


# ============================================================
# TFT数据处理器
# ============================================================

@dataclass
class TFTDataset:
    """
    TFT数据集容器
    
    Attributes
    ----------
    data : pd.DataFrame
        处理后的数据
    static_categoricals : List[str]
        静态类别变量列名
    static_reals : List[str]
        静态实数变量列名
    time_varying_known_categoricals : List[str]
        时变已知类别变量列名
    time_varying_known_reals : List[str]
        时变已知实数变量列名
    time_varying_unknown_reals : List[str]
        时变未知实数变量列名（观测变量）
    feature_to_static : Dict[str, int]
        特征到静态属性的映射
    """
    data: pd.DataFrame
    static_categoricals: List[str] = field(default_factory=list)
    static_reals: List[str] = field(default_factory=list)
    time_varying_known_categoricals: List[str] = field(default_factory=list)
    time_varying_known_reals: List[str] = field(default_factory=list)
    time_varying_unknown_reals: List[str] = field(default_factory=list)
    feature_to_static: Dict[str, int] = field(default_factory=dict)


class TFTDataProcessor:
    """
    TFT数据处理器：将原始数据转换为TFT模型需要的格式
    
    处理流程：
    1. 对观测时变变量进行变换
    2. 从变量名提取静态属性
    3. 处理可预知时变变量（时间特征+运营事件+节假日）
    4. 组织成TFT需要的格式
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化处理器
        
        Parameters
        ----------
        config : Dict, optional
            配置字典，如果为None则使用默认配置
        """
        self.config = config or {}
        self.logger = setup_logger("TFTDataProcessor")
        
        # 子处理器
        self.feature_transformer: Optional[FeatureTransformer] = None
        self.event_processor: Optional[EventProcessor] = None
        
        # 配置项
        self.skip_transforms = self.config.get('data', {}).get(
            'skip_transforms', ['standardize']
        )
        self.suffix_mapping = self.config.get('data', {}).get(
            'static_suffix_mapping', {'post': 0, 'comment': 1}
        )
        
        self._is_fitted = False

        # [20260218新增] 全局最小时间，用于固定 time_idx 的基准
        self.global_min_time = None
    
    def fit(self,
            df_synthetic: pd.DataFrame,
            transform_info: pd.DataFrame,
            df_official: pd.DataFrame) :
        # -> 'TFTDataProcessor':
        """
        拟合数据处理器
        
        Parameters
        ----------
        df_synthetic : pd.DataFrame
            观测时变变量数据
        transform_info : pd.DataFrame
            变换信息表
        df_official : pd.DataFrame
            运营日历数据
            
        Returns
        -------
        self
        """
        self.logger.info("开始拟合数据处理器...")
        
        # 1. 拟合特征变换器
        self.logger.info("拟合特征变换器...")
        self.feature_transformer = FeatureTransformer(
            transform_info=transform_info,
            skip_transforms=self.skip_transforms
        )
        self.feature_transformer.fit(df_synthetic)

        
        # 2. 拟合事件处理器
        self.logger.info("拟合事件处理器...")
        self.event_processor = EventProcessor()
        self.event_processor.fit(df_official)
        
        # [20260218新增] 记录训练集的全局最小时间
        self.global_min_time = df_synthetic['timestamp'].min()
        
        self._is_fitted = True
        self.logger.info("数据处理器拟合完成")
    
        return self
    
    def transform(self,
                df_synthetic: pd.DataFrame,
                df_official: pd.DataFrame) :
    # -> TFTDataset:
        """
        转换数据
        
        Parameters
        ----------
        df_synthetic : pd.DataFrame
            观测时变变量数据
        df_official : pd.DataFrame
            运营日历数据
            
        Returns
        -------
        TFTDataset
            TFT数据集
        """
        if not self._is_fitted:
            raise RuntimeError("处理器尚未拟合，请先调用fit()方法")
        
        self.logger.info("开始数据转换...")

        
        # 确保转换为 datetime 格式
        df_synthetic['timestamp'] = pd.to_datetime(df_synthetic['timestamp'])
        df_official['timestamp'] = pd.to_datetime(df_official['timestamp'])



        # 1. 应用特征变换
        self.logger.info("应用特征变换...")
        df_transformed = self.feature_transformer.transform(df_synthetic)
        
        # 2. 提取静态属性
        self.logger.info("提取静态属性...")
        feature_to_static = {}
        for col in df_transformed.columns:
            static_code = extract_static_suffix(col, self.suffix_mapping)
            if static_code is not None:
                feature_to_static[col] = static_code
        
        # 3. 提取时间特征
        self.logger.info("提取时间特征...")
        df_with_time = extract_time_features(df_transformed,'timestamp')
        
        # 4. 添加事件编码
        self.logger.info("添加事件编码...")
        timestamps = df_with_time.index if isinstance(df_with_time.index, pd.DatetimeIndex) \
                    else pd.to_datetime(df_with_time['timestamp'])
        event_codes = self.event_processor.transform(timestamps, df_official)
        df_with_time['event_code'] = event_codes.values
        
        # 5. 添加时间索引列（用于TFT的time_idx）-20260218修改：使用全局最小时间作为基准，保证训练和推理的一致性
        # 如果是训练阶段，global_min_time 已经在 fit 里设置好了
        # 如果是推理阶段，global_min_time 是从 pickle 加载的训练集基准
        base_time = getattr(self, 'global_min_time', df_synthetic['timestamp'].min())
        df_with_time['time_idx'] = ((df_with_time['timestamp'] - base_time).dt.total_seconds() / 900).astype(int)
        # min_time = df_with_time['timestamp'].min()
        # df_with_time['time_idx'] = ((df_with_time['timestamp'] - min_time).dt.total_seconds() / 900).astype(int)
        # df_with_time['time_idx'] = np.arange(len(df_with_time))
        
        # 6. 添加group_id（单一时间序列）
        # 如果数据中有 scenario_id，使用它作为分组依据

        # df_with_time['group_id'] = 0
        if 'scenario_id' in df_with_time.columns:
            self.logger.info("检测到 scenario_id，将其用作序列区分 (group_id)")
            df_with_time['group_id'] = df_with_time['scenario_id'].astype(str)
        else:
            # 如果是真实推理数据，可能没有 scenario_id，或者只有一个场景
            # 我们给它一个默认值，保证代码运行
            self.logger.warning("未检测到 scenario_id，默认使用单序列模式 (group_id='0')")
            df_with_time['group_id'] = "0"
        
        # 7. 组织特征分组
        observed_reals = list(df_synthetic.columns)  # 原始观测变量
        if 'scenario_id' in observed_reals:
            observed_reals.remove('scenario_id')
            
        
        known_categoricals = ['hour', 'dayofweek', 'month', 'is_weekend', 
                            'is_holiday', 'time_slot', 'event_code']
        # known_reals = []  # 可以添加其他已知实数变量
        known_reals = ['time_idx']
        
        static_categoricals = []
        static_reals = []
        
        # 8. 创建数据集对象
        dataset = TFTDataset(
            data=df_with_time,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=known_categoricals,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=observed_reals,
            feature_to_static=feature_to_static
        )
        
        self.logger.info(f"数据转换完成，形状: {df_with_time.shape}")
        self.logger.info(f"  - 静态类别变量: {static_categoricals}")
        self.logger.info(f"  - 时变已知类别变量: {known_categoricals}")
        self.logger.info(f"  - 时变未知实数变量: {len(observed_reals)}个")
        
        return dataset
    
    def fit_transform(self,
                    df_synthetic: pd.DataFrame,
                    transform_info: pd.DataFrame,
                    df_official: pd.DataFrame) :
    # -> TFTDataset:
        """
        拟合并转换数据
        
        Parameters
        ----------
        df_synthetic : pd.DataFrame
            观测时变变量数据
        transform_info : pd.DataFrame
            变换信息表
        df_official : pd.DataFrame
            运营日历数据
            
        Returns
        -------
        TFTDataset
            TFT数据集
        """
        self.fit(df_synthetic, transform_info, df_official)
        return self.transform(df_synthetic, df_official)
    
    def get_feature_info(self) -> Dict:
        """
        获取特征信息
        
        Returns
        -------
        Dict
            特征信息字典
        """
        if self.feature_transformer is None:
            return {}
        
        return {
            'transform_params': self.feature_transformer.get_params(),
            'event_mapping': self.event_processor.get_event_mapping() if self.event_processor else {},
            'suffix_mapping': self.suffix_mapping
        }


# ============================================================
# 便捷函数
# ============================================================

def prepare_tft_data(df_synthetic: pd.DataFrame,
                     transform_info: pd.DataFrame,
                     df_official: pd.DataFrame,
                     config: Optional[Dict] = None) -> TFTDataset:
    """
    便捷函数：一步完成TFT数据准备
    
    Parameters
    ----------
    df_synthetic : pd.DataFrame
        观测时变变量数据
    transform_info : pd.DataFrame
        变换信息表
    df_official : pd.DataFrame
        运营日历数据
    config : Dict, optional
        配置字典
        
    Returns
    -------
    TFTDataset
        TFT数据集
    """
    processor = TFTDataProcessor(config)
    return processor.fit_transform(df_synthetic, transform_info, df_official)