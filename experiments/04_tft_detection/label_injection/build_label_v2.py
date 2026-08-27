"""
异常注入配置生成器 - 基于step6真实数据分析结果优化
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

# ============================================================
# 1. 配置常量 - 基于真实数据分析
# ============================================================

@dataclass
class ChannelProfile:
    """通道特征画像 - 基于step6分析结果"""
    name: str
    # 主要驱动特征（按频率排序）
    primary_features: List[str]
    # 次要驱动特征
    secondary_features: List[str]
    # 持续时间分布 (窗口数, 15min/窗口)
    duration_p25: int  # P25
    duration_p50: int  # P50
    duration_p75: int  # P75
    duration_p95: int  # P95
    # 强度分布 (相对于P95的倍数)
    magnitude_low: float   # 温和异常
    magnitude_mid: float   # 中等异常
    magnitude_high: float  # 强异常
    # 形态权重
    morphology_weights: Dict[str, float] = field(default_factory=dict)


# 基于step6分析的通道画像
COMMENT_PROFILE = ChannelProfile(
    name="comment",
    primary_features=[
        "neg_ratio_comment",       # 184次，压倒性第一
        "total_volume_comment",    # 117次 (虽然不在驱动列表但comment的volume也重要)
        "semantic_shift_comment",  # 55次
    ],
    secondary_features=[
        "comp_ratio_comment",      # 37次
        "gini_comment",
        "senti_symbol_comment",
        "total_long_comment",
    ],
    duration_p25=4,   # 1.0h
    duration_p50=5,   # 1.25h
    duration_p75=7,   # 1.75h
    duration_p95=15,  # 3.75h
    magnitude_low=1.2,
    magnitude_mid=1.8,
    magnitude_high=2.5,
    morphology_weights={
        "gradual_recovery": 0.45,  # 降低：很多是假阳性
        "gradual_drift": 0.25,
        "volatility_burst": 0.10,
        "event_shock": 0.10,
        "sustained_shift": 0.10,
    }
)

POST_PROFILE = ChannelProfile(
    name="post",
    primary_features=[
        "vis_concentration_post",  # 72次
        "total_volume_post",       # 58次
        "neg_ratio_post",          # 58次
    ],
    secondary_features=[
        "semantic_shift_post",     # 51次
        "gini_post",               # 44次
        "retweet_ratio_post",      # 40次
        "comp_ratio_post",         # 37次
    ],
    duration_p25=5,   # 1.25h
    duration_p50=5,   # 1.25h  (post的中位数和comment接近但尾巴更长)
    duration_p75=10,  # 2.5h
    duration_p95=21,  # 5.25h
    magnitude_low=1.4,
    magnitude_mid=2.0,
    magnitude_high=2.8,
    morphology_weights={
        "gradual_drift": 0.30,
        "mixed": 0.25,
        "sustained_shift": 0.15,
        "gradual_recovery": 0.15,
        "volatility_burst": 0.15,
    }
)


# ============================================================
# 2. 注入方案生成器
# ============================================================

class RealisticInjectionGenerator:
    """
    基于真实数据形态分析的注入配置生成器
    
    核心设计原则：
    1. Comment通道：温和、短时、情感驱动
    2. Post通道：较强、持续、多特征驱动
    3. 整体异常比例控制在3-5%（真实数据的异常率）
    4. 形态分布匹配真实观测
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        seed: int = 42,
        freq_minutes: int = 15,
        logger: Optional[logging.Logger] = None,
    ):
        self.df = df
        self.feature_columns = feature_columns
        self.freq_minutes = freq_minutes
        self.rng = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger("InjectionGen")
        
        # 分类特征
        self.comment_features = [f for f in feature_columns if "comment" in f.lower()]
        self.post_features = [f for f in feature_columns if "post" in f.lower()]
        
        # 数据时间范围
        self.t_start = df.index.min()
        self.t_end = df.index.max()
        self.total_windows = len(df)
        
    def _classify_feature(self, feature: str) -> str:
        """判断特征属于哪个通道"""
        if "comment" in feature.lower():
            return "comment"
        elif "post" in feature.lower():
            return "post"
        return "both"
    
    def _sample_time(
        self, 
        avoid_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        margin_hours: float = 6.0,
    ) -> pd.Timestamp:
        """
        随机采样一个注入时间点，避免与已有注入重叠
        margin_hours: 距离数据边界的最小距离
        """
        margin = pd.Timedelta(hours=margin_hours)
        valid_start = self.t_start + margin
        valid_end = self.t_end - margin
        
        for _ in range(100):  # 最多尝试100次
            # 随机选一个index位置
            idx = self.rng.integers(0, self.total_windows)
            t = self.df.index[idx]
            
            if t < valid_start or t > valid_end:
                continue
                
            # 检查是否与已有注入重叠
            if avoid_ranges:
                overlap = False
                for (rs, re) in avoid_ranges:
                    buffer = pd.Timedelta(hours=3)  # 3小时缓冲
                    if rs - buffer <= t <= re + buffer:
                        overlap = True
                        break
                if overlap:
                    continue
            
            return t
        
        # 兜底：随机返回
        idx = self.rng.integers(0, self.total_windows)
        return self.df.index[idx]
    
    def _sample_duration(self, profile: ChannelProfile, intensity: str = "mid") -> int:
        """
        基于通道画像采样持续时间（窗口数）
        intensity: low/mid/high
        """
        if intensity == "low":
            # 短异常
            return int(self.rng.integers(profile.duration_p25, profile.duration_p50 + 1))
        elif intensity == "mid":
            # 中等异常
            return int(self.rng.integers(profile.duration_p50, profile.duration_p75 + 1))
        else:
            # 长异常
            return int(self.rng.integers(profile.duration_p75, profile.duration_p95 + 1))
    
    def _sample_magnitude(self, profile: ChannelProfile, intensity: str = "mid") -> float:
        """
        基于通道画像采样异常强度
        """
        if intensity == "low":
            base = profile.magnitude_low
            jitter = self.rng.uniform(-0.2, 0.3)
        elif intensity == "mid":
            base = profile.magnitude_mid
            jitter = self.rng.uniform(-0.3, 0.4)
        else:
            base = profile.magnitude_high
            jitter = self.rng.uniform(-0.3, 0.5)
        return max(0.8, base + jitter)
    
    def _sample_feature(self, profile: ChannelProfile, n: int = 1) -> List[str]:
        """
        按频率权重采样特征
        primary特征被选中的概率更高
        """
        # 构建候选池：primary权重3，secondary权重1
        candidates = []
        weights = []
        
        for f in profile.primary_features:
            if f in self.feature_columns:
                candidates.append(f)
                weights.append(3.0)
        
        for f in profile.secondary_features:
            if f in self.feature_columns:
                candidates.append(f)
                weights.append(1.0)
        
        if not candidates:
            # fallback
            channel_feats = (self.comment_features if profile.name == "comment" 
                           else self.post_features)
            if channel_feats:
                return list(self.rng.choice(channel_feats, size=min(n, len(channel_feats)), replace=False))
            return self.feature_columns[:n]
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        n = min(n, len(candidates))
        selected = self.rng.choice(candidates, size=n, replace=False, p=weights)
        return list(selected)

    # ========== 具体注入模式生成 ==========
    
    def _gen_gradual_drift(
        self, 
        profile: ChannelProfile, 
        intensity: str,
        avoid_ranges: List,
    ) -> Dict[str, Any]:
        """
        渐变漂移注入 - 对应真实的gradual_drift形态(23.5%)
        真实表现：数据缓慢偏移，持续1-4小时
        """
        feature = self._sample_feature(profile, n=1)[0]
        t_start = self._sample_time(avoid_ranges)
        duration = self._sample_duration(profile, intensity)
        magnitude = self._sample_magnitude(profile, intensity)
        
        # 漂移类型按真实形态分布
        drift_type = self.rng.choice(
            ["linear", "exponential", "sigmoid"],
            p=[0.4, 0.3, 0.3]
        )
        direction = self.rng.choice(
            ["positive", "negative"],
            p=[0.6, 0.4]  # 正向略多（真实数据中上升偏多）
        )
        
        t_end = t_start + pd.Timedelta(minutes=self.freq_minutes * duration)
        
        return {
            "type": "gradual_drift",
            "feature": feature,
            "start_time": str(t_start),
            "end_time": str(t_end),
            "drift_magnitude": magnitude,
            "drift_type": drift_type,
            "direction": direction,
            "_time_range": (t_start, t_end),
            "_profile": profile.name,
            "_intensity": intensity,
        }
    
    def _gen_event_shock(
        self,
        profile: ChannelProfile,
        intensity: str,
        avoid_ranges: List,
    ) -> Dict[str, Any]:
        """
        事件冲击注入 - 对应真实的gradual_recovery形态(59.9%)
        真实表现：快速上升后缓慢恢复，持续1-2小时
        
        关键洞察：真实的gradual_recovery本质是"冲击+衰减"
        用event_shock来模拟比用spike更准确
        """
        feature = self._sample_feature(profile, n=1)[0]
        t_event = self._sample_time(avoid_ranges)
        magnitude = self._sample_magnitude(profile, intensity)
        
        # 半衰期：基于真实持续时间
        # 真实median持续1.2h=5窗口，半衰期约2-3窗口可覆盖
        halflife = self.rng.integers(2, 6)
        max_duration = self._sample_duration(profile, intensity)
        
        direction = self.rng.choice(["up", "down"], p=[0.55, 0.45])
        
        t_end = t_event + pd.Timedelta(minutes=self.freq_minutes * max_duration)
        
        return {
            "type": "event_shock",
            "feature": feature,
            "event_time": str(t_event),
            "impact_magnitude": magnitude,
            "decay_halflife": int(halflife),
            "direction": direction,
            "max_duration_steps": int(max_duration),
            "_time_range": (t_event, t_end),
            "_profile": profile.name,
            "_intensity": intensity,
        }
    
    def _gen_volatility_burst(
        self,
        profile: ChannelProfile,
        intensity: str,
        avoid_ranges: List,
    ) -> Dict[str, Any]:
        """
        波动率爆发注入 - 对应真实的volatility_burst形态(6.5%)
        真实表现：方差突增，持续1-3小时
        """
        feature = self._sample_feature(profile, n=1)[0]
        t_start = self._sample_time(avoid_ranges)
        duration = self._sample_duration(profile, intensity)
        t_end = t_start + pd.Timedelta(minutes=self.freq_minutes * duration)
        
        # 真实波动率乘数：基于ratio_to_P95 ≈ 1.6-2.2
        vol_mult = self._sample_magnitude(profile, intensity) * 1.2  # 波动率需要略大
        vol_mult = min(vol_mult, 4.0)  # 上限4倍
        
        return {
            "type": "volatility_burst",
            "feature": feature,
            "start_time": str(t_start),
            "end_time": str(t_end),
            "volatility_multiplier": float(vol_mult),
            "method": "scale_deviation",  # 保留原始波形
            "_time_range": (t_start, t_end),
            "_profile": profile.name,
            "_intensity": intensity,
        }
    
    def _gen_sustained_shift(
        self,
        profile: ChannelProfile,
        intensity: str,
        avoid_ranges: List,
    ) -> Dict[str, Any]:
        """
        持续偏移注入 - 对应真实的sustained_shift形态(1.7%)
        用mean_changepoint实现，这是最确定的异常类型
        """
        feature = self._sample_feature(profile, n=1)[0]
        t_cp = self._sample_time(avoid_ranges)
        magnitude = self._sample_magnitude(profile, intensity) * 1.3  # 偏移要更显著
        
        # 过渡步数：0=突变，3-8=渐变
        transition = int(self.rng.choice([0, 0, 3, 5, 8], p=[0.3, 0.2, 0.2, 0.2, 0.1]))
        
        t_end = t_cp + pd.Timedelta(hours=24)  # 变点影响持续较长
        
        return {
            "type": "mean_changepoint",
            "feature": feature,
            "changepoint_time": str(t_cp),
            "mean_shift": float(magnitude),
            "transition_steps": transition,
            "_time_range": (t_cp, t_end),
            "_profile": profile.name,
            "_intensity": intensity,
        }
    
    def _gen_correlated_anomaly(
        self,
        profile: ChannelProfile,
        intensity: str,
        avoid_ranges: List,
    ) -> Dict[str, Any]:
        """
        多特征协同异常 - 对应真实的mixed形态(8.5%)
        真实表现：多个特征同时偏移
        """
        n_features = self.rng.integers(2, 4)  # 2-3个特征同时异常
        features = self._sample_feature(profile, n=n_features)
        
        # 如果是"both"通道，混合选取
        if self.rng.random() < 0.22:  # 22%概率是both通道
            comment_f = self._sample_feature(COMMENT_PROFILE, n=1)
            post_f = self._sample_feature(POST_PROFILE, n=1)
            features = comment_f + post_f
        
        t_start = self._sample_time(avoid_ranges)
        duration = self._sample_duration(profile, intensity)
        
        magnitudes = [float(self._sample_magnitude(profile, intensity)) for _ in features]
        
        t_end = t_start + pd.Timedelta(minutes=self.freq_minutes * duration)
        
        return {
            "type": "correlated_anomaly",
            "features": features,
            "timestamp": str(t_start),
            "magnitudes": magnitudes,
            "duration": int(duration),
            "mode": "add",
            "shared_latent": True,  # 共享衰减曲线更真实
            "_time_range": (t_start, t_end),
            "_profile": profile.name,
            "_intensity": intensity,
        }
    
    def _gen_level_shift(
        self,
        profile: ChannelProfile,
        intensity: str,
        avoid_ranges: List,
    ) -> Dict[str, Any]:
        """
        水平偏移 - 作为变点类型的补充
        """
        feature = self._sample_feature(profile, n=1)[0]
        t_start = self._sample_time(avoid_ranges)
        duration = self._sample_duration(profile, "high")  # 偏移通常持续较久
        t_end = t_start + pd.Timedelta(minutes=self.freq_minutes * duration)
        magnitude = self._sample_magnitude(profile, intensity)
        
        return {
            "type": "level_shift",
            "feature": feature,
            "start_time": str(t_start),
            "end_time": str(t_end),
            "shift_magnitude": float(magnitude),
            "_time_range": (t_start, t_end),
            "_profile": profile.name,
            "_intensity": intensity,
        }

    # ========== 主入口 ==========
    
    def generate_injection_plan(
        self,
        n_anomaly_points: int = 15,      # A类异常（异常点）数量
        n_changepoints: int = 3,          # C类异常（变点）数量
        comment_ratio: float = 0.45,      # comment通道占比
        post_ratio: float = 0.33,         # post通道占比
        # both_ratio自动 = 1 - comment - post = 0.22
    ) -> List[Dict[str, Any]]:
        """
        生成完整的注入方案
        
        设计依据：
        - 真实异常率约3-5%（267个异常段/32160窗口≈0.8%的段数，
          但每段平均5-10窗口，实际受影响窗口约5%）
        - 形态分布匹配真实观测
        - 通道比例匹配：comment 45%, post 33%, both 22%
        
        Args:
            n_anomaly_points: A类异常数量（建议15-25个，覆盖主要形态）
            n_changepoints: C类异常数量（建议2-4个，不宜过多）
            comment_ratio: comment通道的注入占比
            post_ratio: post通道的注入占比
        """
        configs = []
        avoid_ranges = []
        
        both_ratio = 1.0 - comment_ratio - post_ratio
        
        # =======================================
        # Part 1: A类异常（异常点）
        # =======================================
        # 形态分布（基于真实数据，调整后）：
        # - event_shock (模拟gradual_recovery): 40%
        # - gradual_drift (模拟gradual_drift): 25% 
        # - correlated_anomaly (模拟mixed): 15%
        # - volatility_burst: 15%
        # - spike (少量补充): 5%
        
        a_method_weights = {
            "event_shock": 0.40,       # 对应真实gradual_recovery
            "gradual_drift_short": 0.25,  # 短期gradual_drift作为A类
            "correlated_anomaly": 0.15,   # 对应mixed
            "volatility_burst": 0.15,     # 对应volatility_burst
            "spike": 0.05,                # 少量尖峰补充
        }
        
        a_methods = list(a_method_weights.keys())
        a_probs = list(a_method_weights.values())
        
        # 强度分布：大部分温和，少量强异常
        intensity_weights = {"low": 0.30, "mid": 0.50, "high": 0.20}
        intensities = list(intensity_weights.keys())
        i_probs = list(intensity_weights.values())
        
        self.logger.info(f"生成 {n_anomaly_points} 个A类异常...")
        
        for i in range(n_anomaly_points):
            # 选择通道
            channel_roll = self.rng.random()
            if channel_roll < comment_ratio:
                profile = COMMENT_PROFILE
            elif channel_roll < comment_ratio + post_ratio:
                profile = POST_PROFILE
            else:
                # both: 随机选一个profile，但特征会混合
                profile = self.rng.choice([COMMENT_PROFILE, POST_PROFILE])
            
            # 选择形态和强度
            method = self.rng.choice(a_methods, p=a_probs)
            intensity = self.rng.choice(intensities, p=i_probs)
            
            # 生成配置
            try:
                if method == "event_shock":
                    cfg = self._gen_event_shock(profile, intensity, avoid_ranges)
                elif method == "gradual_drift_short":
                    cfg = self._gen_gradual_drift(profile, intensity, avoid_ranges)
                    # 短期drift作为A类：限制持续时间
                    t_start = pd.to_datetime(cfg["start_time"])
                    max_dur = min(
                        self._sample_duration(profile, "low"),  # 使用low强度的持续时间
                        8  # 最多8个窗口=2小时
                    )
                    cfg["end_time"] = str(t_start + pd.Timedelta(minutes=self.freq_minutes * max_dur))
                    cfg["_time_range"] = (t_start, pd.to_datetime(cfg["end_time"]))
                elif method == "correlated_anomaly":
                    cfg = self._gen_correlated_anomaly(profile, intensity, avoid_ranges)
                elif method == "volatility_burst":
                    cfg = self._gen_volatility_burst(profile, intensity, avoid_ranges)
                elif method == "spike":
                    # 基于真实数据的spike：温和、短时
                    feature = self._sample_feature(profile, n=1)[0]
                    t_spike = self._sample_time(avoid_ranges)
                    mag = self._sample_magnitude(profile, intensity)
                    duration = self.rng.integers(2, 5)  # 2-4窗口
                    t_end = t_spike + pd.Timedelta(minutes=self.freq_minutes * duration)
                    cfg = {
                        "type": "positive_spike",
                        "feature": feature,
                        "timestamp": str(t_spike),
                        "magnitude": float(mag),
                        "duration": int(duration),
                        "mode": "add",  # 叠加而非覆盖，更自然
                        "_time_range": (t_spike, t_end),
                        "_profile": profile.name,
                        "_intensity": intensity,
                    }
                else:
                    continue
                
                # 记录时间范围，避免后续重叠
                if "_time_range" in cfg:
                    avoid_ranges.append(cfg["_time_range"])
                
                configs.append(cfg)
                
            except Exception as e:
                self.logger.warning(f"A类异常 #{i} 生成失败: {e}")
                continue
        
        # =======================================
        # Part 2: C类异常（变点）
        # =======================================
        # 变点类型分布：
        # - mean_changepoint (sustained_shift): 40%
        # - level_shift: 30%  
        # - gradual_drift_long (长期漂移): 30%
        
        c_method_weights = {
            "mean_changepoint": 0.40,
            "level_shift": 0.30,
            "gradual_drift_long": 0.30,
        }
        
        c_methods = list(c_method_weights.keys())
        c_probs = list(c_method_weights.values())
        
        self.logger.info(f"生成 {n_changepoints} 个C类异常...")
        
        for i in range(n_changepoints):
            # 变点的通道分布
            channel_roll = self.rng.random()
            if channel_roll < 0.40:
                profile = POST_PROFILE   # post通道变点占比更高
            elif channel_roll < 0.75:
                profile = COMMENT_PROFILE
            else:
                profile = self.rng.choice([COMMENT_PROFILE, POST_PROFILE])
            
            method = self.rng.choice(c_methods, p=c_probs)
            # 变点通常是中-高强度
            intensity = self.rng.choice(["mid", "high"], p=[0.6, 0.4])
            
            try:
                if method == "mean_changepoint":
                    cfg = self._gen_sustained_shift(profile, intensity, avoid_ranges)
                elif method == "level_shift":
                    cfg = self._gen_level_shift(profile, intensity, avoid_ranges)
                elif method == "gradual_drift_long":
                    cfg = self._gen_gradual_drift(profile, intensity, avoid_ranges)
                    # 长期drift作为C类：延长持续时间
                    t_start = pd.to_datetime(cfg["start_time"])
                    dur = self._sample_duration(profile, "high")
                    dur = max(dur, 12)  # 至少12窗口=3小时
                    cfg["end_time"] = str(t_start + pd.Timedelta(minutes=self.freq_minutes * dur))
                    cfg["_time_range"] = (t_start, pd.to_datetime(cfg["end_time"]))
                else:
                    continue
                
                if "_time_range" in cfg:
                    avoid_ranges.append(cfg["_time_range"])
                
                configs.append(cfg)
                
            except Exception as e:
                self.logger.warning(f"C类异常 #{i} 生成失败: {e}")
                continue
        
        # =======================================
        # Part 3: 清理内部标记字段
        # =======================================
        clean_configs = []
        for cfg in configs:
            clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
            clean_configs.append(clean_cfg)
        
        self.logger.info(
            f"注入方案生成完成: "
            f"A类={sum(1 for c in configs if c.get('type') not in ('mean_changepoint', 'level_shift') or (c.get('type') == 'gradual_drift' and c.get('_intensity') != 'high'))}个, "
            f"C类={sum(1 for c in configs if c.get('type') in ('mean_changepoint', 'level_shift') or (c.get('type') == 'gradual_drift' and c.get('_intensity') == 'high'))}个"
        )
        
        return clean_configs


# ============================================================
# 3. Pipeline集成
# ============================================================

def run_realistic_injection_pipeline(
    df_clean: pd.DataFrame,
    feature_columns: List[str],
    n_anomaly_points: int = 18,
    n_changepoints: int = 3,
    seed: int = 42,
    freq_minutes: int = 15,
    t_official_resp: Optional[pd.Timestamp] = None,
    cp_drift_duration_days: float = 2.0,
) -> Dict[str, Any]:
    """
    端到端的真实感异常注入Pipeline
    
    Args:
        df_clean: 干净数据
        feature_columns: 特征列名
        n_anomaly_points: A类异常数量
        n_changepoints: C类异常数量
        seed: 随机种子
        freq_minutes: 采样频率
        t_official_resp: 官方响应时间（可选）
        cp_drift_duration_days: 变点漂移态持续天数
    
    Returns:
        dict: {
            "data": 注入后的DataFrame,
            "labels": 标签DataFrame,
            "records": 注入记录,
            "injection_configs": 使用的注入配置,
            "summary": 摘要统计
        }
    """
    logger = logging.getLogger("RealisticInjection")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    
    logger.info("=" * 60)
    logger.info("开始真实感异常注入Pipeline")
    logger.info(f"数据规模: {len(df_clean)}行, {len(feature_columns)}列")
    logger.info(f"注入计划: A类={n_anomaly_points}, C类={n_changepoints}")
    logger.info("=" * 60)
    
    # Step 1: 生成注入方案
    generator = RealisticInjectionGenerator(
        df=df_clean,
        feature_columns=feature_columns,
        seed=seed,
        freq_minutes=freq_minutes,
        logger=logger,
    )
    
    injection_configs = generator.generate_injection_plan(
        n_anomaly_points=n_anomaly_points,
        n_changepoints=n_changepoints,
        comment_ratio=0.45,
        post_ratio=0.33,
    )
    
    # Step 2: 执行注入
    from build_label import create_injection_pipeline, PipelineConfig
    
    pipeline_config = PipelineConfig(
        freq_minutes=freq_minutes,
        seed=seed,
        label_mode="per_feature",
        include_records=True,
    )
    
    pipeline = InjectionPipeline(pipeline_config)
    
    result = pipeline.run(
        df=df_clean,
        injection_configs=injection_configs,
        fit_baseline=True,
        t_official_resp=t_official_resp,
    )
    
    # Step 3: 统计摘要
    labels = result["labels"]
    summary = {
        "total_points": len(labels),
        "n_anomaly_A": int((labels["label"] == "A").sum()),
        "n_changepoint_C": int((labels["label"] == "C").sum()),
        "n_normal_N": int((labels["label"] == "N").sum()),
        "anomaly_ratio_A": float((labels["label"] == "A").mean()),
        "changepoint_ratio_C": float((labels["label"] == "C").mean()),
        "injection_configs_count": len(injection_configs),
    }
    
    logger.info("=" * 60)
    logger.info("注入完成 - 摘要:")
    logger.info(f"  N (正常):  {summary['n_normal_N']} ({1-summary['anomaly_ratio_A']-summary['changepoint_ratio_C']:.1%})")
    logger.info(f"  A (异常):  {summary['n_anomaly_A']} ({summary['anomaly_ratio_A']:.1%})")
    logger.info(f"  C (变点):  {summary['n_changepoint_C']} ({summary['changepoint_ratio_C']:.1%})")
    logger.info("=" * 60)
    
    result["injection_configs"] = injection_configs
    result["summary"] = summary
    
    return result

