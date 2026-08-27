"""论文第 4.5.4 节对应的冷启动统计阈值分类器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ThresholdConfig:
    """论文表 6.2 与在线实验设置中的检测参数。"""

    point_k: float = 5.0
    point_min_triggers: int = 4
    point_roll_window: int = 8
    collective_k: float = 3.0
    collective_volatility_multiplier: float = 1.5
    collective_density_threshold: float = 0.95
    collective_window: int = 288
    collective_min_periods: int = 96
    changepoint_k: float = 2.0
    changepoint_window: int = 2688
    changepoint_min_periods: int = 288
    changepoint_attribution_lag: int = 1344


class StatisticalThresholdClassifier:
    """输出 N、PA、CA 三类标签，并单独输出 CP（变点）布尔标记。

    这是对 Notebook ``StatisticalClassifier7`` 的论文命名校正版：原代码用
    ``AP`` 表示点异常、用 ``CP`` 表示集体异常，同时另用
    ``is_baseline_drift`` 表示真正的变点，容易与论文定义混淆。
    """

    POINT_SIGNAL_TYPES = (
        "residual_ratio",
        "vsn_rank_shift",
        "att_kl",
        "vsn_js",
    )
    MODEL_TARGETS = ("comment_pc1", "post_pc2")
    COLLECTIVE_SIGNAL_TYPES = ("vsn_rank_shift", "att_kl")

    STD_FLOORS = {
        "vsn_rank_shift": 0.05,
        "att_kl": 0.10,
    }

    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()
        self.point_thresholds: dict[str, float] = {}
        self.static_z_params: dict[str, dict[str, float]] = {}
        self.collective_level_threshold: float | None = None
        self.collective_volatility_threshold: float | None = None
        self.changepoint_threshold: float | None = None
        self._point_relax_factor = 1.0
        self._calibrated = False

    @property
    def point_features(self) -> list[str]:
        window = self.config.point_roll_window
        return [
            f"model_{model}_{signal}_rollmax_{window}"
            for model in self.MODEL_TARGETS
            for signal in self.POINT_SIGNAL_TYPES
        ]

    @property
    def collective_features(self) -> list[str]:
        return [
            f"model_{model}_{signal}"
            for model in self.MODEL_TARGETS
            for signal in self.COLLECTIVE_SIGNAL_TYPES
        ]

    def _require_columns(self, frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"偏差信号缺少论文规定列: {missing}")

    def calibrate(self, quiet_signals: pd.DataFrame) -> None:
        """仅使用部署初期已知平静的冷启动期校准阈值。"""
        required = self.point_features + self.collective_features
        self._require_columns(quiet_signals, required)
        if len(quiet_signals) < self.config.collective_min_periods:
            raise ValueError(
                "冷启动样本不足: "
                f"{len(quiet_signals)} < {self.config.collective_min_periods}"
            )

        self.point_thresholds = {}
        for feature in self.point_features:
            series = quiet_signals[feature].astype(float)
            self.point_thresholds[feature] = float(
                series.mean() + self.config.point_k * series.std()
            )

        self.static_z_params = {}
        for feature in self.collective_features:
            series = quiet_signals[feature].astype(float)
            signal_type = next(
                signal for signal in self.COLLECTIVE_SIGNAL_TYPES if signal in feature
            )
            safe_std = max(float(series.std()), self.STD_FLOORS[signal_type])
            self.static_z_params[feature] = {
                "mean": float(series.mean()),
                "std": safe_std,
            }

        z_max = self._compute_static_z_max(quiet_signals)
        z_mean = float(z_max.mean())
        z_std = float(z_max.std())
        self.collective_level_threshold = z_mean + self.config.collective_k * z_std
        self.collective_volatility_threshold = (
            z_std * self.config.collective_volatility_multiplier
        )
        self.changepoint_threshold = z_mean + self.config.changepoint_k * z_std
        self._calibrated = True

    def set_point_relax_factor(self, factor: float) -> None:
        if factor <= 0:
            raise ValueError("点异常阈值放宽系数必须大于 0")
        self._point_relax_factor = float(factor)

    def _compute_static_z_max(self, frame: pd.DataFrame) -> pd.Series:
        values = []
        for feature, params in self.static_z_params.items():
            values.append((frame[feature].astype(float) - params["mean"]) / params["std"])
        if not values:
            return pd.Series(0.0, index=frame.index)
        return pd.concat(values, axis=1).max(axis=1)

    def predict(self, signals: pd.DataFrame) -> pd.DataFrame:
        if not self._calibrated:
            raise RuntimeError("必须先用冷启动期调用 calibrate()")
        self._require_columns(signals, self.point_features + self.collective_features)
        if "timestamp" not in signals:
            raise ValueError("偏差信号必须包含 timestamp")

        result = signals.copy()
        z_max = self._compute_static_z_max(result)
        result["static_signal_z_max"] = z_max

        high_level = (z_max > self.collective_level_threshold).astype(float)
        result["collective_density"] = high_level.rolling(
            self.config.collective_window,
            min_periods=self.config.collective_min_periods,
        ).mean()
        result["collective_volatility"] = z_max.rolling(
            self.config.collective_window,
            min_periods=self.config.collective_min_periods,
        ).std()
        result["changepoint_trend_mean"] = z_max.rolling(
            self.config.changepoint_window,
            min_periods=self.config.changepoint_min_periods,
        ).mean()
        result["is_changepoint"] = (
            result["changepoint_trend_mean"] > self.changepoint_threshold
        )

        triggers = pd.DataFrame(index=result.index)
        for feature in self.point_features:
            threshold = self.point_thresholds[feature] * self._point_relax_factor
            triggers[feature] = (result[feature] > threshold).astype(int)
        result["point_trigger_count"] = triggers.sum(axis=1)

        collective_mask = (
            (result["collective_density"] > self.config.collective_density_threshold)
            & (
                result["collective_volatility"]
                < self.collective_volatility_threshold
            )
        )
        point_mask = (
            result["point_trigger_count"] >= self.config.point_min_triggers
        ) & ~collective_mask

        result["pred_label"] = "N"
        result.loc[collective_mask, "pred_label"] = "CA"
        result.loc[point_mask, "pred_label"] = "PA"

        result["attribution_time"] = pd.NaT
        result.loc[point_mask, "attribution_time"] = result.loc[point_mask, "timestamp"]
        collective_times = result["timestamp"].shift(self.config.collective_window)
        result.loc[collective_mask, "attribution_time"] = collective_times.loc[collective_mask]

        changepoint_times = result["timestamp"].shift(
            self.config.changepoint_attribution_lag
        )
        result["changepoint_attribution_time"] = pd.NaT
        result.loc[result["is_changepoint"], "changepoint_attribution_time"] = (
            changepoint_times.loc[result["is_changepoint"]]
        )

        result["dominant_model"] = "N/A"
        result["dominant_signal"] = "N/A"
        result["trigger_features"] = ""
        for index in result.index[point_mask]:
            active = [feature for feature in triggers if triggers.at[index, feature] == 1]
            if not active:
                continue
            model_votes = {
                model: sum(model in feature for feature in active)
                for model in self.MODEL_TARGETS
            }
            signal_votes = {
                signal: sum(signal in feature for feature in active)
                for signal in self.POINT_SIGNAL_TYPES
            }
            result.at[index, "dominant_model"] = max(model_votes, key=model_votes.get)
            result.at[index, "dominant_signal"] = max(signal_votes, key=signal_votes.get)
            result.at[index, "trigger_features"] = "|".join(active)

        return result

    def calibration_summary(self) -> dict[str, object]:
        if not self._calibrated:
            raise RuntimeError("分类器尚未校准")
        return {
            "config": asdict(self.config),
            "point_thresholds": self.point_thresholds,
            "static_z_params": self.static_z_params,
            "collective_level_threshold": self.collective_level_threshold,
            "collective_volatility_threshold": self.collective_volatility_threshold,
            "changepoint_threshold": self.changepoint_threshold,
        }
