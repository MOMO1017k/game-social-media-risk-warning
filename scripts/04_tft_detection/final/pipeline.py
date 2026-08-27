"""不依赖训练引擎的论文统计检测入口。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .signals import merge_model_signals
from .statistical_classifier import StatisticalThresholdClassifier, ThresholdConfig


def run_paper_aligned_detection(
    timeline: pd.DataFrame,
    results_by_model: Mapping[str, Mapping[str, Any]],
    baselines_by_model: Mapping[str, Mapping[str, Any]],
    quiet_start: str | pd.Timestamp,
    quiet_end: str | pd.Timestamp,
    threshold_config: ThresholdConfig | None = None,
) -> tuple[pd.DataFrame, StatisticalThresholdClassifier]:
    """从 TFT 推理结果生成偏差信号，按冷启动期校准并输出分层检测结果。"""
    signals = merge_model_signals(
        timeline=timeline,
        results_by_model=results_by_model,
        baselines_by_model=baselines_by_model,
        rolling_window=(threshold_config or ThresholdConfig()).point_roll_window,
    )
    signals["timestamp"] = pd.to_datetime(signals["timestamp"])
    quiet_mask = (
        (signals["timestamp"] >= pd.Timestamp(quiet_start))
        & (signals["timestamp"] < pd.Timestamp(quiet_end))
    )
    quiet_signals = signals.loc[quiet_mask].copy()
    if quiet_signals.empty:
        raise ValueError("冷启动时间范围与 TFT 推理结果没有交集")

    classifier = StatisticalThresholdClassifier(threshold_config)
    classifier.calibrate(quiet_signals)
    return classifier.predict(signals), classifier
