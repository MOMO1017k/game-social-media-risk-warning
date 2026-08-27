"""从两个 TFT 子模型输出中提取论文定义的四类偏差信号。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EPSILON = 1e-10


def _model_prefix(model_name: str) -> str:
    """统一模型列名前缀，避免 Notebook 中的 ``model_model_*`` 重复。"""
    normalized = str(model_name)
    while normalized.startswith("model_"):
        normalized = normalized.removeprefix("model_")
    return f"model_{normalized}"


def _as_probability(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=float).reshape(-1)) + EPSILON
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return values / total


def _validate_result(result: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {"metrics", "attention", "vsn"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"TFT 推理结果缺少字段: {sorted(missing)}")

    metrics = result["metrics"].copy()
    for column in ("time_idx", "residual"):
        if column not in metrics.columns:
            raise ValueError(f"metrics 缺少必需列: {column}")

    attention = np.asarray(result["attention"], dtype=float)
    vsn = np.asarray(result["vsn"], dtype=float)
    if not (len(metrics) == len(attention) == len(vsn)):
        raise ValueError(
            "metrics、attention 与 vsn 的样本数不一致: "
            f"{len(metrics)}, {len(attention)}, {len(vsn)}"
        )
    return metrics, attention, vsn


def extract_deviation_signals(
    model_name: str,
    result: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> pd.DataFrame:
    """计算残差比、注意力 KL、变量选择 JS 与秩相关距离。

    参数结构与原 Notebook 的 ``TFTEngine.analyze_rolling()`` 输出兼容。
    基线至少应包含 ``residual_p95``/``residual_p5``/``residual_std`` 中
    的一项，以及 ``att_mean`` 和 ``vsn_mean``。
    """
    metrics, attention, vsn = _validate_result(result)
    if not baseline:
        raise ValueError(f"模型 {model_name!r} 缺少基线统计量")

    residual_scale = max(
        abs(float(baseline.get("residual_p95", 0.0))),
        abs(float(baseline.get("residual_p5", 0.0))),
        abs(float(baseline.get("residual_std", 0.0))) * 1.645,
        EPSILON,
    )
    residual_ratio = np.clip(metrics["residual"].abs().to_numpy() / residual_scale, 0, 20)

    att_base = baseline.get("att_mean")
    vsn_base = baseline.get("vsn_mean")
    if att_base is None or vsn_base is None:
        raise ValueError(f"模型 {model_name!r} 的基线缺少 att_mean 或 vsn_mean")

    att_reference = _as_probability(np.asarray(att_base))
    vsn_reference = _as_probability(np.asarray(vsn_base))
    baseline_rank = np.argsort(np.argsort(-vsn_reference))

    attention_kl = np.zeros(len(metrics), dtype=float)
    vsn_js = np.zeros(len(metrics), dtype=float)
    vsn_rank_shift = np.zeros(len(metrics), dtype=float)

    for index in range(len(metrics)):
        current_attention = _as_probability(attention[index])
        if current_attention.shape != att_reference.shape:
            raise ValueError(
                f"模型 {model_name!r} 的注意力形状与基线不一致: "
                f"{current_attention.shape} != {att_reference.shape}"
            )
        attention_kl[index] = np.sum(
            current_attention * np.log(current_attention / att_reference)
        )

        current_vsn = _as_probability(vsn[index])
        if current_vsn.shape != vsn_reference.shape:
            raise ValueError(
                f"模型 {model_name!r} 的 VSN 形状与基线不一致: "
                f"{current_vsn.shape} != {vsn_reference.shape}"
            )
        midpoint = 0.5 * (current_vsn + vsn_reference)
        vsn_js[index] = 0.5 * np.sum(current_vsn * np.log(current_vsn / midpoint))
        vsn_js[index] += 0.5 * np.sum(vsn_reference * np.log(vsn_reference / midpoint))

        current_rank = np.argsort(np.argsort(-current_vsn))
        correlation, _ = spearmanr(baseline_rank, current_rank)
        vsn_rank_shift[index] = 1.0 - correlation if np.isfinite(correlation) else 1.0

    prefix = _model_prefix(model_name)
    return pd.DataFrame(
        {
            "time_idx": metrics["time_idx"].to_numpy(),
            f"{prefix}_residual_ratio": residual_ratio,
            f"{prefix}_att_kl": np.clip(attention_kl, 0, 20),
            f"{prefix}_vsn_js": np.clip(vsn_js, 0, 5),
            f"{prefix}_vsn_rank_shift": np.clip(vsn_rank_shift, 0, 2),
        }
    ).drop_duplicates(subset="time_idx", keep="last")


def merge_model_signals(
    timeline: pd.DataFrame,
    results_by_model: Mapping[str, Mapping[str, Any]],
    baselines_by_model: Mapping[str, Mapping[str, Any]],
    rolling_window: int = 8,
) -> pd.DataFrame:
    """对齐两个子模型的静态信号，并生成论文使用的 8 窗口滚动最大值。"""
    required_columns = {"time_idx", "timestamp"}
    missing = required_columns.difference(timeline.columns)
    if missing:
        raise ValueError(f"时间轴缺少字段: {sorted(missing)}")
    if rolling_window < 1:
        raise ValueError("rolling_window 必须为正整数")

    merged = timeline[["time_idx", "timestamp"]].drop_duplicates("time_idx").copy()
    for model_name, result in results_by_model.items():
        if model_name not in baselines_by_model:
            raise ValueError(f"模型 {model_name!r} 没有对应基线")
        model_signals = extract_deviation_signals(
            model_name=model_name,
            result=result,
            baseline=baselines_by_model[model_name],
        )
        merged = merged.merge(model_signals, on="time_idx", how="left", validate="one_to_one")

    static_columns = [column for column in merged if column.startswith("model_")]
    for column in static_columns:
        merged[f"{column}_rollmax_{rolling_window}"] = merged[column].rolling(
            rolling_window, min_periods=1
        ).max()

    return merged.replace([np.inf, -np.inf], np.nan).fillna(0)
