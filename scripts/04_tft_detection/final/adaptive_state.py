"""论文第 4.5.5 节的双轨自适应状态机，不绑定具体 TFT 实现。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class AdaptiveState(str, Enum):
    NORMAL = "NORMAL"
    CA_TENTATIVE = "CA_TENTATIVE"
    ACCUMULATING = "ACCUMULATING"
    SWITCHING = "SWITCHING"
    COOLDOWN = "COOLDOWN"


class AdaptiveAction(str, Enum):
    NONE = "NONE"
    START_FORWARD_COLLECTION = "START_FORWARD_COLLECTION"
    FINETUNE_FROM_HISTORY = "FINETUNE_FROM_HISTORY"
    FINETUNE_FROM_FORWARD_DATA = "FINETUNE_FROM_FORWARD_DATA"
    EXTEND_PARALLEL_VALIDATION = "EXTEND_PARALLEL_VALIDATION"
    ACCEPT_NEW_BASELINE = "ACCEPT_NEW_BASELINE"
    REJECT_NEW_BASELINE = "REJECT_NEW_BASELINE"


@dataclass(frozen=True)
class AdaptiveConfig:
    ca_confirm_windows: int = 192
    drift_retrace_windows: int = 288
    ca_accumulate_windows: int = 1344
    parallel_windows: int = 288
    parallel_extension_windows: int = 480
    recovery_normal_ratio: float = 0.85
    stability_normal_ratio: float = 0.85
    cooldown_windows: int = 672
    point_relax_windows: int = 96
    point_relax_factor: float = 1.2


@dataclass(frozen=True)
class Transition:
    state: AdaptiveState
    action: AdaptiveAction
    current_time_idx: int
    data_start_time_idx: int | None = None
    reason: str = ""


class AdaptiveStateMachine:
    """只负责状态与切换决策；模型微调由外部引擎执行。

    这种拆分避免原 Notebook 在状态机内部重新创建完整 TFT 模型。正式的
    ``TFT_tft_engine.py`` 应在收到微调动作后，仅解冻顶层参数、使用时间顺序
    训练/验证划分、早停并回滚最佳权重。
    """

    def __init__(self, config: AdaptiveConfig | None = None):
        self.config = config or AdaptiveConfig()
        self.state = AdaptiveState.NORMAL
        self.ca_counter = 0
        self.forward_collected = 0
        self.parallel_seen = 0
        self.parallel_extended = False
        self.pending_track: str | None = None
        self.cooldown_until = -1
        self.baseline_version = 0

    def observe(self, classified_batch: pd.DataFrame, current_time_idx: int) -> Transition:
        """处理一批 N/PA/CA 标签和独立的 ``is_changepoint`` 标记。"""
        required = {"pred_label", "is_changepoint"}
        missing = required.difference(classified_batch.columns)
        if missing:
            raise ValueError(f"分类结果缺少字段: {sorted(missing)}")

        if current_time_idx < self.cooldown_until:
            self.state = AdaptiveState.COOLDOWN
            return Transition(self.state, AdaptiveAction.NONE, current_time_idx, reason="cooldown")
        if self.state == AdaptiveState.COOLDOWN:
            self.state = AdaptiveState.NORMAL

        if self.state == AdaptiveState.SWITCHING:
            return Transition(self.state, AdaptiveAction.NONE, current_time_idx, reason="parallel_validation")

        if classified_batch["is_changepoint"].astype(bool).any():
            self.state = AdaptiveState.SWITCHING
            self.pending_track = "drift"
            start = max(0, current_time_idx - self.config.drift_retrace_windows)
            return Transition(
                self.state,
                AdaptiveAction.FINETUNE_FROM_HISTORY,
                current_time_idx,
                data_start_time_idx=start,
                reason="long_term_drift",
            )

        ca_count = int((classified_batch["pred_label"] == "CA").sum())
        if self.state == AdaptiveState.NORMAL:
            self.ca_counter = ca_count if ca_count else max(0, self.ca_counter - 1)
            if self.ca_counter > 0:
                self.state = AdaptiveState.CA_TENTATIVE

        elif self.state == AdaptiveState.CA_TENTATIVE:
            self.ca_counter = self.ca_counter + ca_count if ca_count else max(0, self.ca_counter - 1)
            if self.ca_counter <= 0:
                self.state = AdaptiveState.NORMAL
            elif self.ca_counter >= self.config.ca_confirm_windows:
                self.state = AdaptiveState.ACCUMULATING
                self.forward_collected = 0
                return Transition(
                    self.state,
                    AdaptiveAction.START_FORWARD_COLLECTION,
                    current_time_idx,
                    data_start_time_idx=current_time_idx,
                    reason="collective_anomaly_confirmed",
                )

        elif self.state == AdaptiveState.ACCUMULATING:
            self.forward_collected += len(classified_batch)
            if self.forward_collected >= self.config.ca_accumulate_windows:
                self.state = AdaptiveState.SWITCHING
                self.pending_track = "collective"
                start = current_time_idx - self.forward_collected + 1
                return Transition(
                    self.state,
                    AdaptiveAction.FINETUNE_FROM_FORWARD_DATA,
                    current_time_idx,
                    data_start_time_idx=max(0, start),
                    reason="new_normal_accumulated",
                )

        return Transition(self.state, AdaptiveAction.NONE, current_time_idx)

    def evaluate_parallel_models(
        self,
        old_labels: list[str] | np.ndarray,
        new_labels: list[str] | np.ndarray,
        current_time_idx: int,
    ) -> Transition:
        if self.state != AdaptiveState.SWITCHING or self.pending_track is None:
            raise RuntimeError("只有 SWITCHING 状态可以执行并行切换判定")

        old = np.asarray(old_labels)
        new = np.asarray(new_labels)
        if len(old) != len(new) or len(old) == 0:
            raise ValueError("新旧模型标签必须非空且长度一致")

        self.parallel_seen += len(old)
        required = (
            self.config.parallel_extension_windows
            if self.parallel_extended
            else self.config.parallel_windows
        )
        if self.parallel_seen < required:
            return Transition(self.state, AdaptiveAction.NONE, current_time_idx, reason="collecting_parallel_labels")

        old_normal = float((old == "N").mean())
        new_normal = float((new == "N").mean())

        accept = False
        reject = False
        if self.pending_track == "collective":
            reject = old_normal >= self.config.recovery_normal_ratio
            accept = (
                new_normal >= self.config.stability_normal_ratio
                and new_normal >= old_normal
            )
        else:
            accept = new_normal >= self.config.stability_normal_ratio

        if accept:
            self.baseline_version += 1
            return self._finish_switch(current_time_idx, AdaptiveAction.ACCEPT_NEW_BASELINE)
        if reject:
            return self._finish_switch(current_time_idx, AdaptiveAction.REJECT_NEW_BASELINE)
        if not self.parallel_extended:
            self.parallel_extended = True
            self.parallel_seen = 0
            return Transition(
                self.state,
                AdaptiveAction.EXTEND_PARALLEL_VALIDATION,
                current_time_idx,
                reason=f"old_N={old_normal:.3f}, new_N={new_normal:.3f}",
            )
        return self._finish_switch(current_time_idx, AdaptiveAction.REJECT_NEW_BASELINE)

    def _finish_switch(self, current_time_idx: int, action: AdaptiveAction) -> Transition:
        self.state = AdaptiveState.COOLDOWN
        self.cooldown_until = current_time_idx + self.config.cooldown_windows
        self.ca_counter = 0
        self.forward_collected = 0
        self.parallel_seen = 0
        self.parallel_extended = False
        self.pending_track = None
        return Transition(self.state, action, current_time_idx)
