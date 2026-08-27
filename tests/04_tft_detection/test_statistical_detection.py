import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "04_tft_detection"
sys.path.insert(0, str(MODULE_ROOT))

from final.signals import extract_deviation_signals  # noqa: E402
from final.adaptive_state import (  # noqa: E402
    AdaptiveAction,
    AdaptiveConfig,
    AdaptiveStateMachine,
)
from final.statistical_classifier import (  # noqa: E402
    StatisticalThresholdClassifier,
    ThresholdConfig,
)


class SignalExtractionTests(unittest.TestCase):
    def test_extracts_four_paper_signals(self):
        result = {
            "metrics": pd.DataFrame({"time_idx": [1, 2], "residual": [0.5, -1.0]}),
            "attention": np.array([[0.7, 0.3], [0.2, 0.8]]),
            "vsn": np.array([[0.6, 0.4], [0.1, 0.9]]),
        }
        baseline = {
            "residual_p95": 1.0,
            "residual_p5": -1.0,
            "residual_std": 0.5,
            "att_mean": np.array([0.5, 0.5]),
            "vsn_mean": np.array([0.5, 0.5]),
        }
        signals = extract_deviation_signals("model_comment_pc1", result, baseline)
        self.assertEqual(len(signals), 2)
        self.assertIn("model_comment_pc1_residual_ratio", signals)
        self.assertIn("model_comment_pc1_att_kl", signals)
        self.assertIn("model_comment_pc1_vsn_js", signals)
        self.assertIn("model_comment_pc1_vsn_rank_shift", signals)


class StatisticalClassifierTests(unittest.TestCase):
    def _frame(self, rows=24):
        config = ThresholdConfig(
            collective_window=4,
            collective_min_periods=2,
            changepoint_window=6,
            changepoint_min_periods=4,
            changepoint_attribution_lag=3,
        )
        classifier = StatisticalThresholdClassifier(config)
        frame = pd.DataFrame(
            {
                "time_idx": np.arange(rows),
                "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min"),
            }
        )
        for feature in classifier.point_features:
            frame[feature] = np.linspace(0.0, 0.01, rows)
        for feature in classifier.collective_features:
            frame[feature] = 0.0
        return classifier, frame

    def test_point_anomaly_requires_four_votes(self):
        classifier, frame = self._frame()
        classifier.calibrate(frame.iloc[:12])
        for feature in classifier.point_features[:4]:
            frame.loc[20, feature] = 100.0
        detected = classifier.predict(frame)
        self.assertEqual(detected.loc[20, "pred_label"], "PA")
        self.assertEqual(detected.loc[20, "point_trigger_count"], 4)

    def test_missing_signal_fails_loudly(self):
        classifier, frame = self._frame()
        with self.assertRaises(ValueError):
            classifier.calibrate(frame.drop(columns=[classifier.point_features[0]]))


class AdaptiveStateTests(unittest.TestCase):
    def test_drift_uses_configured_history_window(self):
        machine = AdaptiveStateMachine(AdaptiveConfig(drift_retrace_windows=12))
        batch = pd.DataFrame({"pred_label": ["N"], "is_changepoint": [True]})
        transition = machine.observe(batch, current_time_idx=100)
        self.assertEqual(transition.action, AdaptiveAction.FINETUNE_FROM_HISTORY)
        self.assertEqual(transition.data_start_time_idx, 88)

    def test_collective_anomaly_uses_forward_accumulation(self):
        machine = AdaptiveStateMachine(
            AdaptiveConfig(ca_confirm_windows=2, ca_accumulate_windows=3)
        )
        ca = pd.DataFrame({"pred_label": ["CA"], "is_changepoint": [False]})
        machine.observe(ca, current_time_idx=10)
        confirmed = machine.observe(ca, current_time_idx=11)
        self.assertEqual(confirmed.action, AdaptiveAction.START_FORWARD_COLLECTION)

        forward = pd.DataFrame(
            {"pred_label": ["N", "N", "N"], "is_changepoint": [False] * 3}
        )
        finetune = machine.observe(forward, current_time_idx=14)
        self.assertEqual(finetune.action, AdaptiveAction.FINETUNE_FROM_FORWARD_DATA)


if __name__ == "__main__":
    unittest.main()
