import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class SyntheticDataValidator:
    """
    合成数据质量验证器（适配“只合成正常数据”的场景）

    主要特性：
    - 支持传入 normal_mask，只在“正常时段”上做比较（例如排除 GARCH 标出的异常点）
    - 统计/分布比较默认对真实和合成都做尾部裁剪（winsorize），减少极端异常的影响
    - 输出多种指标：mean/std/median/IQR 差异、KS p 值、ACF 差异、tail 分位差等
    """

    def __init__(self,
                 step,
                 df_real: pd.DataFrame,
                 df_synthetic: pd.DataFrame,
                 normal_mask: pd.Series | None = None,
                 clip_quantile: float = 0.99):
        """
        Parameters
        ----------
        df_real : DataFrame
            真实数据（通常是“含异常”的原始数据，或你预先处理后的正常数据）
        df_synthetic : DataFrame
            合成数据
        normal_mask : Series[bool], optional
            正常时段掩码，index 应与 df_real 对齐。
            True 表示“正常”，False 表示“异常”（将被排除在评估之外）。
        clip_quantile : float
            尾部裁剪分位数，例如 0.99 表示在 [1%, 99%] 范围内 winsorize。
        """
        self.step=step
        # 索引对齐：只保留两边都有的时间点
        common_index = df_real.index.intersection(df_synthetic.index)
        self.df_real = df_real.loc[common_index].copy()
        self.df_synth = df_synthetic.loc[common_index].copy()

        # 正常掩码
        if normal_mask is not None:
            mask = normal_mask.reindex(common_index).fillna(False).astype(bool)
            self.normal_mask = mask
        else:
            self.normal_mask = None

        self.clip_quantile = float(clip_quantile)

    # ---------- 内部工具 ----------

    def _apply_mask(self, series: pd.Series) -> pd.Series:
        """应用 normal_mask，只保留正常时段"""
        if self.normal_mask is None:
            return series
        return series[self.normal_mask]

    def _align_series(self, feature: str):
        """对齐单个特征的真实 & 合成序列，并应用 normal_mask"""
        if feature not in self.df_real.columns or feature not in self.df_synth.columns:
            return None, None
        if self.step ==5:
            col_hist = f'{feature}_transformed' if f'{feature}_transformed' in self.df_real.columns else None
        elif self.step==4:
            col_hist = f'{feature}_resid' if f'{feature}_resid' in self.df_real.columns else None
        elif self.step==3:
            col_hist = f'{feature}_event_adj_resid' if f'{feature}_event_adj_resid' in self.df_real.columns else None
        elif self.step==2:
            col_hist = f'{feature}_ar_resid' if f'{feature}_ar_resid' in self.df_real.columns else None
        else:
            col_hist = feature
        if not col_hist:
            print(f"⚠️ 在 df_real 中未找到 {feature} 的历史列，无法对齐")
            return None, None   
        real = self.df_real[col_hist].astype(float)
        synth = self.df_synth[feature].astype(float)

        real = self._apply_mask(real).dropna()
        synth = self._apply_mask(synth).dropna()

        # 再对齐一次索引（防止其中一方有 NaN 被删后错位）
        common_index = real.index.intersection(synth.index)
        real = real.loc[common_index]
        synth = synth.loc[common_index]

        if len(real) < 10 or len(synth) < 10:
            return None, None

        return real, synth

    def _winsorize_pair(self, real: pd.Series, synth: pd.Series):
        """对真实和合成各自做尾部裁剪（winsorize）"""
        q_low = 1 - self.clip_quantile
        q_high = self.clip_quantile

        r_low, r_high = real.quantile(q_low), real.quantile(q_high)
        s_low, s_high = synth.quantile(q_low), synth.quantile(q_high)

        real_clip = real.clip(r_low, r_high)
        synth_clip = synth.clip(s_low, s_high)
        return real_clip, synth_clip

    # ---------- 统计量比较 ----------

    def compute_statistics_comparison(self, feature: str) -> dict:
        """
        比较单个特征的统计量（在 winsorize 之后）
        返回：
            mean / std / median / IQR 及其百分比差异
        """
        real, synth = self._align_series(feature)
        if real is None:
            print(f"⚠️ 无法对齐 {feature}，跳过统计比较")
            return {}

        real_clip, synth_clip = self._winsorize_pair(real, synth)

        def iqr(x):
            return np.quantile(x, 0.75) - np.quantile(x, 0.25)

        stats_real = {
            'mean': real_clip.mean(),
            'std': real_clip.std(),
            'median': real_clip.median(),
            'iqr': iqr(real_clip)
        }
        stats_synth = {
            'mean': synth_clip.mean(),
            'std': synth_clip.std(),
            'median': synth_clip.median(),
            'iqr': iqr(synth_clip)
        }

        comparison = {}
        for key in stats_real:
            r = stats_real[key]
            s = stats_synth[key]
            if r != 0:
                diff_pct = abs(s - r) / abs(r) * 100
            else:
                diff_pct = abs(s - r) * 100
            diff_abs = abs(s - r)
            comparison[key] = {'real': r, 'synthetic': s, 'diff_pct': diff_pct, 'diff_abs': diff_abs}

        return comparison

    # ---------- 分布检验（去尾部 + 可下采样） ----------

    def compute_distribution_tests(self, feature: str,
                                   max_samples: int = 5000) -> dict:
        """
        在 winsorize 后对真实 & 合成做 KS / MW 检验。
        为避免样本过大导致 p 值过于敏感，若样本量 > max_samples，则随机下采样。
        """
        real, synth = self._align_series(feature)
        if real is None:
            return {}

        real_clip, synth_clip = self._winsorize_pair(real, synth)

        # 下采样
        n = min(len(real_clip), len(synth_clip), max_samples)
        if n < 10:
            return {}

        real_sample = real_clip.sample(n, random_state=42)
        synth_sample = synth_clip.sample(n, random_state=43)

        ks_stat, ks_pval = stats.ks_2samp(real_sample, synth_sample)
        mw_stat, mw_pval = stats.mannwhitneyu(real_sample,
                                             synth_sample,
                                             alternative='two-sided')

        return {
            'ks_test': {'statistic': ks_stat, 'p_value': ks_pval},
            'mw_test': {'statistic': mw_stat, 'p_value': mw_pval}
        }

    # ---------- 自相关结构比较 ----------

    def compute_autocorrelation_comparison(self, feature: str,
                                           max_lag: int = 20) -> dict:
        """
        比较真实 & 合成在 1..max_lag 的自相关（不过滤 tails，用的是正常时段上的原序列）
        返回：
            acf_real / acf_synthetic / mean_acf_diff
        """
        real, synth = self._align_series(feature)
        if real is None:
            return {}

        acf_real = [real.autocorr(lag=i) for i in range(1, max_lag + 1)]
        acf_synth = [synth.autocorr(lag=i) for i in range(1, max_lag + 1)]

        acf_real = [x if np.isfinite(x) else 0.0 for x in acf_real]
        acf_synth = [x if np.isfinite(x) else 0.0 for x in acf_synth]

        acf_diff = float(np.mean(np.abs(np.array(acf_real) - np.array(acf_synth))))

        return {
            'acf_real': acf_real,
            'acf_synthetic': acf_synth,
            'mean_acf_diff': acf_diff
        }

    # ---------- 尾部分位数差异 ----------

    def compute_tail_diff(self, feature: str,
                          quantiles=(0.95, 0.99)) -> dict:
        """
        比较高分位数（tail）差异，衡量极端值是否被过度压制或放大。
        返回：{q: diff_pct}，例如 0.95 / 0.99
        """
        real, synth = self._align_series(feature)
        if real is None:
            return {}

        tail_diffs = {}
        for q in quantiles:
            r_q = real.quantile(q)
            s_q = synth.quantile(q)
            if r_q != 0:
                diff_pct = (s_q - r_q) / abs(r_q) * 100
            else:
                diff_pct = (s_q - r_q) * 100
            tail_diffs[q] = {
                'real': r_q,
                'synthetic': s_q,
                'diff_pct': diff_pct
            }
        return tail_diffs

    # ---------- 质量打分规则 ----------

    @staticmethod
    def _compute_quality_flag(row) -> str:
        """
        简单的规则版质量打分：
        主要针对“正常部分”的拟合，不要求完美复刻原始异常。
        你可以按业务再调整这些阈值。
        """
        mean_diff_pct = row.get('mean_diff_pct', np.nan)
        mean_diff_abs = row.get('mean_diff_pct', np.nan)

        std_diff = row.get('std_diff_pct', np.nan)
        median_diff = row.get('median_diff_pct', np.nan)
        iqr_diff = row.get('iqr_diff_pct', np.nan)
        acf_d = row.get('acf_diff', np.nan)
        tail_99 = row.get('tail_99_diff_pct', np.nan)

        # GOOD：整体水平和动态都比较接近，尾部差异不过分
        if ((mean_diff_pct < 20 or mean_diff_abs < 0.1) and std_diff < 30 and
                median_diff < 20 and iqr_diff < 30 and
                acf_d < 0.06 and abs(tail_99) < 80):
            return 'GOOD'

        # OK：中等偏差，整体形态还算合理
        if (mean_diff_pct < 50 and std_diff < 60 and
                acf_d < 0.12 and abs(tail_99) < 150):
            return 'OK'

        # 其余视作 BAD：要么量级严重不对，要么动态结构很不同，要么尾巴差太多
        return 'BAD'

    # ---------- 汇总所有特征 ----------

    def validate_all_features(self, feature_list: list) -> pd.DataFrame:
        """
        对多个特征做汇总评估，返回 DataFrame。
        指标包括：
            mean_diff_pct / std_diff_pct / median_diff_pct / iqr_diff_pct
            ks_pvalue
            acf_diff
            tail_95_diff_pct / tail_99_diff_pct
            quality (GOOD / OK / BAD)
        """
        results = []

        for feature in feature_list:
            real, synth = self._align_series(feature)
            if real is None:
                continue

            stats_comp = self.compute_statistics_comparison(feature)
            dist_tests = self.compute_distribution_tests(feature)
            acf_comp = self.compute_autocorrelation_comparison(feature)
            tail_comp = self.compute_tail_diff(feature, quantiles=(0.95, 0.99))

            row = {'feature': feature}

            # 统计量差异
            for key in ['mean', 'std', 'median', 'iqr']:
                row[f'{key}_diff_pct'] = stats_comp.get(key, {}).get('diff_pct', np.nan)
                row[f'{key}_real'] = stats_comp.get(key, {}).get('real', np.nan)
                row[f'{key}_synthetic'] = stats_comp.get(key, {}).get('synthetic', np.nan)
                row[f'{key}_diff_abs'] = stats_comp.get(key, {}).get('diff_abs', np.nan)



            # KS p 值
            row['ks_pvalue'] = dist_tests.get('ks_test', {}).get('p_value', np.nan)

            # ACF
            row['acf_diff'] = acf_comp.get('mean_acf_diff', np.nan)

            # tail
            row['tail_95_diff_pct'] = tail_comp.get(0.95, {}).get('diff_pct', np.nan)
            row['tail_99_diff_pct'] = tail_comp.get(0.99, {}).get('diff_pct', np.nan)

            # 质量标记
            row['quality'] = self._compute_quality_flag(row)

            results.append(row)

        return pd.DataFrame(results)

    # ---------- 可视化比较（考虑 normal_mask & 去尾） ----------

    def plot_comparison(self, feature: str,
                        figsize: tuple = (15, 10),
                        save_path: str | None = None,
                        stop: int = 1000):
        """
        可视化比较：时间序列 / 分布 / ACF / QQ
        说明：
          - 时间序列：只画前 stop 个点，且只用 normal_mask 过滤（不裁尾）
          - 分布和 QQ：在 winsorize 后画图，更符合“正常数据”的比较
        """
        real, synth = self._align_series(feature)
        if real is None:
            print(f"特征 {feature} 不存在或有效数据不足")
            return

        # 时间序列用原值（正常时段）
        real_ts = real.iloc[:stop]
        synth_ts = synth.iloc[:stop]

        # 分布 & QQ 用 winsorize 后的值
        real_clip, synth_clip = self._winsorize_pair(real, synth)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1) 时间序列
        ax1 = axes[0, 0]
        ax1.plot(real_ts.index, real_ts.values, alpha=0.7,
                 label='真实数据', linewidth=0.8)
        ax1.plot(synth_ts.index, synth_ts.values, alpha=0.7,
                 label='合成数据', linewidth=0.8)
        ax1.set_title(f'{feature} - 时间序列对比（前{len(real_ts)}点，仅正常时段）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) 分布
        ax2 = axes[0, 1]
        ax2.hist(real_clip, bins=50, alpha=0.5,
                 label='真实数据(裁尾后)', density=True)
        ax2.hist(synth_clip, bins=50, alpha=0.5,
                 label='合成数据(裁尾后)', density=True)
        ax2.set_title(f'{feature} - 分布对比（winsorize@{self.clip_quantile:.2f}）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3) ACF 对比
        ax3 = axes[1, 0]
        acf_comp = self.compute_autocorrelation_comparison(feature)
        lags = range(1, len(acf_comp.get('acf_real', [])) + 1)
        ax3.plot(lags, acf_comp.get('acf_real', []), 'o-',
                 label='真实数据', markersize=4)
        ax3.plot(lags, acf_comp.get('acf_synthetic', []), 's-',
                 label='合成数据', markersize=4)
        ax3.set_title(f'{feature} - 自相关函数对比（正常时段）')
        ax3.set_xlabel('滞后阶数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4) QQ 图（裁尾后）
        ax4 = axes[1, 1]
        quantiles = np.linspace(0.01, 0.99, 50)
        real_q = np.quantile(real_clip, quantiles)
        synth_q = np.quantile(synth_clip, quantiles)
        ax4.scatter(real_q, synth_q, alpha=0.6)
        min_val = min(real_q.min(), synth_q.min())
        max_val = max(real_q.max(), synth_q.max())
        ax4.plot([min_val, max_val], [min_val, max_val],
                 'r--', label='理想线')
        ax4.set_xlabel('真实数据分位数(裁尾后)')
        ax4.set_ylabel('合成数据分位数(裁尾后)')
        ax4.set_title(f'{feature} - QQ图（winsorize@{self.clip_quantile:.2f}）')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()