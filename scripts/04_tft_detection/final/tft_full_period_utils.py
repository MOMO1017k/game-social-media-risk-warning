# ============================================================
# tft_full_period_utils.py
# 方案1 + 方案B: 全周期TFT训练的工具函数
# ============================================================

import numpy as np
import pandas as pd
import copy
from typing import Dict, Set, Tuple


def create_held_out_split(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    block_days: int = 5,
    min_block_windows: int = 96,
) -> Tuple[pd.Series, pd.Series, dict]:
    """
    从时序数据中划出季节性均匀分布的held-out块
    
    策略: 每月最后 block_days 天作为 held-out
    
    设计考量:
    - 每个块 ≥ 96 窗口 (24h)，保证 TFT encoder 有足够上下文
    - 均匀分布在全年各月，覆盖季节性
    - 训练数据中的时间间断由 allow_missing_timesteps=True 处理
    - 不需要配合异常注入（两者在不同阶段独立进行）
    
    Parameters
    ----------
    df : pd.DataFrame
        含时间列的已变换数据
    time_col : str
        时间列名
    block_days : int
        每月末取多少天作为 held-out
    min_block_windows : int
        每个块最小窗口数，不足则跳过该月
        
    Returns
    -------
    train_mask : pd.Series[bool]
    held_out_mask : pd.Series[bool]
    split_info : dict
    """
    ts = pd.to_datetime(df[time_col])
    held_out_mask = pd.Series(False, index=df.index)
    block_info = []
    
    # 按自然月处理
    months_in_data = sorted(ts.dt.to_period('M').unique())
    
    for ym in months_in_data:
        # 该月的数据 mask
        month_mask = ts.dt.to_period('M') == ym
        month_end = ym.end_time
        cutoff = month_end - pd.Timedelta(days=block_days)
        
        # 该月末 block_days 天
        block = month_mask & (ts > cutoff)
        n_windows = int(block.sum())
        
        if n_windows >= min_block_windows:
            held_out_mask = held_out_mask | block
            block_info.append({
                'month': str(ym),
                'n_windows': n_windows,
                'start': ts[block].min().strftime('%Y-%m-%d %H:%M'),
                'end': ts[block].max().strftime('%Y-%m-%d %H:%M'),
            })
        else:
            # 数据不足的月份（可能是首月或末月）
            if n_windows > 0:
                print(f"  ⚠️ {ym}: 仅{n_windows}窗口 < {min_block_windows}，跳过")
    
    train_mask = ~held_out_mask
    ratio = held_out_mask.sum() / len(df)
    
    split_info = {
        'n_train': int(train_mask.sum()),
        'n_held_out': int(held_out_mask.sum()),
        'held_out_ratio': round(float(ratio), 4),
        'n_blocks': len(block_info),
        'block_days': block_days,
        'blocks': block_info,
    }
    
    print(f"\n{'='*50}")
    print(f"Held-out Split (每月末{block_days}天)")
    print(f"{'='*50}")
    print(f"  训练集:   {split_info['n_train']:>6d} windows ({1-ratio:.1%})")
    print(f"  Held-out: {split_info['n_held_out']:>6d} windows ({ratio:.1%})")
    print(f"  有效月份: {split_info['n_blocks']}")
    print(f"  {'─'*46}")
    for b in block_info:
        print(f"  {b['month']}: {b['n_windows']:>4d} windows  "
              f"({b['start']} ~ {b['end']})")
    
    return train_mask, held_out_mask, split_info


def compute_baseline_from_held_out(
    analysis_results: dict,
    held_out_time_indices: Set[int],
) -> dict:
    """
    从 TFT 推理结果中，仅用 held-out 窗口计算无偏 baseline
    
    核心逻辑:
    - TFT 训练时没见过 held-out 数据
    - 因此 TFT 在 held-out 上的残差是无偏的
    - 这些残差的统计量反映 TFT 在"未见过的正常数据"上的表现
    - 部署时 TFT 看到的新正常数据也是"未见过的"
    - 所以 held-out baseline ≈ 部署时正常数据的 baseline
    
    Parameters
    ----------
    analysis_results : dict
        engine.analyze_rolling() 的输出, 包含:
          'metrics' (DataFrame): time_idx, residual, divergence 等
          'attention' (ndarray): shape (n, ...)
          'vsn' (ndarray): shape (n, ...)
    held_out_time_indices : set
        held-out 窗口的 time_idx 集合
        
    Returns
    -------
    baseline : dict
        格式与 TFTEngine._build_baseline 一致
    """
    df_scalars = analysis_results['metrics']
    full_attention = analysis_results['attention']
    full_vsn = analysis_results['vsn']
    
    # 只取 held-out 窗口
    mask = df_scalars['time_idx'].isin(held_out_time_indices).values
    n = int(mask.sum())
    
    print(f"\n[Held-out Baseline]")
    print(f"  推理总窗口:   {len(df_scalars)}")
    print(f"  Held-out匹配: {n}")
    
    if n < 100:
        import warnings
        warnings.warn(
            f"Held-out仅{n}个样本，baseline可能不稳定。"
            f"建议增加block_days或检查时间对齐。"
        )
    
    residuals = df_scalars.loc[mask, 'residual'].values
    divergence = df_scalars.loc[mask, 'divergence'].values
    att = full_attention[mask]
    vsn = full_vsn[mask]
    
    # 鲁棒统计量 (MAD → σ 估计)
    res_median = np.median(residuals)
    res_mad = np.median(np.abs(residuals - res_median))
    res_std = float(res_mad * 1.4826)  # MAD to σ conversion
    
    div_median = np.median(divergence)
    div_mad = np.median(np.abs(divergence - div_median))
    div_std = float(div_mad * 1.4826)
    
    baseline = {
        # 残差分布
        'residual_mean': float(res_median),
        'residual_std': res_std,
        'residual_p5': float(np.percentile(residuals, 5)),
        'residual_p95': float(np.percentile(residuals, 95)),
        
        # 不确定性分布
        'divergence_mean': float(div_median),
        'divergence_std': div_std,
        
        # 注意力权重基线
        'att_mean': np.mean(att, axis=0),
        'att_std': np.std(att, axis=0) + 1e-8,
        
        # VSN 门控基线
        'vsn_mean': np.mean(vsn, axis=0),
        'vsn_std': np.std(vsn, axis=0) + 1e-8,
        
        # 元信息
        'n_samples': n,
        'source': 'held_out',
    }
    
    print(f"  Residual:   μ={baseline['residual_mean']:.4f}, "
          f"σ={baseline['residual_std']:.4f}")
    print(f"  Residual:   P5={baseline['residual_p5']:.4f}, "
          f"P95={baseline['residual_p95']:.4f}")
    print(f"  Divergence: μ={baseline['divergence_mean']:.4f}, "
          f"σ={baseline['divergence_std']:.4f}")
    
    return baseline


def verify_baseline_bias(
    analysis_results: dict,
    held_out_time_indices: Set[int],
    model_name: str = '',
):
    """
    验证 held-out baseline 与 full-data baseline 的偏差
    
    预期: held-out 的残差 σ > full-data 的残差 σ
    (因为 TFT 在训练数据上过拟合，残差偏小)
    """
    df_scalars = analysis_results['metrics']
    
    # Full-data stats (有偏)
    all_res = df_scalars['residual'].values
    all_std = float(np.median(np.abs(all_res - np.median(all_res))) * 1.4826)
    
    # Held-out stats (无偏)
    ho_mask = df_scalars['time_idx'].isin(held_out_time_indices).values
    ho_res = df_scalars.loc[ho_mask, 'residual'].values
    ho_std = float(np.median(np.abs(ho_res - np.median(ho_res))) * 1.4826)
    
    # Train-only stats
    tr_res = df_scalars.loc[~ho_mask, 'residual'].values
    tr_std = float(np.median(np.abs(tr_res - np.median(tr_res))) * 1.4826)
    
    inflation = ho_std / tr_std if tr_std > 0 else float('inf')
    
    print(f"\n[Baseline Bias Verification] {model_name}")
    print(f"  {'Source':<15s} {'Residual σ':>12s} {'N samples':>10s}")
    print(f"  {'─'*40}")
    print(f"  {'Train-only':<15s} {tr_std:>12.4f} {(~ho_mask).sum():>10d}")
    print(f"  {'Held-out':<15s} {ho_std:>12.4f} {ho_mask.sum():>10d}")
    print(f"  {'Full-data':<15s} {all_std:>12.4f} {len(all_res):>10d}")
    print(f"  {'─'*40}")
    print(f"  Inflation ratio (held-out / train): {inflation:.3f}")
    
    if inflation > 1.0:
        print(f"  ✅ Held-out σ > Train σ (预期行为，TFT在训练数据上残差偏小)")
    else:
        print(f"  ⚠️ Inflation < 1.0 (异常，检查数据分割)")
    
    return {
        'train_std': tr_std,
        'held_out_std': ho_std,
        'full_data_std': all_std,
        'inflation_ratio': inflation,
    }