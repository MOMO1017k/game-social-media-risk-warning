import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1) 逆变换函数库（修正版）
#    - [MOD-YJ] 修正 Yeo-Johnson 逆变换：分段应按 y>=0 / y<0，而不是按 lambda 正负
# ============================================================

def inverse_yeo_johnson_correct(y, lmbda):
    """
    正确的 Yeo-Johnson 逆变换（向量化）
    分段由 y 的正负决定：
      y>=0  <-> x>=0
      y<0   <-> x<0
    """
    y = np.asarray(y, dtype=float)
    x = np.empty_like(y, dtype=float)
    eps = 1e-12

    pos = y >= 0
    neg = ~pos

    # x >= 0 branch
    if abs(lmbda) < 1e-10:
        # y = log1p(x) -> x = expm1(y)
        x[pos] = np.expm1(y[pos])
    else:
        base = lmbda * y[pos] + 1.0
        base = np.maximum(base, eps)
        x[pos] = np.power(base, 1.0 / lmbda) - 1.0

    # x < 0 branch
    if abs(lmbda - 2.0) < 1e-10:
        # y = -log1p(-x) -> -y = log1p(-x) -> -x = expm1(-y)
        x[neg] = -np.expm1(-y[neg])
    else:
        # y = -(( (1-x)^(2-l) - 1)/(2-l))  -> (1-x)^(2-l) = 1 - (2-l)*y
        base = 1.0 - (2.0 - lmbda) * y[neg]   # y[neg] < 0 => base > 1 generally
        base = np.maximum(base, eps)
        x[neg] = 1.0 - np.power(base, 1.0 / (2.0 - lmbda))

    return x


def inverse_log1p(y):
    """y = log1p(x) -> x = expm1(y)"""
    y = np.asarray(y, dtype=float)
    return np.expm1(y)


def inverse_sqrt(y):
    """y = sqrt(x) -> x = y^2（sqrt 一般用于非负）"""
    y = np.asarray(y, dtype=float)
    return np.square(y)


def inverse_standardize(y, original_mean, original_std):
    """y = (x-mean)/std -> x = y*std + mean"""
    y = np.asarray(y, dtype=float)
    return y * float(original_std) + float(original_mean)


def inverse_boxcox(y, lambda_param, shift=0.0):
    """
    Box-Cox 逆变换（允许 shift）
      y = ( (x+shift)^λ - 1)/λ , λ!=0
      y = log(x+shift)         , λ=0
    -> x = inv(...) - shift
    """
    y = np.asarray(y, dtype=float)
    lmbda = float(lambda_param)
    shift = float(shift)
    eps = 1e-12

    if abs(lmbda) < 1e-10:
        x_shifted = np.exp(y)
    else:
        base = lmbda * y + 1.0
        base = np.maximum(base, eps)
        x_shifted = np.power(base, 1.0 / lmbda)

    return x_shifted - shift
# ============================================================
# 强制均值校准
# ============================================================

def force_mean_correction(data, target_mean, tolerance=0.05):
    """
    最后一道防线：如果均值偏差过大，通过缩放强制拉回。
    """
    current_mean = np.mean(data)
    if current_mean == 0 or target_mean == 0:
        return data
        
    bias = (current_mean - target_mean) / target_mean
    
    # 只有偏差超过阈值（如 5%）才修正，避免微调导致过拟合
    if abs(bias) > tolerance:
        scale = target_mean / current_mean
        # print(f"    ⚖️ 均值修正触发: {current_mean:.4f} -> {target_mean:.4f} (x{scale:.4f})")
        return data * scale
    return data


# ============================================================
#  分位数校准函数 (核心新增)
# ============================================================
def quantile_calibrate(feature,synth_data, hist_stats, robust=True):
    """
    鲁棒分位数校准：
    防止历史数据中的极端离群值（Outliers）破坏合成数据分布。
    """
    s_data = np.asarray(synth_data, dtype=float)
    n = len(s_data)
    if n == 0:
        return s_data

    # 1. 获取/构造历史分布骨架
    if 'percentiles' in hist_stats:
        # Step 1 计算好的 101 个点 (0% ~ 100%)
        hist_p_values = np.array(hist_stats['percentiles'])
        p_points = np.linspace(0, 100, len(hist_p_values))
    else:
        # 简易兜底
        hist_mean = hist_stats.get('mean', 0)
        hist_min = hist_stats.get('min', 0)
        hist_max = hist_stats.get('max', hist_mean * 2)
        p_points = np.array([0, 50, 100])
        hist_p_values = np.array([hist_min, hist_stats.get('median', hist_mean), hist_max])

    # =======================================================
    # [核心修改] 鲁棒处理：削弱极值的影响
    # =======================================================
    if robust and len(hist_p_values) >= 100:
        # 策略：如果最大值 (P100) 远大于 P99，说明可能是异常值
        # 这里的 "远大于" 定义为：(P100 - P99) > 3 * (P99 - P95)
        # 我们用 P99 加上一个合理的增量来替代真实的 P100
        
        p95 = hist_p_values[95]
        p99 = hist_p_values[99]
        p100 = hist_p_values[100] # 也就是 max
        
        # 计算尾部斜率
        tail_gap = p99 - p95
        if tail_gap == 0: tail_gap = 1e-6
        
        # 检查 P100 是否离谱
        # 如果 P100 比 P99 大出 5 倍的 (P99-P95) 区间，我们认为它是脏数据或极端离群点
        threshold_val = p99 + 5.0 * tail_gap
        
        if p100 > threshold_val:
            print(f"  🛡️ 触发鲁棒截断{feature} :Max值 {p100:.2f} 被限制为 {threshold_val:.2f} (基于P99推断)")
            # 修改映射表中的最大值，防止合成数据被拉得太远
            hist_p_values[-1] = threshold_val 

    # 2. 计算合成数据的百分位秩
    ranks = stats.rankdata(s_data, method='average')
    s_percentiles = (ranks - 1) / (n - 1) * 100

    # 3. 线性插值映射
    calibrated = np.interp(s_percentiles, p_points, hist_p_values)
    
    return calibrated

# ============================================================
# 核心新增：分布校准函数
# ============================================================

def calibrate_distribution(synth_data, hist_mean, hist_std, hist_max, feature_name):
    """
    强制校准合成数据的分布，使其统计特性与历史数据对齐。
    用于解决逆变换后的数值漂移问题。
    """
    s_mean = np.mean(synth_data)
    s_std = np.std(synth_data)
    
    # 如果合成数据全是常数或NaN，无法校准
    if s_std < 1e-9 or np.isnan(s_std):
        return synth_data

    # 1. 计算偏差
    mean_bias = abs(s_mean - hist_mean) / (abs(hist_mean) + 1e-6)
    std_bias = abs(s_std - hist_std) / (hist_std + 1e-6)
    
    # 2. 判定是否需要校准
    # 阈值：如果均值或方差偏差超过 50%，或者最大值极其离谱，则触发校准
    # 对于 "BAD" 的特征，这个阈值通常都会被触发
    # needs_calibration = (mean_bias > 0.5) or (std_bias > 0.5) or (np.max(synth_data) > 10 * hist_max)
    needs_calibration = (mean_bias > 0.2) or (std_bias > 0.2) or (np.max(synth_data) > 10 * hist_max)

    
    if needs_calibration:
        print(f"  🔧 校准触发 {feature_name:<20}: Mean {s_mean:.2f}->{hist_mean:.2f} | Std {s_std:.2f}->{hist_std:.2f}")
        
        # 3. Z-Score 归一化重构 (Moment Matching)
        # S_new = (S - mu_s) / sigma_s * sigma_h + mu_h
        # 这会保留波形形状，但强行将统计量拉回历史水平
        
        # 考虑到 Step 5 可能添加了趋势（Trend），我们可能希望保留一点 Mean 的变化
        # 但如果是严重的漂移，完全保留 Trend 也是错误的
        # 折中方案：校准到 (Historical Mean * 1.1) 允许微涨，或者完全对齐
        # 这里选择完全对齐 Mean 和 Std，这是最安全的“洗白”方式
        
        calibrated = (synth_data - s_mean) / s_std * hist_std + hist_mean
        
        # 4. 再次确保非负
        calibrated = np.maximum(calibrated, 0.0)
        
        return calibrated
    else:
        return synth_data


# ============================================================
# 2) 零值注入策略（修正版）
#    - [MOD-Z] 阈值注入改为“精确选前 k 个最小值”，避免分位数重复值导致超注入
#    - [MOD-Z2] 支持“只补足零值”（不尝试减少已经存在的零）
# ============================================================

def inject_zeros_threshold_exact_only_add(data, target_zero_ratio):
    """
    只“增加零值”以达到目标比例（不会减少现有零）
    - 先计算当前零比例
    - 若不足，则把最小的若干非零值置 0
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        return data.copy(), np.zeros(0, dtype=bool)

    target_zero_ratio = float(target_zero_ratio)
    target_zero_ratio = min(max(target_zero_ratio, 0.0), 1.0)

    current_mask = (data == 0)
    current_zeros = int(current_mask.sum())
    target_zeros = int(round(n * target_zero_ratio))

    if target_zeros <= current_zeros:
        # 已经达到或超过目标：不减少零（保持非负约束/物理约束结果）
        return data.copy(), current_mask

    need = target_zeros - current_zeros

    # 在非零位置里选最小的 need 个
    idx_nonzero = np.where(~current_mask)[0]
    if idx_nonzero.size == 0:
        return data.copy(), current_mask

    vals = data[idx_nonzero]
    order = np.argsort(vals)  # 小到大
    chosen = idx_nonzero[order[:need]]

    out = data.copy()
    mask = current_mask.copy()
    out[chosen] = 0.0
    mask[chosen] = True
    return out, mask
def inject_zeros_latent_threshold(data, target_zero_ratio):
    """
    潜变量阈值法 (Latent Thresholding) 注入零值：
    借用数据的相对大小（分布）来决定零值位置，保留时间序列的自相关性。
    
    逻辑：
    - 找出数据的第 P 分位数 (Threshold)，其中 P = target_zero_ratio。
    - 所有小于等于 Threshold 的值置为 0。
    - 这等价于：将最小的 N * target_ratio 个数置为 0。
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if n == 0:
        return data.copy(), np.zeros(0, dtype=bool)

    target_zero_ratio = float(target_zero_ratio)
    target_zero_ratio = min(max(target_zero_ratio, 0.0), 1.0)
    
    # 目标零值数量
    target_zeros = int(round(n * target_zero_ratio))
    
    # 当前零值数量 (经过分位数校准后，可能已经有一些 0 或 接近 0 的值)
    current_mask = (data <= 1e-9) # 视为 0
    current_zeros = int(current_mask.sum())

    # 如果目标是 0，直接返回 (可能做一些清理)
    if target_zeros == 0:
        return data.copy(), np.zeros(n, dtype=bool)

    # 潜变量排序：获取所有数据的索引，按值从小到大排序
    # 注意：我们对整个序列排序，而不是只针对非零值，这样能保证全局的分布一致性
    sorted_indices = np.argsort(data)
    
    # 确定需要置 0 的索引：最小的前 target_zeros 个
    # 即使 data 中已经有 0，它们也会排在最前面，被包含在 indices_to_zero 中
    # 这样确保了这一步是 "Enforce" (强制) 零值比例，无论是增加还是修剪
    indices_to_zero = sorted_indices[:target_zeros]
    
    out = data.copy()
    
    # 执行注入
    out[indices_to_zero] = 0.0
    
    # 生成掩码
    final_mask = np.zeros(n, dtype=bool)
    final_mask[indices_to_zero] = True
    
    return out, final_mask

# ============================================================
# 3) 物理约束（按你的新要求定制）
#    - [MOD-NONNEG] 所有特征最终必须非负
#    - [MOD-CONPR] conpr* 为压缩比：默认值 1，且建议下界为 1（更符合“压缩比”定义）--已调整为1-压缩比，取值范围[0,1]，越小表示压缩比越大
# ============================================================

def apply_physical_constraints_nonneg(data, feature_name, feature_type, feature_limits=None):
    """
    feature_type:
      - 'count'            : 非负、四舍五入为整数
      - 'ratio_01'         : 截断到 [0,1]
    #   - 'compression_ratio': conpr* 压缩比，默认 1，下界 1（可选上界）--并入ratio
      - 'continuous'       : 连续非负（>=0），可选上界
    """
    data = np.asarray(data, dtype=float)
    out = data.copy()
    info = {
        'n_nan_inf_filled': 0,
        'n_negative_clipped': 0,
        'n_lower_clipped': 0,
        'n_upper_clipped': 0
    }

    # conpr 特殊：默认填充值为 1，其它默认 0
    # default_fill = 1.0 if str(feature_name).startswith('conpr') else 0.0
    default_fill = 0.0


    # 先处理 NaN/Inf（用默认值）
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = default_fill
        info['n_nan_inf_filled'] = int(bad.sum())

    # 全局非负要求：先把 <0 的裁到 0（或后面按类型裁到 1）
    neg = out < 0
    if neg.any():
        out[neg] = 0.0
        info['n_negative_clipped'] = int(neg.sum())

    # 类型约束
    feature_limits = feature_limits or {}

    if feature_type == 'count':
        # 非负整数
        out = np.round(out)
        # 可选上界
        maxv = feature_limits.get('max_value', None)
        if maxv is not None:
            maxv = float(maxv)
            upper = out > maxv
            if upper.any():
                out[upper] = maxv
                info['n_upper_clipped'] += int(upper.sum())

    elif feature_type == 'ratio_01':
        lower = out < 0
        upper = out > 1
        if lower.any():
            out[lower] = 0.0
            info['n_lower_clipped'] += int(lower.sum())
        if upper.any():
            out[upper] = 1.0
            info['n_upper_clipped'] += int(upper.sum())

    # elif feature_type == 'compression_ratio':
    #     # [MOD-CONPR] 默认 1，且建议压缩比下界为 1（不允许小于 1）
    #     lower_bound = float(feature_limits.get('min_value', 1.0))
    #     lower = out < lower_bound
    #     if lower.any():
    #         out[lower] = lower_bound
    #         info['n_lower_clipped'] += int(lower.sum())

        maxv = feature_limits.get('max_value', None)
        if maxv is not None:
            maxv = float(maxv)
            upper = out > maxv
            if upper.any():
                out[upper] = maxv
                info['n_upper_clipped'] += int(upper.sum())

    else:  # 'continuous'
        # 连续非负（>=0）
        lower_bound = float(feature_limits.get('min_value', 0.0))
        lower = out < lower_bound
        if lower.any():
            out[lower] = lower_bound
            info['n_lower_clipped'] += int(lower.sum())

        maxv = feature_limits.get('max_value', None)
        if maxv is not None:
            maxv = float(maxv)
            upper = out > maxv
            if upper.any():
                out[upper] = maxv
                info['n_upper_clipped'] += int(upper.sum())

    # 再次确保非负
    out = np.maximum(out, 0.0)

    # conpr：如果还出现 0（例如全为 NaN），强制成 1
    if str(feature_name).startswith('conpr'):
        out = np.where(out <= 0, 1.0, out)

    return out, info


# ============================================================
# 4) 主流程：逆变换 -> 物理约束 -> 零注入（只补足） -> 再约束
#    - [MOD-ORDER] 零注入放在约束之后，避免 rounding/clip 改变零比例
#    - [MOD-CONPR-ZERO] conpr* 不注入零（默认 1）
# ============================================================

def inverse_transform_and_inject_zeros_nonneg(
    df_synthetic_final,       # Step5 输出（变换域）
    feature_list,
    step1_results,            # 需含 zero_ratio（目标零比例）、可含 mean/std 等
    clean_stats,              # 用于分位数校准
    transform_info,           # 你的 transform_info（含 transform, lambda, original_mean/std 等）
    feature_config=None,      # {'count':[...], 'ratio_01':[...], 'continuous':[...]}；conpr 自动识别
    zero_injection_method=None,  # 这里只实现最稳的 exact-only-add
    # zero_injection_method='latent',  # 这里只实现最稳的 exact-only-add

    output_dir=None,
    safety_margin=1.5  # [新增] 安全系数，允许合成数据是历史最大值的 1.5 倍
):
    os.makedirs(output_dir, exist_ok=True)

    # 默认类型映射（你可覆盖）
    if feature_config is None:
        feature_config = {
            'count': [],
            'ratio_01': [],
            'continuous': []
        }

    # 构建 feature -> type 映射
    feature_type_map = {}
    for t, feats in feature_config.items():
        for f in feats:
            feature_type_map[str(f)] = t

    # conpr* 强制指定为 compression_ratio
    # for f in feature_list:
    #     if str(f).startswith('conpr'):
    #         feature_type_map[str(f)] = 'compression_ratio'

    print("═" * 70)
    print("📌 Step4：逆变换 + 零值注入 + 物理约束（全非负）")
    print("═" * 70)

    time_index = df_synthetic_final.index
    n = len(time_index)

    df_physical = pd.DataFrame(index=time_index)
    report = {
        'inverse_stats': [],
        'zero_stats': [],
        'constraint_stats': []
    }

    print(f"开始逆变换与分位数校准，共 {len(feature_list)} 个特征...")

    for i, feature in enumerate(feature_list):
        if feature not in df_synthetic_final.columns:
            print(f"  ⚠️ {feature} 不在 df_synthetic_final，跳过")
            continue
        # 1. 获取变换参数
        trans = transform_info.get(feature, {}) if transform_info else {}
        method = trans.get('transform', 'none')
        y = df_synthetic_final[feature].to_numpy(dtype=float)

        # 2. 计算安全上界 (Safety Upper Bound)
        # 从 step1_results 获取历史最大值
        hist_stats = step1_results.get(feature, {})
        hist_max = hist_stats.get('max', 1e6) 
        if pd.isna(hist_max): hist_max = 1e6


        # 历史统计量
        hist_mean = float(hist_stats.get('mean', 0.0))
        hist_std = float(hist_stats.get('std', 1.0))
        hist_max = float(hist_stats.get('max', 1.0))
        hist_zero_ratio = float(hist_stats.get('zero_ratio', 0.0))
        
        # 设定硬顶：历史最大值 * 系数 (防止 10^18 爆炸)
        # 对于比率类(ratio)，最大值不应超过 1.0太多(如果是0-1)，或者由历史决定
        # 给一个保底值 10.0，防止全0数据的倍数依然是0
        safe_max_val = float(hist_max) * safety_margin
        
        # 特殊处理 ratio_01 类型，物理上限是 1.0
        ftype = feature_type_map.get(feature, 'continuous')
        if ftype == 'ratio_01':
            safe_max_val = 1.0
                

        # 4. 执行逆变换
        # ---------- (A) 输入安全裁剪（只对 log1p/boxcox 做 log 类裁剪） ----------
        # [MOD] 不把 yeo_johnson 当 log 类截断
        if method == 'log1p':
            y = np.clip(y, -15.0, 15.0)
        elif method == 'boxcox':
            # boxcox 若 lambda≈0 等价 log
            lmbda = float(trans.get('lambda', 1.0))
            if abs(lmbda) < 1e-10:
                y = np.clip(y, -15.0, 15.0)
            # 其它 lambda 不强行 log 截断（但 inverse 里会确保 base>0）
        elif method == 'standardize':
            y = np.clip(y, -50.0, 50.0)

        # ---------- (B) 逆变换 ----------
        if method == 'yeo_johnson':
            lmbda = trans.get('lambda', 0.0)
            lmbda = 0.0 if lmbda is None else float(lmbda)
            x = inverse_yeo_johnson_correct(y, lmbda)

        elif method == 'log1p':
            x = inverse_log1p(y)

        elif method == 'sqrt':
            x = inverse_sqrt(y)

        elif method == 'standardize':
            orig_mean = trans.get('original_mean', step1_results.get(feature, {}).get('mean', 0.0))
            orig_std = trans.get('original_std', step1_results.get(feature, {}).get('std', 1.0))
            orig_mean = 0.0 if orig_mean is None else float(orig_mean)
            orig_std = 1.0 if (orig_std is None or float(orig_std) == 0.0) else float(orig_std)
            x = inverse_standardize(y, orig_mean, orig_std)

        elif method == 'boxcox':
            lmbda = float(trans.get('lambda', 1.0))
            shift = float(trans.get('shift', 0.0))
            x = inverse_boxcox(y, lmbda, shift)

        else:
            x = y.copy()
        print(f"  🔄 逆变换 {feature:<20}: Method={method}, Safety Max={safe_max_val:.2f}")
    #     df_physical[feature] = x
    # return df_physical, report
    # def function_tmp():    
        
        
        # --- (add) 分位数校准 (Quantile Calibration) ---
        target_stats = clean_stats.get(feature, step1_results.get(feature, {}))
        is_bad_feature = feature in ['total_long_post', 'retweet_ratio_post', 'total_short_comment']
        use_robust = not is_bad_feature
        x = quantile_calibrate(feature, x, target_stats, robust=use_robust)

        # if ftype != 'ratio_01': 
        #     x = calibrate_distribution(x, hist_mean, hist_std, hist_max, feature)

        # 后截断 (Post-clipping): 再次确保不爆炸 20260212new
        # 如果逆变换后的值超过了安全上界，强行拉回
        mask_explode = x > safe_max_val
        if mask_explode.any():
            # 记录一下被截断的数量
            n_clip = mask_explode.sum()
            # 仅在大量截断或极值过大时打印，避免刷屏
            if x[mask_explode].max() > safe_max_val * 2:
                print(f"  ⚠️ {feature:<25}: 触发防爆截断! Max {x.max():.2e} -> {safe_max_val:.2f} ({n_clip} pts)")
            x[mask_explode] = safe_max_val

        

        # ---------- (C) 第一次物理约束（保证全非负） ----------
        ftype = feature_type_map.get(feature, 'continuous')
        # 为个别特征加上 max_value 等限制
        limits = {}
        if feature in ['total_short_comment', 'total_long_post' ,'retweet_ratio_post']:    
            limits['max_value'] = target_stats['percentiles'][99]
        elif ftype == 'ratio_01' or 'ratio' in feature:
            limits['max_value'] =  min(hist_max,1.0)
            # Ratio 类型绝对不能超过 1.0 (或 clean_stats 里的最大值)
            # 如果 clean_stats['max'] < 1.0 (比如 0.8)，就截断到 0.8，防止尾部过冲
        else:
            limits['max_value'] = hist_max * safety_margin
           # 注入零后仍可能有极大值，确保后续约束能处理
        x, c_info = apply_physical_constraints_nonneg(
            x, feature_name=feature, feature_type=ftype, feature_limits=limits
        )
        

        # ---------- (D) 零值注入（只补足；conpr 不注入零） ----------
        target_zero = float(step1_results.get(feature, {}).get('zero_ratio', 0.0))
        # target_zero = hist_zero_ratio
        target_zero = min(max(target_zero, 0.0), 1.0)

        # 只在目标零比例>0 时尝试补足
        if target_zero > 0 and zero_injection_method.startswith('threshold'):
            injected, zmask = inject_zeros_threshold_exact_only_add(x, target_zero)
            note = f"threshold_extract (target={target_zero:.1%})"


        else:
            injected, zmask = inject_zeros_latent_threshold(x, target_zero)
            note = f"latent_threshold (target={target_zero:.1%})"

        actual_zero = float(zmask.mean())


        if ftype != 'ratio_01':
        # 第一轮修正
            injected = force_mean_correction(injected, hist_mean, tolerance=0.01)
        
        # # 如果是 Count 类型，取整后均值会变，需要迭代检查
        # if ftype == 'count':
        #     # 预先取整看效果
        #     injected_rounded = np.round(injected)
        #     current_mean = np.mean(injected_rounded)
            
        #     # 如果取整后均值依然偏差 > 2%，进行第二轮反向补偿
        #     if hist_mean > 1e-3 and abs(current_mean - hist_mean) / hist_mean > 0.02:
        #         # 计算补偿系数。例如：目标 2.0，取整后变成 2.2。说明我要把未取整的数据再压低一点
        #         # scale = 2.0 / 2.2 = 0.909
        #         scale = hist_mean / (current_mean + 1e-9)
        #         injected = injected * scale # 对未取整的数据应用补偿
        #         # print(f"  🔄 迭代均值修正 {feature}: Scale {scale:.4f}")

        
        # ---------- (E) 再次物理约束（避免注入后边界问题；count 需要保持整数） ----------
        limits['max_value'] = hist_max * safety_margin  # 注入零后仍可能有极大值，确保后续约束能处理
        injected2, c_info2 = apply_physical_constraints_nonneg(
            injected, feature_name=feature, feature_type=ftype, feature_limits=limits
        )
        print(f"  🧾 再次物理约束{feature:<20}: {np.max(injected):.2f}->{np.max(injected2):.2f}")

        if ftype == 'count':
            injected2 = np.round(injected2)
        
        # 最后的防线：确保非负
        injected2 = np.maximum(injected2, 0.0)
        if ftype == 'ratio_01': injected2 = np.minimum(injected2, 1.0)

        df_physical[feature] = injected2

        # ---------- 记录统计 ----------
        yv = y[np.isfinite(y)]
        xv = x[np.isfinite(x)]
        pv = injected2[np.isfinite(injected2)]

        report['inverse_stats'].append({
            'feature': feature,
            'transform': method,
            'y_mean': float(np.mean(yv)) if yv.size else np.nan,
            'y_std': float(np.std(yv)) if yv.size else np.nan,
            'x_mean_after_inverse_before_zero': float(np.mean(xv)) if xv.size else np.nan,
            'x_std_after_inverse_before_zero': float(np.std(xv)) if xv.size else np.nan
        })
        report['zero_stats'].append({
            'feature': feature,
            'target_zero_ratio': float(target_zero),
            'actual_zero_ratio': float((pv == 0).mean()) if pv.size else np.nan,
            'note': note
        })
        report['constraint_stats'].append({
            'feature': feature,
            'type': ftype,
            **c_info,
            # 第二次约束也记录一下（可选）
            'n_nan_inf_filled_after_zero': int(c_info2.get('n_nan_inf_filled', 0)),
            'n_negative_clipped_after_zero': int(c_info2.get('n_negative_clipped', 0)),
            'n_lower_clipped_after_zero': int(c_info2.get('n_lower_clipped', 0)),
            'n_upper_clipped_after_zero': int(c_info2.get('n_upper_clipped', 0)),
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(feature_list):
            print(f"  ✅ 已处理 {i+1}/{len(feature_list)}")

    # 保存
    df_physical.to_csv(f'{output_dir}/synthetic_physical.csv', encoding='utf-8-sig')
    with open(f'{output_dir}/inverse_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "═" * 70)
    print("✅ Step6 完成：已输出 synthetic_physical.csv（全非负）")
    print("═" * 70)

    return df_physical, report


# ============================================================
# 5) 运行封装
# ============================================================

def run_step6_inverse_transform(
    df_synthetic_final,
    feature_list,
    step1_results,
    clean_stats,
    transform_info,
    zero_injection_method=None,
    feature_config=None,
    output_dir=None
):
    """
    说明：
    - df_synthetic_final 必须是“变换域”的合成结果（来自你修正版 Step3）
    - transform_info 里每个 feature 至少应有 transform 字段；yeo_johnson/boxcox 要有 lambda
    - step1_results 里每个 feature 推荐有 zero_ratio（原始数据的零比例）
    - 所有最终数据强制非负
    - conpr* 强制为1-压缩比：默认 0
    """
    if feature_config is None:
        feature_config = {
            'count': [
                'total_volume_post', 'total_volume_comment',
                'total_short_post', 'total_long_post',
                'total_short_comment', 'total_long_comment',
                'vis_abs_redundancy_post',
                'senti_symbol_post', 'senti_symbol_comment'
            ],
            'ratio_01': [
                'gini_post', 'gini_comment','retweet_ratio_post',
                'origin_ratio_post', 'neg_ratio_post', 'neg_ratio_comment',
                'vis_concentration_post', 'comp_ratio_post', 'comp_ratio_comment'
            ],
            'continuous': [
                'semantic_shift_post', 'semantic_shift_comment'
                
            ]
        }

    return inverse_transform_and_inject_zeros_nonneg(
        df_synthetic_final=df_synthetic_final,
        feature_list=feature_list,
        step1_results=step1_results,
        clean_stats=clean_stats,
        transform_info=transform_info,
        feature_config=feature_config,
        zero_injection_method=zero_injection_method,
        output_dir=output_dir
    )

