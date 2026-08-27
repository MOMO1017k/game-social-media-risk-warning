import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logit, expit
from scipy.stats import boxcox, yeojohnson
import warnings
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 变换函数库 ====================

def safe_log1p(x):
    """安全的log1p变换"""
    x = np.array(x, dtype=float)
    x = np.where(x < 0, 0, x)
    x = np.where(np.isinf(x), np.nan, x)
    return np.log1p(x)


def safe_sqrt(x):
    """安全的平方根变换"""
    x = np.array(x, dtype=float)
    x = np.where(x < 0, 0, x)
    x = np.where(np.isinf(x), np.nan, x)
    return np.sqrt(x)


def safe_logit(x, epsilon=1e-6):
    """安全的Logit变换（带边界保护）"""
    x = np.array(x, dtype=float)
    x = np.clip(x, epsilon, 1 - epsilon)
    x = np.where(np.isnan(x), 0.5, x)
    x = np.where(np.isinf(x), 0.5, x)
    return logit(x)


def standardize(x):
    """Z-score标准化（不改变偏度）"""
    x = np.array(x, dtype=float)
    valid = x[~np.isnan(x) & ~np.isinf(x)]
    if len(valid) == 0:
        return x
    mean_val = valid.mean()
    std_val = valid.std()
    if std_val == 0:
        std_val = 1
    return (x - mean_val) / std_val


def safe_boxcox(x, shift=1e-6):
    """
    安全的Box-Cox变换
    要求数据严格为正
    """
    x = np.array(x, dtype=float)
    valid_mask = ~np.isnan(x) & ~np.isinf(x)
    
    if valid_mask.sum() < 30:
        return standardize(x), None
    
    # Box-Cox要求正值，平移数据
    x_valid = x[valid_mask]
    x_positive = x_valid - x_valid.min() + shift
    
    try:
        transformed, lambda_opt = boxcox(x_positive)
        
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
        print(f"      Box-Cox失败: {e}")
        return standardize(x), None


def safe_yeo_johnson(x):
    """
    Yeo-Johnson变换
    支持负值和零值，比Box-Cox更通用
    """
    x = np.array(x, dtype=float)
    valid_mask = ~np.isnan(x) & ~np.isinf(x)
    
    if valid_mask.sum() < 30:
        return standardize(x), None
    
    try:
        x_valid = x[valid_mask]
        transformed, lambda_opt = yeojohnson(x_valid)
        
        # 应用到全部数据
        result = np.full_like(x, np.nan)
        result[valid_mask] = transformed
        
        return result, lambda_opt
    except Exception as e:
        print(f"      Yeo-Johnson失败: {e}")
        return standardize(x), None



# ==================== 智能变换选择器（更新版） ====================

def choose_best_transform_v2(data, feature_name, ftype, zero_ratio):
    """
    根据数据特性智能选择最佳变换（更新版）
    
    决策逻辑：
    1. 偏度在[-0.5, 0.5]：直接标准化
    2. 偏度在[-1, 1]：优先标准化，除非明确需要
    3. 偏度 > 1：尝试log1p, sqrt, yeo-johnson
    4. 比率型：避免logit（边界敏感）
    5. 高零膨胀：log1p + 零值指示
    """
    valid_data = data[~np.isnan(data) & ~np.isinf(data)]
    
    if len(valid_data) < 30:
        return 'none', data, {'reason': '数据不足'}
    
    original_skew = stats.skew(valid_data)
    
    # ========== 规则1: 偏度已经很好 ==========
    if abs(original_skew) <= 0.5:
        return 'standardize', standardize(data), {
            'reason': f'原始偏度{original_skew:.2f}已接近正态(±0.5)，仅标准化',
            'original_skew': original_skew,
            'transformed_skew': original_skew  # 标准化不改变偏度
        }
    
    # ========== 规则2: 偏度可接受 ==========
    if abs(original_skew) <= 1.0:
        return 'standardize', standardize(data), {
            'reason': f'原始偏度{original_skew:.2f}可接受(±1.0)，仅标准化以避免过度校正',
            'original_skew': original_skew,
            'transformed_skew': original_skew
        }
    
    # ========== 规则3: 需要变换（偏度 > 1） ==========
    candidates = {}
    
    # 候选1: sqrt (最温和)
    if valid_data.min() >= 0:
        sqrt_data = safe_sqrt(valid_data)
        sqrt_skew = stats.skew(sqrt_data[~np.isnan(sqrt_data)])
        candidates['sqrt'] = {
            'data': safe_sqrt(data),
            'skew': sqrt_skew,
            'improvement': abs(original_skew) - abs(sqrt_skew)
        }
    
    # 候选2: log1p (中等强度)
    if valid_data.min() >= 0:
        log_data = safe_log1p(valid_data)
        log_skew = stats.skew(log_data[~np.isnan(log_data)])
        candidates['log1p'] = {
            'data': safe_log1p(data),
            'skew': log_skew,
            'improvement': abs(original_skew) - abs(log_skew)
        }

    # 候选3: Logit (新增，仅限低零值 Ratio)
    # 理由：高零值数据的 0 会被映射为 logit(epsilon)≈-14，成为离群值，破坏正态性。
    if ftype == 'ratio' and zero_ratio < 0.05 and valid_data.min() >= 0 and valid_data.max() <= 1:
        logit_data = safe_logit(data)
        logit_valid = logit_data[~np.isnan(logit_data) & ~np.isinf(logit_data)]
        if len(logit_valid) > 0:
            candidates['logit'] = {
                'data': logit_data,
                'skew': stats.skew(logit_valid)
            }
    
    # 候选4: Yeo-Johnson (自适应)
    yj_data, yj_lambda = safe_yeo_johnson(valid_data)
    if yj_lambda is not None:
        yj_full, _ = safe_yeo_johnson(data)
        yj_skew = stats.skew(yj_data[~np.isnan(yj_data)])
        candidates['yeo_johnson'] = {
            'data': yj_full,
            'skew': yj_skew,
            'lambda': yj_lambda,
            'improvement': abs(original_skew) - abs(yj_skew)
        }
    
    # 候选5: Box-Cox (仅对严格正值)
    if valid_data.min() > 0:
        bc_data, bc_lambda = safe_boxcox(valid_data)
        if bc_lambda is not None:
            bc_full, _ = safe_boxcox(data)
            bc_skew = stats.skew(bc_data[~np.isnan(bc_data)])
            candidates['boxcox'] = {
                'data': bc_full,
                'skew': bc_skew,
                'lambda': bc_lambda,
                'improvement': abs(original_skew) - abs(bc_skew)
            }
    
    # ========== 选择最佳变换 ==========
    if not candidates:
        return 'standardize', standardize(data), {
            'reason': '无可用变换，使用标准化',
            'original_skew': original_skew,
            'transformed_skew': original_skew
        }
    
    # 选择标准：
    # 1. 偏度改善最大
    # 2. 但不能过度校正（变换后偏度反向且绝对值更大）
    
    best_transform = None
    best_improvement = -np.inf
    
    for name, info in candidates.items():
        # 检查是否过度校正
        new_skew = info['skew']
        if new_skew * original_skew < 0 and abs(new_skew) > 0.5:
            # 偏度反向且绝对值超过0.5，视为过度校正
            continue
        
        if info['improvement'] > best_improvement:
            best_improvement = info['improvement']
            best_transform = name
    
    if best_transform is None:
        # 所有变换都过度校正，使用最温和的sqrt
        if 'sqrt' in candidates:
            best_transform = 'sqrt'
        else:
            return 'standardize', standardize(data), {
                'reason': '所有变换都过度校正，使用标准化',
                'original_skew': original_skew,
                'transformed_skew': original_skew
            }
    
    result = candidates[best_transform]
    
    return best_transform, result['data'], {
        'reason': f'偏度从{original_skew:.2f}改善到{result["skew"]:.2f}',
        'original_skew': original_skew,
        'transformed_skew': result['skew'],
        'improvement': result['improvement'],
        'lambda': result.get('lambda')
    }

# ==================== 提取ACF签名 ====================
def extract_acf_signature(arr, lags_short=40, seasonal_lags=(96, 192, 672)):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 300:
        return {'ok': False, 'n': int(n)}

    nlags = int(max([lags_short, *seasonal_lags]))
    acf_vals = sm_acf(arr, nlags=nlags, fft=True)
    pacf_vals = sm_pacf(arr, nlags=min(nlags, 200), method='yw')

    conf = 1.96 / np.sqrt(n)

    return {
        'ok': True,
        'n': int(n),
        'conf_95_approx': float(conf),
        'acf_short': {str(k): float(acf_vals[k]) for k in range(1, lags_short + 1)},
        'pacf_short': {str(k): float(pacf_vals[k]) for k in range(1, min(lags_short, len(pacf_vals)-1) + 1)},
        'acf_seasonal': {str(k): float(acf_vals[k]) for k in seasonal_lags if k < len(acf_vals)},
        'significant_lags_short': [k for k in range(1, lags_short + 1) if abs(acf_vals[k]) > conf],
    }


import re
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def fit_autoreg_params(arr, lags=(1, 4, 96), trend='c'):
    """
    用 AutoReg 拟合 AR 参数（供 Step3 用）。
    兼容不同 statsmodels 版本：res.params 可能是 ndarray 或 Series。
    返回:
      - phi: {1:...,4:...,96:...}
      - const
      - sigma2
      - param_map: 全量参数名->值（便于排查）
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    lags = sorted(set(int(x) for x in lags if int(x) > 0))

    if n < max(lags) + 50:
        return {'ok': False, 'n': int(n), 'reason': 'too_short'}

    try:
        res = AutoReg(arr, lags=lags, trend=trend, old_names=False).fit()

        # 关键修复：不同版本 params 可能是 ndarray；必须用“名字+数值”构造映射
        names = None
        if hasattr(res, "param_names"):
            names = list(res.param_names)
        elif hasattr(res.model, "exog_names"):
            names = list(res.model.exog_names)
        else:
            names = [f"p{i}" for i in range(len(res.params))]

        values = np.asarray(res.params, dtype=float)
        param_map = {k: float(v) for k, v in zip(names, values)}

        # 提取 const（不同版本可能叫 const / intercept）
        const = 0.0
        for k in ("const", "intercept"):
            if k in param_map:
                const = float(param_map[k])
                break

        # 提取各 lag 系数（兼容 y.L1 / L1.y 等命名）
        phi = {}
        for L in lags:
            patterns = [
                rf"^y\.L{L}$",     # old_names=False 常见
                rf"^L{L}\.y$",     # old_names=True 常见
                rf"^L{L}$",        # 兜底
            ]
            found = None
            for name in param_map.keys():
                if any(re.match(p, name) for p in patterns):
                    found = name
                    break
            if found is not None:
                phi[int(L)] = float(param_map[found])

        return {
            'ok': True,
            'n': int(n),
            'trend': trend,
            'lags': lags,
            'const': float(const),
            'phi': phi,
            'sigma2': float(getattr(res, "sigma2", np.nan)),
            'param_map': param_map,   # 建议保留，方便你核对参数名
        }

    except Exception as e:
        return {'ok': False, 'n': int(n), 'reason': str(e)}


# ==================== 主变换流程（V3版） ====================

def apply_transformations_v3(df, step1_results, TRANSFORM_OVERRIDES_V2, output_dir):
    """
    步骤2（V3版）：智能变换
    
    核心改进：
    1. 偏度≤1的特征只做标准化
    2. 完全避免Logit（边界敏感导致过度校正）
    3. 优先使用温和变换（sqrt > log1p > yeo-johnson）
    4. 特殊规则覆盖问题特征
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df_transformed = df.copy()
    transform_info = {}
    
    print("="*70)
    print("📌 步骤2（V3版）：智能数据变换")
    print("="*70)
    
    print("\n📋 特殊规则覆盖:")
    for feat, config in TRANSFORM_OVERRIDES_V2.items():
        print(f"   • {feat}: {config['transform']}")
        print(f"     └─ {config['reason']}")
    print()
    
    for feature, result in step1_results.items():
        if feature not in df.columns:
            continue
        
        ftype = result['type']
        zero_ratio = result['zero_ratio']
        original_data = df[feature].replace([np.inf, -np.inf], np.nan).values
        valid_data = original_data[~np.isnan(original_data)]
        
        if len(valid_data) < 10:
            continue
        
        original_skew = stats.skew(valid_data)
        
        print(f"\n{'='*55}")
        print(f"🔧 处理特征: {feature}")
        print(f"   类型: {ftype}, 零值比例: {zero_ratio:.1%}")
        print(f"   原始偏度: {original_skew:.3f}")
        
        # ========== 检查特殊规则覆盖 ==========
        lambda_opt = None
        if feature in TRANSFORM_OVERRIDES_V2:
            config = TRANSFORM_OVERRIDES_V2[feature]
            override_transform = config['transform']
            override_reason = config['reason']
            
            print(f"   ⚠️ 应用特殊规则: {override_transform}")

            if override_transform == 'sqrt':
                transformed = safe_sqrt(original_data)
            elif override_transform == 'log1p':
                transformed = safe_log1p(original_data)
            elif override_transform == 'standardize':
                transformed = standardize(original_data)
            elif override_transform == 'yeo_johnson':
                transformed, lambda_opt = safe_yeo_johnson(original_data)
                # transform_info[feature]['lambda'] = None if lambda_opt is None else float(lambda_opt)
                print(f"      Yeo-Johnson λ = {lambda_opt:.3f}" if lambda_opt else "")
            elif override_transform == 'boxcox':
                transformed, lambda_opt = safe_boxcox(original_data)
                print(f"      Box-Cox λ = {lambda_opt:.3f}" if lambda_opt else "")
            else:
                transformed = original_data
            
            # 计算变换后偏度
            trans_valid = transformed[~np.isnan(transformed)]
            trans_skew = stats.skew(trans_valid) if len(trans_valid) > 0 else original_skew
            
            df_transformed[f'{feature}_transformed'] = transformed
            transform_info[feature] = {
                'type': ftype,
                'transform': override_transform,
                'reason': override_reason,
                'original_skew': original_skew,
                'transformed_skew': trans_skew,
                'improvement': abs(original_skew) - abs(trans_skew),
                'is_override': True,
                'lambda': None if lambda_opt is None else float(lambda_opt)
            }
            
        else:
            # ========== 智能选择变换 ==========
            best_transform, transformed, info = choose_best_transform_v2(
                original_data, feature, ftype, zero_ratio
            )
            
            print(f"   ✅ 自动选择: {best_transform}")
            print(f"      {info.get('reason', '')}")
            
            df_transformed[f'{feature}_transformed'] = transformed
            
            transform_info[feature] = {
                'type': ftype,
                'transform': best_transform,
                'original_skew': info.get('original_skew', original_skew),
                'transformed_skew': info.get('transformed_skew', original_skew),
                'improvement': info.get('improvement', 0),
                'reason': info.get('reason', ''),
                'is_override': False,
                'lambda': info.get('lambda', 0.0)

            }
            
        # ========== 新增：提取 transformed 序列的 ACF 目标参数 ==========
        trans_col = f'{feature}_transformed'
        trans_series = df_transformed[trans_col].replace([np.inf, -np.inf], np.nan).dropna()

        # 只在长度足够时计算
        if len(trans_series) >= 300:
            acf_sig = extract_acf_signature(trans_series.values, lags_short=40, seasonal_lags=(96, 192, 672))
            ar_fit  = fit_autoreg_params(trans_series.values, lags=(1, 4, 96), trend='c')

            transform_info[feature]['acf_signature_transformed'] = acf_sig
            transform_info[feature]['ar_params_transformed_target'] = ar_fit



            # 可选：打印关键 lag
            if acf_sig.get('ok', False):
                a = acf_sig['acf_seasonal']
                s = acf_sig['acf_short']
                print(f"   📌 ACF(target): lag1={float(s.get('1', np.nan)):.3f}, lag4={float(s.get('4', np.nan)):.3f}, lag96={float(a.get('96', np.nan)):.3f}")
        else:
            transform_info[feature]['acf_signature_transformed'] = {'ok': False, 'n': int(len(trans_series))}
            transform_info[feature]['ar_params_transformed_target'] = {'ok': False, 'n': int(len(trans_series))}
        
        # ========== 高零膨胀：添加零值指示变量 ==========
        if zero_ratio > 0.5:
            is_zero = (df[feature].values == 0).astype(int)
            df_transformed[f'{feature}_is_zero'] = is_zero
            transform_info[feature]['has_zero_indicator'] = True
            print(f"   📍 添加零值指示变量（零值比例 {zero_ratio:.1%}）")
        
        # 打印变换后偏度
        trans_col = f'{feature}_transformed'
        trans_valid = df_transformed[trans_col].replace([np.inf, -np.inf], np.nan).dropna()
        # final_skew = stats.skew(trans_valid) if len(trans_valid) > 0 else np.nan
        if len(trans_valid) > 0:
            final_mean = float(trans_valid.mean())
            final_std = float(trans_valid.std())
            final_skew = float(stats.skew(trans_valid))
        else:
            final_mean = np.nan
            final_std = np.nan
            final_skew = np.nan

        transform_info[feature]['transformed_mean'] = final_mean
        transform_info[feature]['transformed_std'] = final_std

        # print(f"   变换后偏度: {final_skew:.3f}")
        print(f"   📊 统计: Mean={final_mean:.3f}, Std={final_std:.3f}, Skew={final_skew:.3f}")
        
        # 评估
        improvement = abs(original_skew) - abs(final_skew)
        if improvement > 0:
            print(f"   📈 偏度改善: {improvement:.3f}")
        elif abs(final_skew) <= 1.0:
            print(f"   ✅ 偏度可接受 (|{final_skew:.2f}| ≤ 1)")
        else:
            print(f"   ⚠️ 偏度仍较大，但已是最优变换")
    
    # ========== 绑定绘图 ==========
    plot_transformation_comparison_v3(df, df_transformed, step1_results, transform_info, output_dir)
    
    # 保存变换信息
    transform_summary = []
    for feat, info in transform_info.items():
        trans_skew = info['transformed_skew']
        orig_skew = info['original_skew']
        improvement = abs(orig_skew) - abs(trans_skew)
        
        # 评估状态
        if abs(trans_skew) <= 0.5:
            status = '✅优秀'
        elif abs(trans_skew) <= 1.0:
            status = '✅良好'
        elif improvement > 0:
            status = '⚠️改善'
        else:
            status = '❌未改善'
        
        transform_summary.append({
            '特征': feat,
            '类型': info['type'],
            '变换方法': info['transform'],
            '原始偏度': f"{orig_skew:.3f}",
            '变换偏度': f"{trans_skew:.3f}",
            '偏度改善': f"{improvement:.3f}",
            '状态': status,
            '零值指示': '✅' if info.get('has_zero_indicator') else '',
            '规则覆盖': '✅' if info.get('is_override') else ''
        })
    
    summary_df = pd.DataFrame(transform_summary)
    summary_df.to_csv(f'{output_dir}/变换信息汇总.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ 步骤2（V3版）完成！")
    print(f"📁 结果保存至: {output_dir}")
    print("="*70)
    
    # 打印汇总表
    print("\n📋 变换汇总表:")
    display_cols = ['特征', '变换方法', '原始偏度', '变换偏度', '状态', '规则覆盖']
    print(summary_df[display_cols].to_string(index=False))
    
    # 统计
    n_excellent = (summary_df['状态'] == '✅优秀').sum()
    n_good = (summary_df['状态'] == '✅良好').sum()
    n_improved = (summary_df['状态'] == '⚠️改善').sum()
    n_failed = (summary_df['状态'] == '❌未改善').sum()
    
    print(f"\n📊 统计: 优秀{n_excellent} | 良好{n_good} | 改善{n_improved} | 未改善{n_failed}")
    
    return df_transformed, transform_info, summary_df


def plot_transformation_comparison_v3(df_original, df_transformed, step1_results, transform_info, output_dir):
    """绘制变换前后对比图（V3版）"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    features = list(transform_info.keys())
    n_features = len(features)
    
    # 每页显示4个特征
    features_per_page = 4
    n_pages = (n_features + features_per_page - 1) // features_per_page
    
    for page in range(n_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, n_features)
        page_features = features[start_idx:end_idx]
        
        fig, axes = plt.subplots(len(page_features), 4, figsize=(20, 5 * len(page_features)))
        if len(page_features) == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(page_features):
            # 原始数据
            original = df_original[feature].replace([np.inf, -np.inf], np.nan).dropna()
            
            # 变换后数据
            trans_col = f'{feature}_transformed'
            if trans_col in df_transformed.columns:
                transformed = df_transformed[trans_col].replace([np.inf, -np.inf], np.nan).dropna()
            else:
                transformed = original
            
            info = transform_info.get(feature, {})
            transform_method = info.get('transform', 'unknown')
            is_override = info.get('is_override', False)
            
            orig_skew = stats.skew(original)
            trans_skew = stats.skew(transformed)
            
            # 1. 原始直方图
            axes[i, 0].hist(original, bins=50, density=True, alpha=0.7, 
                           edgecolor='black', color='steelblue')
            axes[i, 0].set_title(f'{feature}\n原始分布 (偏度={orig_skew:.2f})', fontsize=10)
            axes[i, 0].set_ylabel('密度')
            axes[i, 0].axvline(original.mean(), color='red', linestyle='--', linewidth=1, label='均值')
            
            # 2. 变换后直方图
            # 根据结果选择颜色
            if abs(trans_skew) <= 0.5:
                color = 'green'
                status = '✅优秀'
            elif abs(trans_skew) <= 1.0:
                color = 'limegreen'
                status = '✅良好'
            elif abs(trans_skew) < abs(orig_skew):
                color = 'orange'
                status = '⚠️改善'
            else:
                color = 'red'
                status = '❌'
            
            method_label = f'{transform_method}' + (' [规则]' if is_override else ' [自动]')
            axes[i, 1].hist(transformed, bins=50, density=True, alpha=0.7, 
                           edgecolor='black', color=color)
            axes[i, 1].set_title(f'{method_label}\n变换后 (偏度={trans_skew:.2f}) {status}', fontsize=10)
            axes[i, 1].axvline(transformed.mean(), color='red', linestyle='--', linewidth=1)
            
            # 3. 原始Q-Q图
            stats.probplot(original, dist="norm", plot=axes[i, 2])
            axes[i, 2].set_title('原始 Q-Q Plot', fontsize=10)
            axes[i, 2].grid(True, alpha=0.3)
            
            # 4. 变换后Q-Q图
            stats.probplot(transformed, dist="norm", plot=axes[i, 3])
            axes[i, 3].set_title('变换后 Q-Q Plot', fontsize=10)
            axes[i, 3].grid(True, alpha=0.3)
            
            # 添加网格
            for j in range(2):
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/变换对比_第{page+1}页.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # ========== 绘制偏度改善汇总图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    features_list = list(transform_info.keys())
    orig_skews = [transform_info[f]['original_skew'] for f in features_list]
    trans_skews = [transform_info[f]['transformed_skew'] for f in features_list]
    transforms = [transform_info[f]['transform'] for f in features_list]
    
    x = np.arange(len(features_list))
    width = 0.35
    
    # 偏度绝对值对比
    bars1 = axes[0].bar(x - width/2, np.abs(orig_skews), width, label='原始|偏度|', 
                        alpha=0.7, color='steelblue')
    bars2 = axes[0].bar(x + width/2, np.abs(trans_skews), width, label='变换|偏度|', 
                        alpha=0.7, color='green')
    
    # 标注改善情况
    for j, (os_val, ts_val) in enumerate(zip(orig_skews, trans_skews)):
        if abs(ts_val) > abs(os_val):
            axes[0].scatter(x[j] + width/2, abs(ts_val) + 0.1, marker='x', color='red', s=80, zorder=5)
        elif abs(ts_val) <= 0.5:
            axes[0].scatter(x[j] + width/2, abs(ts_val) + 0.1, marker='o', color='gold', s=80, zorder=5)
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(features_list, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('|偏度|')
    axes[0].set_title('偏度变化对比\n(o=优秀, ✕=未改善)')
    axes[0].legend()
    axes[0].axhline(1.0, color='orange', linestyle='--', linewidth=1.5, label='可接受阈值')
    axes[0].axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='优秀阈值')
    axes[0].grid(True, alpha=0.3)
    
    # 偏度变化方向（原始 vs 变换）
    colors = ['green' if abs(t) <= abs(o) else 'red' for o, t in zip(orig_skews, trans_skews)]
    axes[1].scatter(orig_skews, trans_skews, c=colors, s=60, alpha=0.7, edgecolors='black')
    
    # 添加对角线（不变线）
    lim = max(abs(min(orig_skews + trans_skews)), abs(max(orig_skews + trans_skews))) + 0.5
    axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='不变线')
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    # 添加特征标签
    for j, feat in enumerate(features_list):
        axes[1].annotate(feat.split('_')[0], (orig_skews[j], trans_skews[j]), 
                        fontsize=7, alpha=0.7)
    
    axes[1].set_xlabel('原始偏度')
    axes[1].set_ylabel('变换偏度')
    axes[1].set_title('偏度变化散点图\n(绿=改善, 红=变差)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 变换方法分布
    transform_counts = pd.Series(transforms).value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(transform_counts)))
    wedges, texts, autotexts = axes[2].pie(transform_counts.values, labels=transform_counts.index, 
                                            autopct='%1.0f%%', colors=colors, startangle=90)
    axes[2].set_title('变换方法分布')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/变换效果汇总.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 已生成 {n_pages} 页变换对比图 + 汇总图")


# ==================== 变换效果详细评估 ====================

def evaluate_transformations_v3(df_original, df_transformed, transform_info, output_dir='./step2_transform_v3'):
    """详细评估变换效果"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    evaluation = []
    
    print("\n" + "="*70)
    print("📊 变换效果详细评估")
    print("="*70)
    
    for feature, info in transform_info.items():
        if feature not in df_original.columns:
            continue
        
        original = df_original[feature].replace([np.inf, -np.inf], np.nan).dropna()
        
        trans_col = f'{feature}_transformed'
        if trans_col not in df_transformed.columns:
            continue
        
        transformed = df_transformed[trans_col].replace([np.inf, -np.inf], np.nan).dropna()
        
        orig_skew = stats.skew(original)
        trans_skew = stats.skew(transformed)
        orig_kurt = stats.kurtosis(original)
        trans_kurt = stats.kurtosis(transformed)
        
        # Shapiro-Wilk正态性检验
        sample_size = min(5000, len(original))
        orig_sample = original.sample(sample_size, random_state=42) if len(original) > sample_size else original
        trans_sample = transformed.sample(sample_size, random_state=42) if len(transformed) > sample_size else transformed
        
        _, orig_shapiro_p = stats.shapiro(orig_sample)
        _, trans_shapiro_p = stats.shapiro(trans_sample)
        
        # 综合评估
        skew_score = 'A' if abs(trans_skew) <= 0.5 else ('B' if abs(trans_skew) <= 1.0 else ('C' if abs(trans_skew) < abs(orig_skew) else 'D'))
        
        evaluation.append({
            '特征': feature,
            '变换': info['transform'],
            '原始偏度': round(orig_skew, 3),
            '变换偏度': round(trans_skew, 3),
            '偏度改善': round(abs(orig_skew) - abs(trans_skew), 3),
            '原始峰度': round(orig_kurt, 3),
            '变换峰度': round(trans_kurt, 3),
            '原始正态p': f"{orig_shapiro_p:.2e}",
            '变换正态p': f"{trans_shapiro_p:.2e}",
            '评级': skew_score
        })
        
        # 打印详情
        grade_emoji = {'A': '🌟', 'B': '✅', 'C': '⚠️', 'D': '❌'}[skew_score]
        print(f"\n{grade_emoji} {feature} [{info['transform']}]:")
        print(f"   偏度: {orig_skew:.3f} → {trans_skew:.3f}")
        print(f"   峰度: {orig_kurt:.3f} → {trans_kurt:.3f}")
        print(f"   正态p: {orig_shapiro_p:.2e} → {trans_shapiro_p:.2e}")
    
    eval_df = pd.DataFrame(evaluation)
    eval_df.to_csv(f'{output_dir}/变换效果评估.csv', index=False, encoding='utf-8-sig')
    
    # 统计各评级数量
    grade_counts = eval_df['评级'].value_counts()
    
    print("\n" + "="*70)
    print("📋 评级统计:")
    print(f"   🌟 A级 (|偏度|≤0.5): {grade_counts.get('A', 0)}")
    print(f"   ✅ B级 (|偏度|≤1.0): {grade_counts.get('B', 0)}")
    print(f"   ⚠️ C级 (有改善): {grade_counts.get('C', 0)}")
    print(f"   ❌ D级 (未改善): {grade_counts.get('D', 0)}")
    print("="*70)
    
    return eval_df


# ==================== 主函数 ====================

def run_step2_transform_v4(df_feature, step1_results, TRANSFORM_OVERRIDES_V2, output_dir):
    """
    步骤2主函数（V3版）
    
    Parameters:
    -----------
    df_feature : pd.DataFrame, 原始特征数据
    step1_results : dict, 步骤1的检验结果
    output_dir : str, 输出目录
    
    Returns:
    --------
    df_transformed : pd.DataFrame, 变换后的数据
    transform_info : dict, 变换信息
    eval_df : pd.DataFrame, 变换效果评估
    """
    # 1. 应用变换
    df_transformed, transform_info, summary_df = apply_transformations_v3(
        df_feature, step1_results,TRANSFORM_OVERRIDES_V2, output_dir
    )
    
    # 2. 评估变换效果
    eval_df = evaluate_transformations_v3(
        df_feature, df_transformed, transform_info, output_dir
    )
    
    return df_transformed, transform_info, eval_df


