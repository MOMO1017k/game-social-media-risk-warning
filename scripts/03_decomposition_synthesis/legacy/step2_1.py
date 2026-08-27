import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist, norm
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 核心工具函数 ====================

def fix_correlation_matrix(corr_matrix, min_eigval=1e-8):
    """
    修复非正定相关矩阵（确保可以进行Cholesky分解）
    使用最近正定矩阵投影方法
    """
    # 确保对称性
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    
    # 确保对角线为1
    np.fill_diagonal(corr_matrix, 1.0)
    
    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    
    # 如果已经正定，直接返回
    if np.min(eigvals) >= min_eigval:
        return corr_matrix, False
    
    # 将负特征值和过小的特征值修复为正值
    eigvals_fixed = np.maximum(eigvals, min_eigval)
    
    # 重构矩阵
    corr_fixed = eigvecs @ np.diag(eigvals_fixed) @ eigvecs.T
    
    # 重新归一化对角线为1
    d = np.sqrt(np.diag(corr_fixed))
    corr_fixed = corr_fixed / np.outer(d, d)
    
    # 确保对角线精确为1
    np.fill_diagonal(corr_fixed, 1.0)
    
    return corr_fixed, True


def empirical_quantile_transform(u_samples, historical_resid):
    """
    经验分位数映射：将U[0,1]样本转换为经验分布样本
    修正版：使用 np.quantile 进行更平滑的插值，处理边界问题
    """
    historical_resid = np.asarray(historical_resid)
    valid_mask = ~np.isnan(historical_resid) & ~np.isinf(historical_resid)
    clean_resid = historical_resid[valid_mask]
    
    if len(clean_resid) < 10:
        # 样本太少，回退到标准正态分布映射
        # 保持均值和方差的大致刻度
        mu, std = np.mean(clean_resid), np.std(clean_resid)
        return norm.ppf(u_samples) * std + mu
        
    # 使用 numpy 的 quantile 函数进行逆变换
    # method='linear' 是默认的 Type 7 插值，适合连续分布模拟
    # u_samples 必须在 [0, 1] 之间
    
    # 为了防止 u=0 或 u=1 导致无穷大或越界（虽然 np.quantile 能处理，但最好钳位）
    u_clamped = np.clip(u_samples, 1e-9, 1 - 1e-9)
    
    transformed = np.quantile(clean_resid, u_clamped, method='linear')
    
    return transformed


def generate_synthetic_step1_integrated(
    noise_analysis_dict,      # 各特征的噪音分析结果
    corr_matrix,              # Pearson相关矩阵
    std_resid_dict,           # 各特征的历史标准化残差池
    feature_list,             # 特征列表
    start_date,               # 合成数据起始日期
    months=11,                # 合成月数
    freq='15min',             # 时间频率
    seed=42,                  # 随机种子
    clip_mode='smooth',       # 截断模式: 'smooth' (默认) 或 'conservative'
    output_dir='./result/step1_synthetic'
):
    """
    整合版第一步：生成相关白噪声
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print("═" * 70)
    print(f"📌 第一步：生成 {months} 个月的相关白噪声 (Mode: {clip_mode})")
    print("═" * 70)
    
    # ══════════════════════════════════════════════════════════════════════
    # 1. 构建时间轴
    # ══════════════════════════════════════════════════════════════════════
    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + pd.DateOffset(months=months)
    
    time_index = pd.date_range(
        start=start_ts, 
        end=end_ts, 
        freq=freq, 
        inclusive='left'
    )
    
    n_samples = len(time_index)
    n_features = len(feature_list)
    
    print(f"   📅 时间范围: {start_ts} 至 {end_ts}")
    print(f"   📊 样本总数: {n_samples:,} | 特征数: {n_features}")
    
    # ══════════════════════════════════════════════════════════════════════
    # 2. 处理相关矩阵 (默认输入为 Pearson)
    # ══════════════════════════════════════════════════════════════════════
    # 对齐特征
    rho_target = corr_matrix.loc[feature_list, feature_list].values.astype(float).copy()
    
    # 修复非正定性 (Copula 需要 Σ 为正定)
    rho_fixed, was_fixed = fix_correlation_matrix(rho_target)
    
    if was_fixed:
        print(f"   ⚠️ 输入 Pearson 矩阵非正定，已自动修复")
    
    # ══════════════════════════════════════════════════════════════════════
    # 3. Cholesky 分解生成相关正态噪声
    # ══════════════════════════════════════════════════════════════════════
    # 尝试 Cholesky 分解，如果失败不再捕获异常，直接报错
    L = cholesky(rho_fixed, lower=True)
    
    # 生成独立标准正态随机数 Z ~ N(0, I)
    z_independent = np.random.standard_normal((n_samples, n_features))
    
    # 引入相关性: X = Z @ L.T
    z_correlated = z_independent @ L.T
    
    # ══════════════════════════════════════════════════════════════════════
    # 4. 转换为均匀分布 U[0,1]
    # ══════════════════════════════════════════════════════════════════════
    u_correlated = norm.cdf(z_correlated)
    
    # 避免边界值 (防止 ppf 返回 ±∞)
    epsilon = 1e-10
    u_correlated = np.clip(u_correlated, epsilon, 1 - epsilon)
    
    # ══════════════════════════════════════════════════════════════════════
    # 5. 边际分布逆变换 & 截断
    # ══════════════════════════════════════════════════════════════════════
    synthetic_noise = np.zeros((n_samples, n_features))
    generation_info = {}
    
    print(f"\n   {'特征':<25} {'分布':<15} {'截断策略'}")
    print("   " + "─" * 60)
    
    for i, feature in enumerate(feature_list):
        noise_analysis = noise_analysis_dict.get(feature, {})
        std_resid_pool = std_resid_dict.get(feature, np.array([]))
        
        # 清洗残差池
        std_resid_pool = np.asarray(std_resid_pool)
        valid_mask = ~np.isnan(std_resid_pool) & ~np.isinf(std_resid_pool)
        std_resid_pool = std_resid_pool[valid_mask]
        
        best_dist = noise_analysis.get('best_distribution', 'Non-parametric')
        col_data = None
        dist_info = {}

        # --- A. 分布逆变换 (代码A逻辑) ---
        if 'N(0,1)' in best_dist:
            col_data = norm.ppf(u_correlated[:, i])
            dist_info = {'type': 'Normal', 'params': 'N(0,1)'}
            
        elif 't(' in best_dist or noise_analysis.get('is_t_dist', False):
            df = max(float(noise_analysis.get('t_df', 5.0)), 5.0)
            loc = noise_analysis.get('t_loc', 0.0)
            scale = max(float(noise_analysis.get('t_scale', 1.0)), 1e-6)
            
            col_data = t_dist.ppf(u_correlated[:, i], df, loc, scale)
            dist_info = {'type': 't-dist', 'params': f'df={df:.1f}'}
            
        else:
            # 经验分布 (代码A逻辑)
            if len(std_resid_pool) >= 100:
                col_data = empirical_quantile_transform(u_correlated[:, i], std_resid_pool)
                dist_info = {'type': 'Empirical', 'params': f'n={len(std_resid_pool)}'}
            else:
                # 降级为 t 分布
                df = max(float(noise_analysis.get('t_df', 5.0)), 5.0)
                col_data = t_dist.ppf(u_correlated[:, i], df, 0, 1)
                dist_info = {'type': 't-dist(fallback)', 'params': f'df={df:.1f}'}

        # --- B. 数据截断 (整合逻辑) ---
        # 如果残差池太小，无法计算分位数，强制使用固定阈值
        if len(std_resid_pool) < 10:
            clip_min, clip_max = -6.0, 6.0
            clip_desc = "Fixed[-6,6]"
        else:
            if clip_mode == 'smooth':
                # 代码B逻辑: 平滑策略 (0.05% ~ 99.95% + 10% Margin)
                # 允许比历史极值稍微大一点，保留尾部活力
                lower = np.percentile(std_resid_pool, 0.05)
                upper = np.percentile(std_resid_pool, 99.95)
                margin = (upper - lower) * 0.1
                clip_min = lower - margin
                clip_max = upper + margin
                clip_desc = "Smooth(w/Margin)"
                
            elif clip_mode == 'conservative':
                # 代码A逻辑: 保守策略 (0.1% ~ 99.9%)
                # 严格限制在历史范围内
                clip_min = np.quantile(std_resid_pool, 0.001)
                clip_max = np.quantile(std_resid_pool, 0.999)
                clip_desc = "Conservative(Strict)"
            else:
                # 默认 fallback
                clip_min, clip_max = -10.0, 10.0
                clip_desc = "Loose"

        # 执行截断
        col_data = np.clip(col_data, clip_min, clip_max)
        synthetic_noise[:, i] = col_data
        
        # 记录信息
        generation_info[feature] = {
            'distribution': dist_info,
            'clipping': {'mode': clip_mode, 'min': float(clip_min), 'max': float(clip_max)}
        }
        
        if i < 5:  # 只打印前5个
            print(f"   {feature:<25} {dist_info['type']:<15} {clip_desc}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. 构建结果 & 验证
    # ══════════════════════════════════════════════════════════════════════
    df_synthetic_noise = pd.DataFrame(
        synthetic_noise,
        index=time_index,
        columns=feature_list
    )
    df_synthetic_noise.index.name = 'timestamp'
    
    # 验证相关性 (代码A逻辑，适配 Pearson 输入)
    print("\n   📈 相关性保持度验证:")
    
    # 计算合成数据的 Pearson 相关系数
    synth_corr_pearson = df_synthetic_noise.corr(method='pearson').values
    
    # 计算偏差
    corr_diff = np.abs(synth_corr_pearson - rho_fixed)
    np.fill_diagonal(corr_diff, 0)
    
    print(f"     • Pearson MAE: {corr_diff.mean():.4f}")
    print(f"     • Pearson Max Error: {corr_diff.max():.4f}")
    
    # 简单的白噪音检验 (Ljung-Box) - 无 try/except
    print("\n   📈 白噪音检验 (前3个特征):")
    for feature in feature_list[:3]:
        lb_res = acorr_ljungbox(df_synthetic_noise[feature], lags=[10], return_df=True)
        p_val = lb_res['lb_pvalue'].iloc[0]
        status = "✅" if p_val > 0.05 else "⚠️"
        print(f"     • {feature}: p={p_val:.4f} {status}")

    # ══════════════════════════════════════════════════════════════════════
    # 7. 可视化 & 保存
    # ══════════════════════════════════════════════════════════════════════
    print("\n【生成可视化报告】")
    
    # 7.1 热力图对比
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(1, 3, 1)
    sns.heatmap(rho_fixed, ax=ax1, cmap='RdBu_r', center=0, cbar=False,
                xticklabels=feature_list, yticklabels=feature_list)
    ax1.set_title('Target Correlation (Pearson)')
    
    ax2 = fig.add_subplot(1, 3, 2)
    sns.heatmap(synth_corr_pearson, ax=ax2, cmap='RdBu_r', center=0, cbar=False,
                xticklabels=feature_list, yticklabels=feature_list)
    ax2.set_title('Synthetic Correlation (Pearson)')
    
    ax3 = fig.add_subplot(1, 3, 3)
    sns.heatmap(corr_diff, ax=ax3, cmap='Reds', vmin=0, vmax=0.1,
                xticklabels=feature_list, yticklabels=feature_list)
    ax3.set_title(f'Difference (MAE={corr_diff.mean():.4f})')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_check.png')
    plt.close()
    
    # 7.2 保存数据
    df_synthetic_noise.to_csv(f'{output_dir}/synthetic_noise.csv', encoding='utf-8-sig')
    
    # Parquet 需要 pyarrow 引擎，假设环境已有
    df_synthetic_noise.to_parquet(f'{output_dir}/synthetic_noise.parquet')

    # 7.3 保存报告
    report = {
        'params': {'months': months, 'clip_mode': clip_mode},
        'quality': {'pearson_mae': float(corr_diff.mean())},
        'details': generation_info
    }
    with open(f'{output_dir}/generation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"   ✅ 完成。数据已保存至: {output_dir}")
    
    return df_synthetic_noise, report