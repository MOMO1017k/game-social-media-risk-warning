import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import norm, t as t_dist
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 配置与工具函数 ====================

VERSION_COLORS = {
    '大版本上半预热': '#FFCDD2', '大版本上半更新': '#EF5350',
    '大版本下半预热': '#FFECB3', '大版本下半更新': '#FFA726',
    '小版本上半预热': '#C8E6C9', '小版本上半更新': '#66BB6A',
    '小版本下半预热': '#BBDEFB', '小版本下半更新': '#42A5F5',
}

def add_version_background(ax, periods_df, y_min, y_max):
    """绘制版本背景色块"""
    if periods_df is None or periods_df.empty: return
    for _, row in periods_df.iterrows():
        rect = Rectangle(
            (mdates.date2num(row['start']), y_min),
            mdates.date2num(row['end']) - mdates.date2num(row['start']),
            y_max - y_min,
            facecolor=row['color'], alpha=0.15, edgecolor='none', zorder=0
        )
        ax.add_patch(rect)

def get_std_resid(garch_result):
    """从GARCH结果中提取标准化残差并对齐时间索引"""
    std_resid = garch_result['analysis']['std_resid']
    idx = garch_result['ci_df'].index
    
    if isinstance(std_resid, pd.Series):
        if not isinstance(std_resid.index, pd.DatetimeIndex):
            if len(std_resid) == len(idx):
                std_resid.index = idx
            else:
                m = min(len(std_resid), len(idx))
                std_resid = std_resid.iloc[:m]
                std_resid.index = idx[:m]
        return std_resid
    else:
        arr = np.asarray(std_resid, dtype=float)
        m = min(len(arr), len(idx))
        return pd.Series(arr[:m], index=idx[:m])

def check_stats(series, name):
    """计算核心统计检验指标"""
    clean_s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_s) < 20:
        return {'LB_p': np.nan, 'ARCH_p': np.nan, 'Norm_p': np.nan}
        
    res = {}
    try:
        lb = acorr_ljungbox(clean_s, lags=[10], return_df=True)
        res['LB_p'] = lb['lb_pvalue'].iloc[0]
        res['is_white'] = res['LB_p'] > 0.05
    except:
        res['LB_p'] = 0.0
        res['is_white'] = False
        
    try:
        lm = het_arch(clean_s, ddof=4)
        res['ARCH_p'] = lm[1]
        res['no_arch'] = res['ARCH_p'] > 0.05
    except:
        res['ARCH_p'] = 0.0
        res['no_arch'] = False
        
    res['mean'] = clean_s.mean()
    res['std'] = clean_s.std()
    res['skew'] = clean_s.skew()
    res['kurt'] = clean_s.kurtosis()
    
    return res

# ==================== 2. 核心：异常识别与温莎化清洗 ====================

def identify_and_winsorize(series, ci_df, garch_result, q_low=0.005, q_high=0.995):
    """识别异常并温莎化清洗"""
    params = garch_result['params']
    nu = params.get('nu', 100)
    
    limit_upper = t_dist.ppf(q_high, df=nu)
    limit_lower = t_dist.ppf(q_low, df=nu)
    
    mask_pos = series > limit_upper
    mask_neg = series < limit_lower
    anomaly_mask = mask_pos | mask_neg
    
    cleaned_series = series.copy()
    valid_data = series[~anomaly_mask]
    
    if len(valid_data) > 0:
        win_upper = valid_data.quantile(q_high)
        win_lower = valid_data.quantile(q_low)
    else:
        win_upper = limit_upper
        win_lower = limit_lower
        
    cleaned_series[mask_pos] = win_upper
    cleaned_series[mask_neg] = win_lower
    
    return cleaned_series, anomaly_mask, (limit_lower, limit_upper)

# ==================== 3. 主流程：处理、对比与绘图 ====================

def process_cleaning_comparison(df_input, garch_results, feature_list, version_dict=None, output_dir='./step7_clean_verify'):
    os.makedirs(output_dir, exist_ok=True)
    print("="*80)
    print("📌 步骤7：GARCH残差清洗、验证与相关性分析")
    print("   目标：温莎化处理异常值，对比清洗前后统计性质，生成Copula输入")
    print("="*80)
    
    periods_df = None
    if version_dict:
        periods = [{'name': k, 'start': pd.to_datetime(v[0]), 'end': pd.to_datetime(v[1]), 
                    'color': VERSION_COLORS.get(k, '#E0E0E0')} for k, v in version_dict.items()]
        periods_df = pd.DataFrame(periods).sort_values('start')
        
    comparison_stats = []
    cleaned_data_dict = {}
    
    for feature in feature_list:
        if feature not in garch_results: continue
        print(f"\n🔍 处理特征: {feature}")
        
        result = garch_results[feature]
        ci_df = result['ci_df']
        
        std_resid_raw = get_std_resid(result)
        std_resid_clean, mask, (lim_l, lim_h) = identify_and_winsorize(std_resid_raw, ci_df, result)
        
        cleaned_data_dict[feature] = std_resid_clean
        result['analysis']['std_resid_clean'] = std_resid_clean
        
        stat_raw = check_stats(std_resid_raw, 'Raw')
        stat_clean = check_stats(std_resid_clean, 'Clean')
        
        comparison_stats.append({
            '特征': feature,
            '异常点数': mask.sum(),
            '异常比例': f"{mask.sum()/len(std_resid_raw):.2%}",
            'Raw_LB_p': stat_raw['LB_p'],
            'Raw_ARCH_p': stat_raw['ARCH_p'],
            'Raw_Kurt': stat_raw['kurt'],
            'Clean_LB_p': stat_clean['LB_p'],
            'Clean_ARCH_p': stat_clean['ARCH_p'],
            'Clean_Kurt': stat_clean['kurt'],
            '白噪声改善': '✅' if stat_clean['LB_p'] > stat_raw['LB_p'] else '➖',
            '分布改善(峰度)': '✅' if abs(stat_clean['kurt']) < abs(stat_raw['kurt']) else '❌'
        })
        
        # --- 绘图 (保持原样) ---
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2)
        
        ax1 = fig.add_subplot(gs[0, :])
        y_min, y_max = std_resid_raw.min()*1.1, std_resid_raw.max()*1.1
        add_version_background(ax1, periods_df, y_min, y_max)
        ax1.plot(std_resid_raw.index, std_resid_raw, c='gray', alpha=0.5, lw=1, label='清洗前 (Raw)')
        ax1.plot(std_resid_clean.index, std_resid_clean, c='green', alpha=0.8, lw=1, ls='--', label='清洗后 (Winsorized)')
        if mask.any():
            anom_points = std_resid_raw[mask]
            ax1.scatter(anom_points.index, anom_points, c='red', s=20, marker='x', label='被清洗的异常点', zorder=5)
        ax1.set_title(f'{feature} - 残差清洗效果 (Winsorization)', fontsize=12, fontweight='bold')
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 0])
        sns.kdeplot(std_resid_raw, ax=ax2, color='gray', fill=True, alpha=0.3, label='清洗前分布')
        sns.kdeplot(std_resid_clean, ax=ax2, color='green', fill=False, lw=2, label='清洗后分布')
        x = np.linspace(-4, 4, 100)
        ax2.plot(x, norm.pdf(x), 'r:', label='标准正态 N(0,1)')
        ax2.set_title(f'分布形态修正 (峰度: {stat_raw["kurt"]:.2f} -> {stat_clean["kurt"]:.2f})')
        ax2.legend()

        ax3 = fig.add_subplot(gs[1, 1])
        plot_acf(std_resid_clean, ax=ax3, lags=20, alpha=0.05, title='清洗后残差 ACF')

        ax4 = fig.add_subplot(gs[2, :])
        vol_proxy_raw = (std_resid_raw**2).rolling(24).mean()
        vol_proxy_clean = (std_resid_clean**2).rolling(24).mean()
        ax4.plot(vol_proxy_raw.index, vol_proxy_raw, c='gray', alpha=0.5, label='清洗前波动')
        ax4.plot(vol_proxy_clean.index, vol_proxy_clean, c='green', lw=1.5, label='清洗后波动')
        ax4.set_title(f'波动率聚集残留检查')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{feature}_resid_clean_verify.png', dpi=100)
        plt.close()

    df_comp = pd.DataFrame(comparison_stats)
    df_comp.to_csv(f'{output_dir}/残差清洗前后对比报告.csv', index=False, encoding='utf-8-sig')
    print("\n📋 清洗效果对比摘要:")
    print(df_comp[['特征', '异常比例', 'Raw_Kurt', 'Clean_Kurt', '白噪声改善']].to_string(index=False))
    
    return cleaned_data_dict

# ==================== 4. 步骤7后续：清洗后的相关性分析 ====================

def analyze_cleaned_correlation(cleaned_data_dict, output_dir):
    print("\n" + "="*80)
    print("📌 步骤7.5：基于清洗后数据的相关性分析")
    print("="*80)
    
    if not cleaned_data_dict: return None
    
    df_clean = pd.DataFrame(cleaned_data_dict).dropna()
    
    corr_pearson = df_clean.corr(method='pearson')
    corr_kendall = df_clean.corr(method='kendall')
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    
    sns.heatmap(corr_pearson, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0, 
                square=True, ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Pearson 相关矩阵 (线性)')
    
    sns.heatmap(corr_kendall, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0, 
                square=True, ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Kendall Tau 相关矩阵 (Copula输入)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/清洗后残差相关性矩阵.png', dpi=150)
    plt.close()
    
    corr_kendall.to_csv(f'{output_dir}/corr_matrix_kendall.csv', encoding='utf-8-sig')
    
    return corr_kendall

# ==================== [新增] 5. 边缘分布拟合 (Copula准备) ====================

def fit_marginal_distributions(cleaned_data_dict, output_dir):
    """
    对清洗后的残差进行分布拟合，确定Copula所需的边缘分布参数
    主要对比 Normal 和 Student-t
    """
    print("\n" + "="*80)
    print("📌 步骤7.8：清洗后残差的边缘分布拟合 (Copula参数准备)")
    print("="*80)
    
    fit_results = []
    
    for feature, data in cleaned_data_dict.items():
        # 移除空值
        data = data.dropna()
        if len(data) < 50: continue
        
        # 1. 拟合正态分布
        mu, std = norm.fit(data)
        ll_n = np.sum(norm.logpdf(data, mu, std))
        aic_n = 2*2 - 2*ll_n
        
        # 2. 拟合t分布
        # t分布通常能更好地捕捉残差的厚尾特性
        try:
            params_t = t_dist.fit(data)
            df_t, loc_t, scale_t = params_t
            ll_t = np.sum(t_dist.logpdf(data, df_t, loc_t, scale_t))
            aic_t = 2*3 - 2*ll_t
        except:
            aic_t = np.inf
            params_t = (np.nan, np.nan, np.nan)
            
        # 3. 判定最优分布
        if aic_t < aic_n:
            best_dist = 'Student-t'
            best_params = f"df={params_t[0]:.2f}, loc={params_t[1]:.4f}, scale={params_t[2]:.4f}"
        else:
            best_dist = 'Normal'
            best_params = f"loc={mu:.4f}, scale={std:.4f}"
            
        fit_results.append({
            '特征': feature,
            'AIC_Normal': aic_n,
            'AIC_t': aic_t,
            '最优分布': best_dist,
            '分布参数': best_params
        })
        
    df_fit = pd.DataFrame(fit_results)
    df_fit.to_csv(f'{output_dir}/边缘分布拟合参数.csv', index=False, encoding='utf-8-sig')
    
    print("✅ 边缘分布参数已保存。")
    print(df_fit[['特征', '最优分布', '分布参数']].head().to_string(index=False))
    return df_fit

# ==================== 入口函数 ====================

def run_step7_clean_verify(df_with_ci, garch_results, feature_list, version_dict=None, output_dir='./step7_clean_verify'):
    
    # 1. 清洗与验证
    cleaned_data = process_cleaning_comparison(
        df_input=df_with_ci,
        garch_results=garch_results,
        feature_list=feature_list,
        version_dict=version_dict,
        output_dir=output_dir
    )
    
    # 2. 相关性分析
    corr_matrix = analyze_cleaned_correlation(cleaned_data, output_dir)
    
    # --- [关键补充] ---
    
    # 3. 保存清洗后的数据 (Copula建模必须输入)
    if cleaned_data:
        df_cleaned_output = pd.DataFrame(cleaned_data)
        save_path = f'{output_dir}/step7_cleaned_residuals.csv'
        df_cleaned_output.to_csv(save_path, encoding='utf-8-sig')
        print(f"\n💾 清洗后的残差数据已保存: {save_path}")
        
    # 4. 边缘分布拟合 (Copula PIT变换必须参数)
    dist_params = fit_marginal_distributions(cleaned_data, output_dir)
    
    return cleaned_data, corr_matrix, dist_params