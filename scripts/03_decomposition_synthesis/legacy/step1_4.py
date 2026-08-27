

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# ==================== 组合分析配置 ====================

# 为15种组合定义颜色
COMBO_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# 事件类型标记样式
CATEGORY_MARKERS = {
    'promotion_other': 'o',
    'promotion_normal': 'o',      # 圆形 - 推广
    'promotion_main': 'o',      # 圆形 - 推广
                # 圆形 - 推广
    'crisis': 'X',         # X形 - 危机
    'guide': 's',          # 方形 - 引导
    'maintenance': 'D',    # 菱形 - 维护
    'daily': '^',          # 三角 - 日常
}

# 事件类型简称
CATEGORY_LABELS = {
    'promotion_other': '推广-其他',
    'promotion_normal': '推广-影响力一般',
    'promotion_main': '推广-影响力较大',

    'crisis': '道歉公告',
    'guide': '攻略',
    'maintenance': '维护公告',
    'daily': '日常互动',
}


# ==================== 衰减函数 ====================

def exponential_decay(t, a, tau):
    """指数衰减: a * exp(-t/tau)"""
    return a * np.exp(-t / (tau + 1e-6))

def power_decay(t, a, alpha):
    """幂律衰减: a * (t+1)^(-alpha)"""
    return a * np.power(t + 1, -alpha)

def fit_decay(time_points, response_values, decay_type='exponential'):
    """拟合衰减曲线"""
    if len(time_points) < 3 or len(response_values) < 3:
        return None, 0
    
    try:
        if decay_type == 'exponential':
            a0 = np.abs(response_values[0]) if len(response_values) > 0 else 1
            tau0 = len(time_points) / 3
            popt, _ = curve_fit(exponential_decay, time_points, response_values, 
                               p0=[a0, tau0], maxfev=5000, 
                               bounds=([0, 0.1], [np.inf, 100]))
            fitted = exponential_decay(time_points, *popt)
        else:  # power
            a0 = np.abs(response_values[0]) if len(response_values) > 0 else 1
            popt, _ = curve_fit(power_decay, time_points, response_values,
                               p0=[a0, 0.5], maxfev=5000,
                               bounds=([0, 0.01], [np.inf, 3]))
            fitted = power_decay(time_points, *popt)
        
        ss_res = np.sum((response_values - fitted) ** 2)
        ss_tot = np.sum((response_values - np.mean(response_values)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return popt, max(0, r_squared)
    except:
        return None, 0


# ==================== 组合分析核心函数 ====================

def prepare_combo_event_data(df_off):
    """
    准备官方事件数据，创建组合变量
    
    Parameters:
    -----------
    df_off : pd.DataFrame, 官方事件数据
             需要包含列: timestamp, category, official_author
    
    Returns:
    --------
    df_events : 处理后的事件数据（含组合变量）
    combo_stats : 组合统计信息
    """
    df_events = df_off.copy()
    
    # 确保timestamp是datetime类型
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
    
    # 创建组合变量: author_category
    df_events['combo'] = df_events['official_author'] + '_' + df_events['category']
    
    # 按时间排序
    df_events = df_events.sort_values('timestamp').reset_index(drop=True)
    
    # 统计各组合
    combo_stats = df_events['combo'].value_counts().reset_index()
    combo_stats.columns = ['组合', '事件数']
    
    print("="*60)
    print("📋 事件组合统计:")
    print("="*60)
    print(f"   总事件数: {len(df_events)}")
    print(f"   组合类型数: {df_events['combo'].nunique()}")
    print("\n   各组合事件数:")
    for _, row in combo_stats.iterrows():
        print(f"      • {row['组合']}: {row['事件数']}次")
    
    # 创建组合颜色映射
    combos = df_events['combo'].unique()
    print(combos)
    combo_color_map = {combo: COMBO_COLORS[i % 20] for i, combo in enumerate(combos)}
    
    return df_events, combo_stats, combo_color_map



# ==================== 动态衰减回归分析 ====================
def determine_best_decay_type(df_feature, df_events, feature_name, window_after=32):
    """
    既确定最优衰减类型，也计算冲击统计量 (兼容旧接口)
    """
    col_name = f'{feature_name}_resid' if f'{feature_name}_resid' in df_feature.columns else feature_name
    series = df_feature[col_name]
    
    # 初始化完整的返回结构
    result_structure = {
        'by_combo': {},
        'column_used': col_name
    }
    
    combo_configs = {} # 用于回归
    
    for combo in df_events['combo'].unique():
        combo_events = df_events[df_events['combo'] == combo]
        impacts = []
        
        # 1. 提取响应曲线
        for _, event in combo_events.iterrows():
            start = event['timestamp']
            end = start + pd.Timedelta(minutes=15 * window_after)
            segment = series[(series.index >= start) & (series.index < end)].values
            if len(segment) > 0:
                if len(segment) < window_after:
                    segment = np.pad(segment, (0, window_after-len(segment)), 'constant')
                impacts.append(segment[:window_after])
        
        if not impacts:
            continue

        # 计算统计量
        mean_impact = np.nanmean(impacts, axis=0)
        # 寻找峰值（绝对值最大点）
        peak_idx = np.argmax(np.abs(mean_impact))
        overall_peak = mean_impact[peak_idx]
        peak_time_hours = peak_idx * 15 / 60
        
        # 准备拟合数据 (去除基线)
        mean_impact_fit = np.abs(mean_impact - mean_impact[0])
        t = np.arange(len(mean_impact_fit))
        
        # 2. 拟合比较
        popt_exp, r2_exp = fit_decay(t, mean_impact_fit, 'exponential')
        popt_pow, r2_pow = fit_decay(t, mean_impact_fit, 'power')
        
        # 3. 择优
        decay_info = {}
        if r2_pow > r2_exp + 0.05:
            config = {'type': 'power', 'param': popt_pow[1]}
            decay_info = {
                'type': 'power', 'params': popt_pow, 'r_squared': r2_pow,
                'half_life_min': np.nan # 幂律衰减半衰期定义复杂，暂略
            }
        else:
            tau = popt_exp[1] if popt_exp is not None else 8.0
            config = {'type': 'exponential', 'param': tau}
            decay_info = {
                'type': 'exponential', 'params': popt_exp, 'r_squared': r2_exp,
                'half_life_min': tau * np.log(2) * 15
            }

        combo_configs[combo] = config
        
        # 4. 填充 result_structure (兼容绘图函数)
        result_structure['by_combo'][combo] = {
            'mean_impact': mean_impact,
            'overall_peak': overall_peak,
            'peak_time': peak_idx,
            'peak_time_hours': peak_time_hours,
            'n_events': len(impacts),
            'decay': decay_info,
            'config': config # 额外保存config供回归使用
        }
            
    return result_structure

def create_dynamic_event_indicators(df_feature, df_events, combo_configs, window_after=32):
    """
    根据每个组合的最优配置，生成对应的衰减指示变量
    """
    df = df_feature.copy()
    event_columns = []
    
    for combo, config in combo_configs.items():
        combo_events = df_events[df_events['combo'] == combo]
        
        safe_combo = combo.replace(' ', '_').replace('-', '_')
        decay_col = f'event_{safe_combo}_{config["type"]}' # 列名带上类型
        
        df[decay_col] = 0.0
        decay_param = config['param']
        
        # 预计算衰减模板 (加速)
        t_template = np.arange(window_after + 1)
        if config['type'] == 'exponential':
            # exp(-t/tau)
            decay_template = np.exp(-t_template / (decay_param + 1e-6))
        else:
            # (t+1)^(-alpha)
            decay_template = np.power(t_template + 1, -decay_param)
            
        for _, event in combo_events.iterrows():
            event_time = event['timestamp']
            
            # 找到索引位置
            try:
                start_idx = df.index.get_indexer([event_time], method='nearest')[0]
            except:
                continue
                
            if start_idx < 0 or start_idx >= len(df): continue
            
            end_idx = min(start_idx + window_after, len(df))
            length = end_idx - start_idx
            
            # 叠加模板
            df.iloc[start_idx:end_idx, df.columns.get_loc(decay_col)] += decay_template[:length]
            
        event_columns.append(decay_col)
        
    return df, event_columns


def combo_regression_analysis_dynamic(df_feature, feature_name, df_events, use_residual=True):
    """
    带组合事件变量的回归分析:动态衰减回归分析
    """
    # 确定Y变量
    if use_residual and f'{feature_name}_resid' in df_feature.columns:
        y_col = f'{feature_name}_resid'
    elif f'{feature_name}_transformed' in df_feature.columns:
        y_col = f'{feature_name}_transformed'
    else:
        y_col = feature_name
    
    if y_col not in df_feature.columns:
        return None
    
    print(f"\n   📊 回归分析 (Y: {y_col})")
    
    # 1. 确定最优衰减配置
    impact_analysis = determine_best_decay_type(df_feature, df_events, feature_name)
    # 提取 configs
    combo_configs = {}
    if 'by_combo' in impact_analysis:
        for combo, data in impact_analysis['by_combo'].items():
            combo_configs[combo] = data['config']
    
    if not combo_configs:
        return None
    
    # 2. 创建动态变量
    df_reg, event_cols = create_dynamic_event_indicators(df_feature, df_events, combo_configs)
    
    # 3. 回归 (OLS)
    y = df_reg[y_col].values
    X = df_reg[event_cols].values
    
    # 清洗
    valid_mask = ~(np.isnan(y) | np.isinf(y) | np.any(np.isnan(X) | np.isinf(X), axis=1))
    y_clean = y[valid_mask]
    X_clean = X[valid_mask]
    
    if len(y_clean) < 100 or X_clean.shape[1] == 0:
        print(f"   ⚠️ 数据不足或无有效事件变量")
        return None
    
    # 标准化X
    # X_mean = X_clean.mean(axis=0)
    # X_std = X_clean.std(axis=0) + 1e-10
    # X_scaled = (X_clean - X_mean) / X_std
    
    # 添加常数项
    # X_with_const = sm.add_constant(X_scaled)
    X_with_const = sm.add_constant(X_clean)

    
    
    model = OLS(y_clean, X_with_const).fit()
    

    results = {
            'feature': feature_name,
            'r_squared': model.rsquared,
            'intercept': model.params[0], # 获取截距
            'coefficients': {},
            'configs': combo_configs,
            'significant_combos': []
        }

    # 计算残差并保存
    resid_clean = y_clean - model.fittedvalues
    resid_series = pd.Series(np.nan, index=df_feature.index)
    valid_index = df_feature.index[valid_mask]
    resid_series.loc[valid_index] = resid_clean
    resid_col = f'{feature_name}_event_adj_resid'
    df_feature[resid_col] = resid_series
    results['event_adj_resid_col'] = resid_col # 现在可以赋值了

    for i, col in enumerate(['const'] + event_cols):
        # 确保索引不过界 (statsmodels 的 params 顺序与 exog 对应)
        # 我们的 X_with_const 列顺序是 ['const'] + event_cols
        if col == 'const': 
            continue
        
        # 确保索引不越界 (防御性编程)
        if i >= len(model.params):
            break

        # 使用位置索引获取统计量
        coef = model.params[i]
        p_val = model.pvalues[i]
        t_score = model.tvalues[i]
        
        temp = col.replace('event_', '')
        decay_type = temp.split('_')[-1]
        combo_name = temp.replace(f'_{decay_type}', '')
        
        if p_val < 0.05:
            results['significant_combos'].append(combo_name)

        results['coefficients'][combo_name] = {
            'coef': coef,
            'pvalue': p_val,
            't_score': t_score,
            'decay_type': decay_type,
            'decay_param': combo_configs[combo_name]['param']
        }
        
    return results
        


# ==================== 可视化 ====================

def plot_combo_impact(df_feature, feature_name, df_events, 
                     impact_results, combo_color_map, output_dir):
    """
    绑定绘制组合事件冲击分析图
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定使用的列
    col_name = f'{feature_name}_transformed'
    if col_name not in df_feature.columns:
        col_name = feature_name
    if col_name not in df_feature.columns:
        return
    
    series = df_feature[col_name]
    
    fig = plt.figure(figsize=(24, 20))
    
    # ========== 图1: 时序图 + 所有事件标记 ==========
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(series.index, series.values, linewidth=0.5, alpha=0.7, color='steelblue')
    
    # 标记事件
    for _, event in df_events.iterrows():
        event_time = event['timestamp']
        combo = event['combo']
        color = combo_color_map.get(combo, 'gray')
        cat = event['category']
        marker = CATEGORY_MARKERS.get(cat, 'o')
        
        if series.index.min() <= event_time <= series.index.max():
            ax1.axvline(event_time, color=color, alpha=0.4, linewidth=0.8)
    
    ax1.set_title(f'{feature_name} - 时序图与事件标记', fontsize=12, fontweight='bold')
    ax1.set_ylabel('值')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(True, alpha=0.3)
    
    # ========== 图2: 所有组合的冲击响应曲线 ==========
    ax2 = fig.add_subplot(4, 2, 2)
    
    if 'by_combo' in impact_results and impact_results['by_combo']:
        for combo, data in impact_results['by_combo'].items():
            if len(data['mean_impact']) > 0:
                time_axis = np.arange(len(data['mean_impact'])) * 15 / 60  # 小时
                color = combo_color_map.get(combo, 'gray')
                
                ax2.plot(time_axis, data['mean_impact'], linewidth=2, 
                        label=f"{combo} (n={data['n_events']})", color=color, alpha=0.8)
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1, label='事件发生')
        ax2.set_xlabel('事件后时间 (小时)')
        ax2.set_ylabel('相对变化 (%)')
        ax2.set_title('各组合的平均冲击响应', fontsize=12)
        ax2.legend(fontsize=7, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3)
    
    # ========== 图3: 冲击强度热力图 (author × category) ==========
    ax3 = fig.add_subplot(4, 2, 3)
    
    # 构建热力图数据
    authors = df_events['official_author'].unique()
    categories = df_events['category'].unique()
    
    heatmap_data = np.zeros((len(authors), len(categories)))
    heatmap_data[:] = np.nan
    
    for i, author in enumerate(authors):
        for j, cat in enumerate(categories):
            combo = f"{author}_{cat}"
            if combo in impact_results['by_combo']:
                heatmap_data[i, j] = impact_results['by_combo'][combo]['overall_peak']
    
    # 绘制热力图
    im = ax3.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', 
                    vmin=-np.nanmax(np.abs(heatmap_data)), 
                    vmax=np.nanmax(np.abs(heatmap_data)))
    
    ax3.set_xticks(np.arange(len(categories)))
    ax3.set_yticks(np.arange(len(authors)))
    ax3.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories], fontsize=9)
    ax3.set_yticklabels(authors, fontsize=9)
    
    # 添加数值标注
    for i in range(len(authors)):
        for j in range(len(categories)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                               ha='center', va='center', fontsize=8,
                               color='white' if abs(heatmap_data[i, j]) > np.nanmax(np.abs(heatmap_data))/2 else 'black')
    
    ax3.set_title('冲击强度热力图 (官方来源 × 事件类型)', fontsize=12)
    plt.colorbar(im, ax=ax3, label='峰值冲击 (%)')
    
    # ========== 图4: 按事件类型聚合的冲击 ==========
    ax4 = fig.add_subplot(4, 2, 4)
    
    if 'by_category' in impact_results and impact_results['by_category']:
        categories_sorted = sorted(impact_results['by_category'].items(), 
                                   key=lambda x: -abs(x[1]['mean_peak']))
        
        cats = [c[0] for c in categories_sorted]
        peaks = [c[1]['mean_peak'] for c in categories_sorted]
        n_events = [c[1]['n_events'] for c in categories_sorted]
        
        colors = ['green' if p > 0 else 'red' for p in peaks]
        bars = ax4.bar(range(len(cats)), peaks, color=colors, alpha=0.7, edgecolor='black')
        
        ax4.set_xticks(range(len(cats)))
        ax4.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in cats], rotation=45, ha='right')
        ax4.set_ylabel('平均峰值冲击 (%)')
        ax4.set_title('按事件类型的平均冲击', fontsize=12)
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 标注事件数
        for i, (bar, n) in enumerate(zip(bars, n_events)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'n={n}', ha='center', va='bottom', fontsize=8)
    
    # ========== 图5: 按官方来源聚合的冲击 ==========
    ax5 = fig.add_subplot(4, 2, 5)
    
    if 'by_author' in impact_results and impact_results['by_author']:
        authors_sorted = sorted(impact_results['by_author'].items(),
                               key=lambda x: -abs(x[1]['mean_peak']))
        
        auths = [a[0] for a in authors_sorted]
        peaks = [a[1]['mean_peak'] for a in authors_sorted]
        n_events = [a[1]['n_events'] for a in authors_sorted]
        
        colors = ['green' if p > 0 else 'red' for p in peaks]
        bars = ax5.barh(range(len(auths)), peaks, color=colors, alpha=0.7, edgecolor='black')
        
        ax5.set_yticks(range(len(auths)))
        ax5.set_yticklabels(auths, fontsize=9)
        ax5.set_xlabel('平均峰值冲击 (%)')
        ax5.set_title('按官方来源的平均冲击', fontsize=12)
        ax5.axvline(0, color='black', linewidth=0.5)
        ax5.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, n) in enumerate(zip(bars, n_events)):
            ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'n={n}', ha='left', va='center', fontsize=8)
    
    # ========== 图6: 冲击响应时间分布 ==========
    ax6 = fig.add_subplot(4, 2, 6)
    
    if 'by_combo' in impact_results:
        peak_times = [data['peak_time_hours'] for data in impact_results['by_combo'].values()
                     if 'peak_time_hours' in data]
        
        if peak_times:
            ax6.hist(peak_times, bins=15, alpha=0.7, edgecolor='black', color='steelblue')
            ax6.axvline(np.mean(peak_times), color='red', linestyle='--', 
                       label=f'平均响应时间: {np.mean(peak_times):.1f}h')
            ax6.set_xlabel('峰值响应时间 (小时)')
            ax6.set_ylabel('组合数')
            ax6.set_title('冲击响应时间分布', fontsize=12)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # ========== 图7: 衰减曲线对比 ==========
    ax7 = fig.add_subplot(4, 2, 7)
    
    decay_plotted = False
    for combo, data in impact_results.get('by_combo', {}).items():
        if data.get('decay') and data['decay']['r_squared'] > 0.4:
            decay_info = data['decay']
            peak_idx = data['peak_time']
            
            if peak_idx >= len(data['mean_impact']) - 3:
                continue
            
            # 原始数据
            time_axis = np.arange(len(data['mean_impact']) - peak_idx) * 15 / 60
            values = np.abs(data['mean_impact'][peak_idx:])
            
            color = combo_color_map.get(combo, 'gray')
            ax7.scatter(time_axis, values, alpha=0.3, s=15, color=color)
            
            # 拟合曲线
            t_fit = np.linspace(0, time_axis[-1], 50)
            if decay_info['type'] == 'exponential':
                y_fit = exponential_decay(t_fit * 4, *decay_info['params'])
                halflife = decay_info.get('half_life_min', 0) / 60
                label = f"{combo}: τ½={halflife:.1f}h"
            else:
                y_fit = power_decay(t_fit * 4, *decay_info['params'])
                label = f"{combo}: α={decay_info['params'][1]:.2f}"
            
            ax7.plot(t_fit, y_fit, linewidth=2, color=color, label=label)
            decay_plotted = True
    
    if decay_plotted:
        ax7.set_xlabel('峰值后时间 (小时)')
        ax7.set_ylabel('|冲击强度| (%)')
        ax7.set_title('事件冲击衰减曲线', fontsize=12)
        ax7.legend(fontsize=7, loc='upper right')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, '无有效衰减曲线\n(R² > 0.4)', ha='center', va='center', fontsize=12)
        ax7.axis('off')
    
    # ========== 图8: 统计汇总 ==========
    ax8 = fig.add_subplot(4, 2, 8)
    
    summary_text = f"""
    特征: {feature_name}
    使用列: {impact_results.get('column_used', 'N/A')}
    
    ═══════════════════════════════════
    组合事件冲击分析汇总
    ═══════════════════════════════════
    
    分析的组合数: {len(impact_results.get('by_combo', {}))}
    
    """
    
    # 按冲击强度排序
    if 'by_combo' in impact_results:
        sorted_combos = sorted(impact_results['by_combo'].items(),
                              key=lambda x: -abs(x[1]['overall_peak']))
        
        summary_text += "冲击强度排名 (Top 5):\n"
        for i, (combo, data) in enumerate(sorted_combos[:5]):
            direction = "↑" if data['overall_peak'] > 0 else "↓"
            summary_text += f"  {i+1}. {combo}\n"
            summary_text += f"     {direction} {abs(data['overall_peak']):.1f}% @ {data['peak_time_hours']:.1f}h\n"
            summary_text += f"     事件数: {data['n_events']}\n"
            if data.get('decay') and data['decay']['r_squared'] > 0.3:
                if data['decay']['type'] == 'exponential':
                    halflife = data['decay'].get('half_life_min', 0)
                    summary_text += f"     半衰期: {halflife/60:.1f}h\n"
    
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax8.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature_name}_组合事件冲击分析.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 图片已保存: {output_dir}/{feature_name}_组合事件冲击分析.png")


# ==================== 主函数 ====================

def run_step4_combo_impact_v2(df_decomposed, feature_list, df_off,
                          use_residual=True, output_dir=None):
    """
    步骤4主函数：官方来源×事件类型 组合冲击分析
    
    Parameters:
    -----------
    df_decomposed : pd.DataFrame, 步骤3的输出（含分解成分）
    feature_list : list, 特征列表
    df_off : pd.DataFrame, 官方事件数据（需要timestamp, category, official_author）
    use_residual : bool, 是否优先使用残差
    output_dir : str, 输出目录
    
    Returns:
    --------
    all_impact_results : dict, 所有特征的组合冲击结果
    all_regression_results : dict, 回归分析结果
    summary_df : pd.DataFrame, 汇总表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("📌 步骤4：官方来源×事件类型 组合冲击分析")
    print("="*70)
    
    # 准备事件数据
    df_events, combo_stats, combo_color_map = prepare_combo_event_data(df_off)
    
    # 保存组合统计
    combo_stats.to_csv(f'{output_dir}/事件组合统计.csv', index=False, encoding='utf-8-sig')
    
    all_impact_results = {}
    all_regression_results = {}
    
    for feature in feature_list:
        print(f"\n{'='*60}")
        print(f"🔧 分析特征: {feature}")
        print('='*60)
        
        # 组合冲击分析
        impact_result = determine_best_decay_type(
            df_decomposed, df_events, feature, window_after=32
        )
        
        if impact_result and 'by_combo' in impact_result and impact_result['by_combo']:
            all_impact_results[feature] = impact_result
            
            # 打印关键发现
            sorted_combos = sorted(impact_result['by_combo'].items(),
                                  key=lambda x: -abs(x[1]['overall_peak']))
            
            print("\n   📈 冲击强度排名:")
            for i, (combo, data) in enumerate(sorted_combos[:3]):
                direction = "↑正向" if data['overall_peak'] > 0 else "↓负向"
                print(f"      {i+1}. {combo}: {direction} {abs(data['overall_peak']):.1f}%")
            
            # 绑定绘图
            plot_combo_impact(df_decomposed, feature, df_events,
                            impact_result, combo_color_map, output_dir)
        else:
            print(f"   ⚠️ 无有效冲击数据")
        
        # 回归分析
        reg_result = combo_regression_analysis_dynamic(
            df_decomposed, feature, df_events, use_residual=use_residual
        )
        
        if reg_result:
            all_regression_results[feature] = reg_result
    
    # ========== 生成汇总表 ==========
    # ========== 1. 生成回归模型详细评估表 (人读) ==========
    # 这是一个长表，包含每个特征、每个组合的详细回归指标
    regression_details = []
    
    for feature, reg_res in all_regression_results.items():
        # 基础模型信息
        base_info = {
            '特征': feature,
            '拟合优度(R2)': round(reg_res['r_squared'], 4),
            '截距项(Intercept)': round(reg_res['intercept'], 6),
            '显著组合数': len(reg_res['significant_combos'])
        }
        
        # 遍历该特征下的所有组合
        for combo, coef_info in reg_res['coefficients'].items():
            row = base_info.copy()
            row.update({
                '组合名称': combo,
                '回归系数(Coef)': round(coef_info['coef'], 6),
                'P值(P-value)': round(coef_info['pvalue'], 4),
                '显著性': '⭐⭐' if coef_info['pvalue'] < 0.01 else ('⭐' if coef_info['pvalue'] < 0.05 else ''),
                '衰减类型': coef_info['decay_type'],
                '衰减参数': round(coef_info['decay_param'], 2)
            })
            regression_details.append(row)
            
    if regression_details:
        df_details = pd.DataFrame(regression_details)
        # 调整列顺序，让人更容易阅读
        cols = ['特征', '组合名称', '回归系数(Coef)', 'P值(P-value)', '显著性', 
                '拟合优度(R2)', '截距项(Intercept)', '衰减类型', '衰减参数']
        df_details = df_details[cols]
        df_details.to_csv(f'{output_dir}/回归模型详细评估.csv', index=False, encoding='utf-8-sig')
        print(f"   📝 详细评估表已保存: {output_dir}/回归模型详细评估.csv")

    # ========== 2. 导出合成所需的完整参数 JSON (机器读) ==========
    # 将 regression results 转换为 JSON 友好的格式
    # 结构: {feature: {intercept: ..., combos: {combo_name: {coef, decay...}}}}
    json_params = {}
    for feature, reg_res in all_regression_results.items():
        json_params[feature] = {
            'intercept': reg_res['intercept'],
            'r_squared': reg_res['r_squared'],
            'coefficients': reg_res['coefficients'] # 包含 coef, pvalue, type, param
        }
        
    import json
    with open(f'{output_dir}/regression_params_for_synthesis.json', 'w', encoding='utf-8') as f:
        json.dump(json_params, f, indent=2, ensure_ascii=False)
    print(f"   💾 合成参数文件已保存: {output_dir}/regression_params_for_synthesis.json")
    
    summary_data = []
    
    for feature in feature_list:
        row = {'特征': feature}
        
        if feature in all_impact_results:
            impact = all_impact_results[feature]
            
            # 找最强冲击的组合
            if impact['by_combo']:
                max_combo = max(impact['by_combo'].items(),
                               key=lambda x: abs(x[1]['overall_peak']))
                row['最强组合'] = max_combo[0]
                row['最大冲击%'] = f"{max_combo[1]['overall_peak']:.1f}"
                row['响应时间h'] = f"{max_combo[1]['peak_time_hours']:.1f}"
                
                # 衰减半衰期
                if max_combo[1].get('decay') and max_combo[1]['decay'].get('half_life_min'):
                    row['半衰期h'] = f"{max_combo[1]['decay']['half_life_min']/60:.1f}"
                else:
                    row['半衰期h'] = '-'
        
        if feature in all_regression_results:
            reg = all_regression_results[feature]
            row['R²'] = f"{reg['r_squared']:.4f}"
            row['显著组合数'] = len(reg['significant_combos'])
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/组合事件冲击汇总.csv', index=False, encoding='utf-8-sig')
    
    # ========== 生成组合效应对比表 ==========
    combo_effect_data = []
    for feature in all_impact_results:
        for combo, data in all_impact_results[feature]['by_combo'].items():
            combo_effect_data.append({
                '特征': feature,
                '组合': combo,
                '事件数': data['n_events'],
                '峰值冲击%': round(data['overall_peak'], 2),
                '响应时间h': round(data['peak_time_hours'], 2),
                '方向': '正向' if data['overall_peak'] > 0 else '负向'
            })
    
    combo_effect_df = pd.DataFrame(combo_effect_data)
    combo_effect_df.to_csv(f'{output_dir}/全部组合效应明细.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ 步骤4完成！")
    print(f"📁 结果保存至: {output_dir}")
    print("="*70)
    
    print("\n📋 组合事件冲击汇总:")
    print(summary_df.to_string(index=False))
    
    # 统计
    if combo_effect_data:
        print(f"\n📊 统计: 分析了 {len(all_impact_results)} 个特征 × {df_events['combo'].nunique()} 个组合")
        
        # 找出影响最大的组合
        combo_avg = combo_effect_df.groupby('组合')['峰值冲击%'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
        print("\n📈 平均冲击强度最大的组合 (Top 5):")
        for combo, avg_impact in combo_avg.head(5).items():
            print(f"   • {combo}: 平均|冲击|={avg_impact:.1f}%")
    
    return all_impact_results, all_regression_results, summary_df, df_decomposed

