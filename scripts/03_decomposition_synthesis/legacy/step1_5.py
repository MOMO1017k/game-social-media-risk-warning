import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from arch import arch_model
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== GARCH 配置 ====================

VERSION_COLORS = {
    '大版本上半预热': '#FFCDD2', '大版本上半更新': '#EF5350',
    '大版本下半预热': '#FFECB3', '大版本下半更新': '#FFA726',
    '小版本上半预热': '#C8E6C9', '小版本上半更新': '#66BB6A',
    '小版本下半预热': '#BBDEFB', '小版本下半更新': '#42A5F5',
}

VERSION_ABBR = {
    '大版本上半预热': '大上预', '大版本上半更新': '大上更',
    '大版本下半预热': '大下预', '大版本下半更新': '大下更',
    '小版本上半预热': '小上预', '小版本上半更新': '小上更',
    '小版本下半预热': '小下预', '小版本下半更新': '小下更',
}


# ==================== 数据准备 ====================

def prepare_garch_data(df, feature_name):
    """
    准备GARCH建模数据
    修改：优先使用 AR 残差 (_ar_resid)，假设均值回归已在 Step 1.4.5 完成
    """
    # 1. 优先查找 AR 残差
    col_name = f'{feature_name}_ar_resid'
    
    # 2. 如果没有，回退查找 Prophet 残差 (兼容旧流程，但会警告)
    if col_name not in df.columns:
        print(f"   ⚠️ 未找到 {col_name}，尝试查找 Prophet 残差...")
        for suffix in ['_prophet_resid', '_resid', '']:
            alt_col = f'{feature_name}{suffix}'
            if alt_col in df.columns:
                col_name = alt_col
                print(f"   ⚠️ 使用 {col_name} 代替 (建议先运行 AR 过滤)")
                break
    
    if col_name not in df.columns:
        print(f"   ❌ 找不到 {feature_name} 的残差数据列")
        return None
    
    print(f'   使用列: {col_name}')
    series = df[col_name].copy()
    
    # 清洗数据
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 检查数据质量
    if len(series) < 200:
        print(f"   ⚠️ 数据不足200条 ({len(series)})")
        return None
    
    # 标准化（GARCH对尺度敏感，转为百分比尺度或标准化尺度）
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0 or np.isnan(std_val):
        print(f"   ⚠️ 标准差为0或NaN")
        return None
    
    # 缩放: (x - mu) / sigma * 100
    # 乘以100是为了让优化器更容易收敛（避免数值过小）
    series_scaled = (series - mean_val) / std_val * 100  
    
    return series_scaled, mean_val, std_val


# ==================== GARCH 模型拟合 ====================

def fit_garch_models(series, max_p=2, max_q=2):
    """
    拟合多种GARCH模型配置，并在 正态分布 和 t分布 之间进行优选
    
    Returns:
    --------
    best_model : 最优模型结果对象
    best_name : 最优模型名称
    all_results : 所有模型的结果字典
    """
    all_results = {}
    best_aic = np.inf
    best_model = None
    best_name = None
    
    # 确保数据是numpy数组
    data = series.values if hasattr(series, 'values') else series
    
    # 定义要尝试的分布类型
    # Normal: 标准正态分布
    # t: Student's t 分布 (捕获厚尾特征)
    distributions = ['Normal', 't'] 
    
    # 定义模型配置生成器
    configs = []
    
    # 1. 标准 GARCH (遍历 p, q)
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            configs.append({'vol': 'Garch', 'p': p, 'q': q, 'o': 0, 'name': f'GARCH({p},{q})'})
            
    # 2. EGARCH (非对称) - 固定 p=1, q=1 以减少计算量，通常足够
    configs.append({'vol': 'EGARCH', 'p': 1, 'q': 1, 'o': 0, 'name': 'EGARCH(1,1)'})
    
    # 3. GJR-GARCH (杠杆) - 固定 p=1, o=1, q=1
    configs.append({'vol': 'Garch', 'p': 1, 'q': 1, 'o': 1, 'name': 'GJR-GARCH(1,1,1)'})

    # 开始遍历所有组合
    for config in configs:
        for dist in distributions:
            try:
                # 构建模型名称
                full_name = f"{config['name']}-{dist}"
                
                # 初始化模型
                # mean='Constant' 或 'Zero'。因为输入已经是 AR 残差，理论上均值为0，但设为 Constant 更稳健
                model = arch_model(data, 
                                   vol=config['vol'], 
                                   p=config['p'], 
                                   q=config['q'], 
                                   o=config['o'],
                                   dist=dist,      # 关键修改：传入分布类型
                                   mean='Constant', 
                                   rescale=False)
                
                # 拟合
                result = model.fit(disp='off', show_warning=False)
                
                # 记录结果
                all_results[full_name] = {
                    'model': result,
                    'aic': result.aic,
                    'bic': result.bic,
                    'loglik': result.loglikelihood,
                    'dist': dist
                }
                
                # 更新最优模型 (AIC 越小越好)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
                    best_name = full_name
                    
            except Exception as e:
                # 某些复杂模型可能无法收敛，跳过
                pass
    
    if best_model is None:
        print("   ❌ 所有GARCH模型拟合失败")
        return None, None, None
    
    print(f"   最优模型: {best_name} (AIC={best_aic:.2f})")
    
    return best_model, best_name, all_results


def analyze_garch_results(model_result, series, model_name):
    """
    分析GARCH模型结果
    """
    analysis = {}
    params = model_result.params
    
    # 1. 基础数据
    analysis['cond_vol'] = model_result.conditional_volatility
    analysis['std_resid'] = model_result.std_resid
    analysis['aic'] = model_result.aic
    analysis['bic'] = model_result.bic
    
    # 2. 识别分布类型和参数
    is_t_dist = 't' in model_name or 'nu' in params
    analysis['distribution'] = 'Student-t' if is_t_dist else 'Normal'
    if is_t_dist:
        analysis['nu'] = params.get('nu', 5.0) # 自由度
        
    # 3. 计算持续性 (Persistence)
    # 对于 GARCH/GJR: Persistence = alpha + beta + 0.5*gamma
    # 对于 EGARCH: Persistence = beta
    if 'EGARCH' in model_name:
        analysis['persistence'] = abs(params.get('beta[1]', 0))
    else:
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        gamma = params.get('gamma[1]', 0)
        analysis['persistence'] = alpha + beta + 0.5 * gamma
        
    analysis['has_clustering'] = analysis['persistence'] > 0.7
    
    # 4. 统计特征
    analysis['mean_vol'] = analysis['cond_vol'].mean()
    analysis['max_vol'] = analysis['cond_vol'].max()
    
    # 5. Ljung-Box 检验 (检验标准化残差平方的自相关 -> 检查 ARCH 效应是否消除)
    try:
        # 检验 std_resid^2
        lb2 = acorr_ljungbox(model_result.std_resid ** 2, lags=[10], return_df=True)
        analysis['lb_p_sq'] = float(lb2['lb_pvalue'].iloc[0])
        analysis['arch_cleared'] = analysis['lb_p_sq'] > 0.05
    except:
        analysis['lb_p_sq'] = np.nan
        analysis['arch_cleared'] = False
    
    return analysis


def compute_confidence_intervals(series, model_result, confidence_levels=[0.95, 0.99]):
    """
    计算动态置信区间
    修改：根据模型分布（Normal vs t）动态选择临界值
    """
    cond_vol = model_result.conditional_volatility
    mean_pred = model_result.params.get('mu', 0)
    
    # 判断是否为 t 分布
    params = model_result.params
    if 'nu' in params:
        dist_type = 't'
        nu = params['nu']
    else:
        dist_type = 'norm'
    
    ci_dict = {'timestamp': series.index[:len(cond_vol)], 'mean': mean_pred}
    
    for cl in confidence_levels:
        alpha = 1 - cl
        # 计算双尾临界值
        if dist_type == 't':
            # t分布的分位数
            crit_val = stats.t.ppf(1 - alpha/2, df=nu)
        else:
            # 正态分布的分位数
            crit_val = stats.norm.ppf(1 - alpha/2)
            
        ci_dict[f'upper_{int(cl*100)}'] = mean_pred + crit_val * cond_vol
        ci_dict[f'lower_{int(cl*100)}'] = mean_pred - crit_val * cond_vol
    
    ci_df = pd.DataFrame(ci_dict)
    ci_df.set_index('ds', inplace=True)
    
    return ci_df


# ==================== 可视化 ====================

def add_version_background(ax, periods_df, y_min, y_max):
    """添加版本周期背景"""
    if periods_df is None: return
    for _, row in periods_df.iterrows():
        rect = Rectangle(
            (mdates.date2num(row['start']), y_min),
            mdates.date2num(row['end']) - mdates.date2num(row['start']),
            y_max - y_min,
            facecolor=row['color'], alpha=0.2, edgecolor='none', zorder=0
        )
        ax.add_patch(rect)

def plot_garch_results(series, model_result, model_name, analysis, 
                      ci_df, periods_df, feature_name, output_dir):
    """
    绘制GARCH分析结果（包含 t 分布拟合展示）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 15))
    cond_vol = analysis['cond_vol']
    std_resid = analysis['std_resid']
    
    # ========== 图1: 原始残差 + 动态置信区间 ==========
    ax1 = fig.add_subplot(3, 2, 1)
    
    y_data = series.values[:len(cond_vol)]
    y_min, y_max = np.nanmin(y_data) * 1.1, np.nanmax(y_data) * 1.1
    
    add_version_background(ax1, periods_df, y_min, y_max)
    
    ax1.plot(series.index, y_data, linewidth=0.5, color='steelblue', label='AR残差')
    ax1.fill_between(ci_df.index, ci_df['lower_95'], ci_df['upper_95'],
                    alpha=0.3, color='orange', label='95% CI')
    ax1.fill_between(ci_df.index, ci_df['lower_99'], ci_df['upper_99'],
                    alpha=0.15, color='red', label='99% CI')
    
    ax1.set_ylim(y_min, y_max)
    ax1.set_title(f'{feature_name} - 动态波动区间 ({model_name})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ========== 图2: 条件波动率 ==========
    ax2 = fig.add_subplot(3, 2, 2)
    vol_min, vol_max = cond_vol.min() * 0.9, cond_vol.max() * 1.1
    add_version_background(ax2, periods_df, vol_min, vol_max)
    
    ax2.plot(series.index, cond_vol, color='red', linewidth=1, label='条件波动率')
    ax2.axhline(cond_vol.mean(), color='black', linestyle='--', label='均值')
    
    ax2.set_title('条件波动率 (Conditional Volatility)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== 图3: 标准化残差分布 vs 理论分布 ==========
    ax3 = fig.add_subplot(3, 2, 3)
    
    ax3.hist(std_resid, bins=50, density=True, alpha=0.6, color='gray', label='标准化残差')
    x_range = np.linspace(std_resid.min(), std_resid.max(), 100)
    
    # 绘制正态分布参照
    ax3.plot(x_range, stats.norm.pdf(x_range), 'g--', linewidth=1.5, label='Normal(0,1)')
    
    # 如果模型是 t 分布，绘制拟合的 t 分布
    if analysis['distribution'] == 'Student-t':
        nu = analysis['nu']
        ax3.plot(x_range, stats.t.pdf(x_range, df=nu), 'r-', linewidth=2, label=f'Model t(df={nu:.1f})')
        
    ax3.set_title(f'残差分布 ({analysis["distribution"]})', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== 图4: Q-Q Plot ==========
    ax4 = fig.add_subplot(3, 2, 4)
    if analysis['distribution'] == 'Student-t':
        stats.probplot(std_resid, dist="t", sparams=(analysis['nu'],), plot=ax4)
        ax4.set_title(f'Q-Q Plot (vs t-dist df={analysis["nu"]:.1f})', fontsize=12)
    else:
        stats.probplot(std_resid, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (vs Normal)', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # ========== 图5: ARCH 效应检验 (残差平方 ACF) ==========
    ax5 = fig.add_subplot(3, 2, 5)
    plot_acf(std_resid**2, ax=ax5, lags=30, alpha=0.05, title='标准化残差平方 ACF (检验ARCH消除)')
    ax5.grid(True, alpha=0.3)
    
    # ========== 图6: 模型汇总信息 ==========
    ax6 = fig.add_subplot(3, 2, 6)
    
    summary_text = f"""
    模型汇总: {feature_name}
    ──────────────────────────
    最优模型: {model_name}
    分布类型: {analysis['distribution']}
    AIC: {analysis['aic']:.1f}
    
    参数:
    """
    if 'nu' in analysis:
        summary_text += f"  自由度 (nu): {analysis['nu']:.2f} (厚尾特征)\n"
    summary_text += f"  持续性: {analysis.get('persistence', 0):.4f}\n"
    summary_text += f"  ARCH效应消除: {'✅ 是' if analysis.get('arch_cleared') else '❌ 否'}\n"
    
    # 异常统计
    upper_99 = ci_df['upper_99'].values
    lower_99 = ci_df['lower_99'].values
    anomalies = (y_data > upper_99) | (y_data < lower_99)
    summary_text += f"\n异常点 (超出99% CI): {anomalies.sum()} ({anomalies.mean():.2%})"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11, va='top', family='monospace')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature_name}_GARCH分析.png', dpi=120)
    plt.close()
    
    return anomalies


# ==================== 主函数 ====================

def run_step5_garch_v3(df_input, feature_list, version_dict=None, output_dir=None):
    """
    步骤5主函数：GARCH波动率建模 (融合优化版)
    
    Parameters:
    -----------
    df_input : pd.DataFrame, 包含 AR 残差 (_ar_resid) 的数据
    feature_list : list, 特征列表
    version_dict : dict, 版本周期字典
    output_dir : str, 输出目录
    
    Returns:
    --------
    garch_results : dict, GARCH建模详细结果
    df_output : pd.DataFrame, 增加了 _garch_std_resid 和 _garch_vol 列的数据
    summary_df : pd.DataFrame, 汇总表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("📌 步骤5：GARCH 波动率建模 (融合优化版)")
    print("   - 输入: AR模型残差 (_ar_resid)")
    print("   - 模型: 自动选择 GARCH/EGARCH/GJR-GARCH + Normal/t-dist")
    print("   - 目标: 获取纯净的标准化残差 (用于Copula)")
    print("="*70)
    
    # 准备版本周期数据
    periods_df = None
    if version_dict is not None:
        periods = []
        for name, (start, end) in version_dict.items():
            periods.append({
                'name': name,
                'start': pd.to_datetime(start), 'end': pd.to_datetime(end),
                'color': VERSION_COLORS.get(name, '#E0E0E0')
            })
        periods_df = pd.DataFrame(periods).sort_values('start').reset_index(drop=True)
    
    garch_results = {}
    df_output = df_input.copy()
    summary_data = []
    
    for feature in feature_list:
        print(f"\n{'='*60}")
        print(f"🔧 GARCH建模: {feature}")
        print('='*60)
        
        # 1. 准备数据 (查找 AR 残差)
        result = prepare_garch_data(df_output, feature)
        if result is None:
            continue
        
        series_scaled, mean_val, std_val = result
        
        # 2. 拟合 GARCH 模型 (对比 Normal 和 t 分布)
        best_model, model_name, all_models = fit_garch_models(series_scaled)
        
        if best_model is None:
            continue
        
        # 3. 分析结果
        analysis = analyze_garch_results(best_model, series_scaled, model_name)
        
        # 4. 计算置信区间 (基于模型分布动态计算)
        ci_df = compute_confidence_intervals(series_scaled, best_model)
        
        # 5. [核心新增] 保存用于 Copula 的关键列
        # 标准化残差 (Std Resid) -> 应该是 i.i.d 的
        # 条件波动率 (Volatility)
        col_std_resid = f'{feature}_garch_std_resid'
        col_vol = f'{feature}_garch_vol'
        
        # 对齐索引赋值
        df_output.loc[series_scaled.index, col_std_resid] = analysis['std_resid']
        df_output.loc[series_scaled.index, col_vol] = analysis['cond_vol']
        
        # 6. 绘图
        anomalies_mask = plot_garch_results(
            series_scaled, best_model, model_name, analysis,
            ci_df, periods_df, feature, output_dir
        )
        
        # 7. 保存结果
        garch_results[feature] = {
            'model': best_model,
            'model_name': model_name,
            'analysis': analysis,
            'params': best_model.params.to_dict(),
            'ci_df': ci_df,
            'scaling': {
                'mean': mean_val, 
                'std': std_val, 
                'scale_factor': 100.0,
                'garch_mu': float(best_model.params.get('mu', 0.0))} # 记录缩放参数以便还原
        }
        
        # 汇总信息
        summary_data.append({
            '特征': feature,
            '最优模型': model_name,
            '分布': analysis['distribution'],
            'AIC': f"{analysis['aic']:.1f}",
            '自由度(nu)': f"{analysis.get('nu', 0):.2f}",
            '持续性': f"{analysis.get('persistence', 0):.4f}",
            'ARCH消除': '✅' if analysis.get('arch_cleared') else '❌',
            '异常比例': f"{anomalies_mask.mean():.2%}"
        })
        
    # 导出汇总
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/GARCH建模汇总.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ 步骤5完成！")
    print(f"📁 结果保存至: {output_dir}")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    return garch_results, df_output, summary_df