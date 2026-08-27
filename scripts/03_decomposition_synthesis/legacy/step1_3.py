import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 版本周期工具 ====================

def prepare_version_periods(version_dict,VERSION_ABBR,VERSION_COLORS):
    """
    将 version_dict 变成 DataFrame + changepoints 列表
    version_dict: { '大版本上半预热': ('2025-01-20 12:00', '2025-01-24 11:00'), ... }
    """
    periods = []
    changepoints = []

    for name, (start, end) in version_dict.items():
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        periods.append({
            'name': name,
            'abbr': VERSION_ABBR.get(name, name[:4]),
            'start': start_time,
            'end': end_time,
            'duration_hours': (end_time - start_time).total_seconds() / 3600,
            'color': VERSION_COLORS.get(name, '#E0E0E0')
        })
        changepoints.append(start_time)

    periods_df = pd.DataFrame(periods).sort_values('start').reset_index(drop=True)

    print("="*70)
    print("📋 版本周期信息:")
    print("="*70)
    for _, row in periods_df.iterrows():
        print(f"   • {row['name']}")
        print(f"     {row['start'].strftime('%m/%d %H:%M')} ~ {row['end'].strftime('%m/%d %H:%M')} "
              f"({row['duration_hours']:.1f}小时)")

    return periods_df, changepoints


def add_version_background(ax, periods_df, y_min, y_max):
    """在图中画版本区间背景"""
    if periods_df is None or len(periods_df) == 0:
        return
    for _, row in periods_df.iterrows():
        rect = Rectangle(
            (mdates.date2num(row['start']), y_min),
            mdates.date2num(row['end']) - mdates.date2num(row['start']),
            y_max - y_min,
            facecolor=row['color'],
            alpha=0.2,
            edgecolor='none',
            zorder=0
        )
        ax.add_patch(rect)


# ==================== Prophet 拟合 & 分解 ====================

def fit_prophet_for_feature(series, changepoints=None,
                            weekly_seasonality=True,
                            daily_seasonality=True,
                            changepoint_prior_scale=0.1):
    """
    使用 Prophet 在【变换后的原始序列】上做趋势+季节性分解
    series: pd.Series, index 为 DatetimeIndex, 值为变换后的 y
    """
    # 清洗
    series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series_clean) < 100:
        print("   ⚠️ 有效点数 < 100，跳过 Prophet 拟合")
        return None, None, None

    df_p = pd.DataFrame({'ds': series_clean.index, 'y': series_clean.values})

    # 过滤 changepoints 在数据范围内
    if changepoints:
        data_start = series_clean.index.min()
        data_end = series_clean.index.max()
        valid_changepoints = [cp for cp in changepoints if data_start <= cp <= data_end]
    else:
        valid_changepoints = None

    print(f"   有效版本变化点: {len(valid_changepoints) if valid_changepoints else 0}")

    # 初始化 Prophet
    model = Prophet(
        yearly_seasonality=False,          # 不做年季节性
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,  # 日内季节性（对于 15min 数据）
        changepoints=valid_changepoints if valid_changepoints else None,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95
    )

    # 拟合
    model.fit(df_p)

    # 预测（样本内）
    forecast = model.predict(df_p[['ds']])

    # R²
    y_actual = df_p['y'].values
    y_pred = forecast['yhat'].values
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"   Prophet R² = {r2:.4f}")
    return model, forecast, r2


def build_decomp_from_prophet(series, forecast):
    """
    基于 Prophet 结果构建类似 STL 的分解结果字典
    输出给 Step9 使用（seasonality模板）、以及统计用。
    """
    series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    df_fit = pd.DataFrame({'y': series_clean})
    df_fit['yhat'] = forecast['yhat'].values[:len(series_clean)]
    df_fit['trend'] = forecast['trend'].values[:len(series_clean)]

    # Prophet 的季节性列名可能有: 'weekly', 'daily'
    if 'daily' in forecast.columns:
        df_fit['seasonal_day'] = forecast['daily'].values[:len(series_clean)]
    else:
        df_fit['seasonal_day'] = 0.0

    if 'weekly' in forecast.columns:
        df_fit['seasonal_week'] = forecast['weekly'].values[:len(series_clean)]
    else:
        df_fit['seasonal_week'] = 0.0

    df_fit['resid'] = df_fit['y'] - df_fit['yhat']

    # 方差占比（仅做描述，不追求严格正交）
    total_var = df_fit['y'].var()
    if total_var <= 0:
        var_decomp = {'trend': 0, 'seasonal': 0, 'residual': 100}
    else:
        trend_var = df_fit['trend'].var()
        seas_var = (df_fit['seasonal_day'] + df_fit['seasonal_week']).var()
        resid_var = df_fit['resid'].var()
        var_decomp = {
            'trend': trend_var / total_var * 100,
            'seasonal': seas_var / total_var * 100,
            'residual': resid_var / total_var * 100
        }

    # 构造类似 STL 的结果结构
    result = {
        'trend_day': pd.Series(df_fit['trend'].values, index=series_clean.index),
        'seasonal_day': pd.Series(df_fit['seasonal_day'].values, index=series_clean.index),
        'seasonal_week': pd.Series(df_fit['seasonal_week'].values, index=series_clean.index),
        'resid_final': pd.Series(df_fit['resid'].values, index=series_clean.index),
        'var_decomposition_day': var_decomp,
        'n_points': len(series_clean),
        'date_range': (series_clean.index.min(), series_clean.index.max())
    }
    return result, df_fit


def analyze_version_effects_prophet(series, forecast, periods_df, feature_name):
    """
    对每个版本周期统计：
      - 期间内 y 均值 / std（变换域）
      - Prophet 残差均值 / std
      - 趋势起点/终点/变化量
    返回 DataFrame（多特征后面 concat）
    """
    series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series_clean) == 0 or forecast is None:
        return pd.DataFrame([])

    df_tmp = pd.DataFrame({
        'ds': series_clean.index,
        'y': series_clean.values,
        'yhat': forecast['yhat'].values[:len(series_clean)],
        'trend': forecast['trend'].values[:len(series_clean)],
    })
    df_tmp['resid'] = df_tmp['y'] - df_tmp['yhat']

    stats_list = []
    for _, p in periods_df.iterrows():
        mask = (df_tmp['ds'] >= p['start']) & (df_tmp['ds'] < p['end'])
        sub = df_tmp[mask]
        if len(sub) == 0:
            continue
        stats_list.append({
            'feature': feature_name,
            'version': p['name'],
            'abbr': p['abbr'],
            'n_points': len(sub),
            'mean_value': sub['y'].mean(),
            'std_value': sub['y'].std(),
            'mean_resid': sub['resid'].mean(),
            'std_resid': sub['resid'].std(),
            'trend_start': sub['trend'].iloc[0],
            'trend_end': sub['trend'].iloc[-1],
            'trend_change': sub['trend'].iloc[-1] - sub['trend'].iloc[0]
        })
    return pd.DataFrame(stats_list)


# ==================== 可视化 ====================

def plot_prophet_decomposition(series, forecast, feature_name,
                               r_squared, periods_df, output_dir):
    """
    绘制单个特征的 Prophet 分解图（原 STL 图的 Prophet 版）
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    df_plot = pd.DataFrame({
        'ds': series_clean.index,
        'y': series_clean.values,
        'yhat': forecast['yhat'].values[:len(series_clean)],
        'trend': forecast['trend'].values[:len(series_clean)],
    })
    if 'weekly' in forecast.columns:
        df_plot['weekly'] = forecast['weekly'].values[:len(series_clean)]
    else:
        df_plot['weekly'] = 0.0
    if 'daily' in forecast.columns:
        df_plot['daily'] = forecast['daily'].values[:len(series_clean)]
    else:
        df_plot['daily'] = 0.0
    df_plot['resid'] = df_plot['y'] - df_plot['yhat']

    fig = plt.figure(figsize=(20, 14))

    # 1) 原始+拟合
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(df_plot['ds'], df_plot['y'], 'b-', linewidth=0.6, alpha=0.7, label='实际值')
    ax1.plot(df_plot['ds'], df_plot['yhat'], 'r-', linewidth=1.0, alpha=0.9, label='Prophet拟合')
    y_min = df_plot['y'].min() - df_plot['y'].std()
    y_max = df_plot['y'].max() + df_plot['y'].std()
    add_version_background(ax1, periods_df, y_min, y_max)
    ax1.set_title(f'{feature_name} - Prophet拟合 (R²={r_squared:.4f})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.tick_params(axis='x', rotation=30)

    # 2) 趋势
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(df_plot['ds'], df_plot['trend'], 'orange', linewidth=1.0)
    t_min = df_plot['trend'].min() - df_plot['trend'].std()
    t_max = df_plot['trend'].max() + df_plot['trend'].std()
    add_version_background(ax2, periods_df, t_min, t_max)
    ax2.set_title('趋势 (Trend)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.tick_params(axis='x', rotation=30)

    # 3) 周季节性
    ax3 = fig.add_subplot(4, 1, 3)
    # 按星期几聚合 weekly
    weekly = df_plot.set_index('ds')['weekly']
    weekly_pattern = weekly.groupby(weekly.index.dayofweek).mean()
    dow_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    ax3.bar(range(7), weekly_pattern.values, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(dow_labels)
    ax3.set_ylabel('季节性')
    ax3.set_title('周季节性 (Weekly)', fontsize=11)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4) 日内季节性
    ax4 = fig.add_subplot(4, 1, 4)
    daily = df_plot.set_index('ds')['daily']
    hourly_pattern = daily.groupby(daily.index.hour + daily.index.minute/60).mean()
    ax4.plot(hourly_pattern.index, hourly_pattern.values, 'g-', linewidth=1.5)
    ax4.fill_between(hourly_pattern.index, hourly_pattern.values, alpha=0.3, color='green')
    ax4.set_xlabel('小时')
    ax4.set_ylabel('季节性')
    ax4.set_title('日内季节性 (Daily)', fontsize=11)
    ax4.set_xticks(range(0, 25, 3))
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature_name}_Prophet分解.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   📊 图片已保存: {output_dir}/{feature_name}_Prophet分解.png')


# ==================== 主函数：合并版 Step3_Prophet ====================

def run_step3_prophet(
    df_transformed,       # 步骤2输出（含 feature_transformed 等），索引为 DatetimeIndex
    feature_list,
    version_dict=None,   # {版本名: (start, end)}，可选；若无版本分析则传 None
    VERSION_ABBR=None,
    VERSION_COLORS=None,
    output_dir=None
):
    """
    合并版 Step3：
      - 用 Prophet 对各特征做趋势+季节性分解（替代 STL）
      - 在变换后的原始序列上做版本周期统计（替代原 Step5）
      - 生成 Prophet 残差列（供 Step4 事件冲击、Step6 GARCH 使用）
      - 生成 decomp_results（供 Step9 合成中抽取季节性模板）

    返回：
      prophet_results : {feature: {model, forecast, r_squared}}
      df_prophet_resid : DataFrame，在 df_transformed 基础上增加：
                         - feature_resid（Prophet 残差）
                         - feature_prophet_resid（同上）
      decomp_results : {feature: {...}}，包含 seasonal_day/seasonal_week 等
      all_version_stats_df : 所有特征×版本的统计明细（Step9 生成 version_offsets 用）
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("📌 合并版 Step3：Prophet 分解 + 版本统计")
    print("="*70)

    # 确保时间索引
    if not isinstance(df_transformed.index, pd.DatetimeIndex):
        df_transformed = df_transformed.copy()
        df_transformed.index = pd.to_datetime(df_transformed.index)

    print(f"   数据范围: {df_transformed.index.min()} ~ {df_transformed.index.max()}")
    print(f"   数据点数: {len(df_transformed)}")

    # 版本周期信息
    if version_dict is not None:
        periods_df, version_changepoints = prepare_version_periods(version_dict,VERSION_ABBR,VERSION_COLORS)
    else:
        periods_df, version_changepoints = pd.DataFrame(), None

    prophet_results = {}
    df_prophet_resid = df_transformed.copy()
    decomp_results = {}
    all_version_stats = []

    for feature in feature_list:
        print(f"\n{'='*60}")
        print(f"🔧 Prophet分解: {feature}")
        print('='*60)

        # 选择用于 Prophet 的列：优先 *_transformed，其次原始列
        col_name = f'{feature}_transformed' if f'{feature}_transformed' in df_transformed.columns else feature
        if col_name not in df_transformed.columns:
            print(f"   ❌ 列 {col_name} 不存在，跳过")
            continue

        series = df_transformed[col_name]
        print(f"   使用列: {col_name}")
        print(f"   非空点数: {series.replace([np.inf, -np.inf], np.nan).dropna().shape[0]}")

        # Prophet 拟合
        model, forecast, r2 = fit_prophet_for_feature(
            series,
            changepoints=version_changepoints,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.1
        )
        if model is None:
            continue

        prophet_results[feature] = {
            'model': model,
            'forecast': forecast,
            'r_squared': r2,
            'column_used': col_name
        }

        # 生成分解结果（给 Step9 用）
        decomp_result, df_fit = build_decomp_from_prophet(series, forecast)
        decomp_results[feature] = decomp_result

        # 残差序列：对齐到原 index，其他位置为 NaN
        resid_series = pd.Series(index=df_transformed.index, dtype=float)
        resid_series.loc[df_fit.index] = df_fit['resid'].values

        # 保存为两种命名：feature_resid（供 Step4）、feature_prophet_resid（供 Step6）
        df_prophet_resid[f'{feature}_resid'] = resid_series
        # df_prophet_resid[f'{feature}_prophet_resid'] = resid_series

        # 版本统计（若有 version_dict）
        if version_dict is not None and len(periods_df) > 0:
            v_stats = analyze_version_effects_prophet(
                series, forecast, periods_df, feature_name=feature
            )
            if len(v_stats) > 0:
                all_version_stats.append(v_stats)
                # 打印趋势变化最大的一段
                max_idx = v_stats['trend_change'].abs().idxmax()
                row = v_stats.loc[max_idx]
                direction = '↑上升' if row['trend_change'] > 0 else '↓下降'
                print(f"   📈 趋势变化最大: {row['version']} ({direction} Δ={row['trend_change']:.4f})")

        # 绘图
        plot_prophet_decomposition(
            series, forecast, feature,
            r_squared=r2,
            periods_df=periods_df,
            output_dir=output_dir
        )

    # 汇总版本统计
    if all_version_stats:
        all_version_stats_df = pd.concat(all_version_stats, ignore_index=True)
        all_version_stats_df.to_csv(
            f'{output_dir}/版本周期统计明细_Prophet.csv', index=False, encoding='utf-8-sig'
        )
    else:
        all_version_stats_df = pd.DataFrame()

    # 生成 Prophet 拟合汇总表
    summary_rows = []
    for feature, res in prophet_results.items():
        row = {
            '特征': feature,
            'Prophet_R²': f"{res['r_squared']:.4f}",
            '使用列': res['column_used']
        }
        # 若有版本统计，附上趋势变化最大的版本
        if not all_version_stats_df.empty:
            vs = all_version_stats_df[all_version_stats_df['feature'] == feature]
            if not vs.empty:
                max_idx = vs['trend_change'].abs().idxmax()
                r = vs.loc[max_idx]
                row['最大变化版本'] = r['abbr']
                row['趋势变化'] = f"{r['trend_change']:.4f}"
                row['变化方向'] = '↑' if r['trend_change'] > 0 else '↓'
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{output_dir}/Prophet分解汇总.csv', index=False, encoding='utf-8-sig')

    print("\n" + "="*70)
    print("✅ 合并版 Step3_Prophet 完成！")
    print(f"📁 结果保存至: {output_dir}")
    print("="*70)

    if not summary_df.empty:
        print("\n📋 Prophet分解汇总:")
        print(summary_df.to_string(index=False))

    return prophet_results, df_prophet_resid, decomp_results, all_version_stats_df, summary_df