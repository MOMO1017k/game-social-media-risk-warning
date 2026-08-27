import pandas as pd
import numpy as np
from prophet.serialize import model_from_json
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 辅助配置 ====================

# 用来解析 version key 到标准类型的逻辑
def parse_version_type(version_name: str):
    """
    将 'v1.2_大上预' 解析为 '大版本上半预热'
    """
    s = str(version_name)
    size = '大版本' if '大' in s else '小版本'
    half = '上半' if '上' in s else '下半'
    phase = '预热' if '预' in s else '更新'
    return f'{size}{half}{phase}'

# ==================== 2. 核心逻辑：构建基准趋势 ====================

def learn_baselines_from_history(df_history, version_dict_history, feature_name):
    """
    从历史数据中学习每种版本类型（如'大版本上半更新'）的平均值
    """
    # 1. 准备容器
    type_values = {}  # { '大版本上半更新': [1.2, 1.3, ...], ... }
    
    # 2. 遍历历史版本字典
    for v_name, (start, end) in version_dict_history.items():
        std_type = parse_version_type(v_name)
        
        # 截取该时间段的历史数据
        mask = (df_history.index >= start) & (df_history.index < end)
        period_data = df_history.loc[mask, feature_name].dropna()
        
        if len(period_data) > 0:
            if std_type not in type_values:
                type_values[std_type] = []
            type_values[std_type].extend(period_data.values)
            
    # 3. 计算均值
    baselines = {}
    global_mean = df_history[feature_name].mean()
    
    # 定义所有可能的类型
    all_types = [
        f'{s}{h}{p}' 
        for s in ['大版本', '小版本'] 
        for h in ['上半', '下半'] 
        for p in ['预热', '更新']
    ]
    
    for t in all_types:
        if t in type_values and len(type_values[t]) > 0:
            baselines[t] = np.mean(type_values[t])
        else:
            # 如果历史没出现过这个类型（比如历史只有小版本，未来有大版本）
            # 这里需要定义简单的推演规则，例如：大版本 = 全局均值 * 1.2 (如果是原始域)
            # 但因为是在【变换域】(Log/BoxCox)，加减法即可
            # 策略：如果没有数据，暂且回退到全局均值，或者根据规则微调
            if '大版本' in t:
                baselines[t] = global_mean + 0.1  # 假设大版本比均值高一点
            else:
                baselines[t] = global_mean
                
    print(f"   📊 学习到的版本基准 ({feature_name}):")
    for k, v in baselines.items():
        print(f"      - {k}: {v:.4f}")
        
    return baselines, global_mean

def construct_schedule_trend(future_index, version_dict_complete, baselines, global_mean,smooth_window=None):
    """
    根据未来排期表，构建阶梯式趋势线
    """
    trend_series = pd.Series(index=future_index, data=global_mean) # 默认填全局均值
    
    for v_name, (start, end) in version_dict_complete.items():
        std_type = parse_version_type(v_name)
        # 获取该类型的基准值
        val = baselines.get(std_type, global_mean)
        
        # 填入时间段
        mask = (trend_series.index >= start) & (trend_series.index < end)
        trend_series.loc[mask] = val
        
    # 平滑处理：消除版本切换时的生硬台阶
    # 使用 Gaussian 或 Rolling Mean 平滑接缝，窗口设为 12小时 (48点)
    # trend_smooth = trend_series.rolling(window=48, min_periods=1, center=True, win_type='gaussian').mean(std=10)
    # trend_smooth = trend_smooth.fillna(trend_series) # 补全边缘
    if smooth_window and smooth_window > 0:
        trend_smooth = trend_series.rolling(
            window=smooth_window, min_periods=1, center=True, win_type='gaussian'
        ).mean(std=smooth_window/3)
        return trend_smooth.fillna(trend_series)
    else:
        return trend_series

def apply_seasonality_from_profile(future_index, profile_path):
    """
    [修改原因]：Step 5 核心替换逻辑。
    不加载模型，而是读取 JSON，根据 future_index 的时间特征（几点几分、星期几）
    直接查找对应的 seasonality 数值。
    """
    if not os.path.exists(profile_path):
        print(f"      ⚠️ 警告：未找到季节性模板 {profile_path}，将使用0填充")
        return np.zeros(len(future_index))
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    # 1. 映射日内季节性 (Daily)
    # JSON key 是 "HH:MM", value 是 float
    # 使用 map 极速查找，比 predict 快几十倍
    time_keys = future_index.strftime('%H:%M')
    daily_map = profile['daily']
    daily_component = pd.Series(time_keys).map(daily_map).fillna(0).values
    
    # 2. 映射周季节性 (Weekly)
    # JSON key 是 string "0", "1"... 但 map 需要匹配
    # 注意：json load 进来 key 可能是字符串，需要确保类型匹配
    weekly_map = {int(k): v for k, v in profile['weekly'].items()}
    dow_keys = future_index.dayofweek
    weekly_component = pd.Series(dow_keys).map(weekly_map).fillna(0).values
    
    return daily_component + weekly_component
# ==================== 3. 主流程 Step 5 ====================

def run_step5_schedule_synthesis(
    df_step4_input,          # 包含噪声和冲击的残差
    df_history,              # 原始历史数据（变换后）
    version_dict_history,    # 历史数据的版本字典
    version_dict_future,     # 你的 version_dict_complete
    prophet_model_dir,
    feature_list,
    output_dir,
    prophet_results,   # [新增参数] 传入 Step 3 返回的 prophet_results 字典
    r2_threshold=0.1   # [新增参数] R² 阈值，默认 0.1
):
    print("=" * 70)
    print("📌 Step 5: 基于排期表的趋势重构 (Schedule-Based Synthesis)")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df_final = pd.DataFrame(index=df_step4_input.index)
    
    for feature in feature_list:
        print(f"\n   Processing {feature}...")
        
        # --- A. 学习基准 ---
        # 确保 feature 在历史数据中存在
        hist_col = f'{feature}_transformed' if f'{feature}_transformed' in df_history.columns else feature
        baselines, global_mean = learn_baselines_from_history(
            df_history, version_dict_history, hist_col
        )
        
        # --- B. 构建趋势 (Trend) ---
        # 根据 version_dict_complete 铺设地基
        trend_component = construct_schedule_trend(
            df_step4_input.index, version_dict_future, baselines, global_mean,smooth_window=0
        )
        
        # --- C. 获取季节性 (Seasonality) ---
        # 使用 Prophet 模型，只预测 seasonality，令 trend=0
        safe_name = feature.replace('/', '_')
        if feature in prophet_results:
            r2 = prophet_results[feature]['r_squared']
        else:
            r2 = 0.0
            print(f"      ⚠️ 未找到 R² 数据，默认为 0")
        if r2 < r2_threshold:
            print(f"      📉 R² ({r2:.4f}) < {r2_threshold}，忽略季节性 (Seasonality=0)")
            seasonality_component = 0.0
        else:
            profile_path = os.path.join(prophet_model_dir, f'profile_{safe_name}.json')
            seasonality_component = apply_seasonality_from_profile(df_step4_input.index, profile_path)
        # model_path = os.path.join(prophet_model_dir, f'prophet_model_{safe_name}.json')
        
        # seasonality_component = 0.0
        # if os.path.exists(model_path):
        #     try:
        #         m = model_from_json(open(model_path, 'r').read())
        #         future_df = pd.DataFrame({'ds': df_step4_input.index})
        #         # 将所有额外回归量置0
        #         if m.extra_regressors:
        #             for reg in m.extra_regressors:
        #                 future_df[reg] = 0.0
                        
        #         forecast = m.predict(future_df)
        #         # Prophet 的 additive_terms 包含 weekly + daily
        #         seasonality_component = forecast['additive_terms'].values
        #     except Exception as e:
        #         print(f"      ⚠️ Prophet 模型加载失败，跳过季节性: {e}")
        
        # --- D. 最终合成 ---
        # Final = 排期趋势 + 周期性 + (残差+冲击)
        resid_component = df_step4_input[feature].values
        
        final_series = trend_component.values + seasonality_component + resid_component
        df_final[feature] = final_series
        
        print(f"      趋势均值: {trend_component.mean():.4f} | 周期幅度: {np.ptp(seasonality_component):.4f}")

    # 保存
    save_path = f'{output_dir}/synthetic_transformed_schedule_based.csv'
    df_final.to_csv(save_path, encoding='utf-8-sig')
    print("\n" + "="*70)
    print(f"✅ 合成完成！结果已保存至: {save_path}")
    print("现在可以进行 Step 6 (逆变换) 了。")
    print("="*70)
    
    return df_final