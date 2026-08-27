import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. 基础工具函数
# ============================================================

def generate_decay_wave(length, impact_coef, decay_type, decay_param):
    """
    生成衰减波形
    """
    if length <= 0: return np.array([])
    t = np.arange(length)
    
    decay_param = float(decay_param)
    if decay_param < 1e-6: decay_param = 1e-6

    if decay_type == 'exponential':
        # exp(-t/tau)
        wave = np.exp(-t / decay_param)
    elif decay_type == 'power':
        # (t+1)^(-alpha)
        wave = np.power(t + 1, -decay_param)
    else:
        # 默认
        wave = np.exp(-t / 8.0)
        
    return impact_coef * wave

def load_decomposition_params(json_path):
    """从 JSON 加载分解参数"""
    if not os.path.exists(json_path):
        print(f"⚠️ 警告: 参数文件 {json_path} 不存在")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================
# 2. 过滤逻辑
# ============================================================

def check_modeling_criteria(feature, combo_name, coef, p_val, model_selection=None):
    """
    【双重过滤】判断是否对该特征的该组合事件进行建模
    """
    if model_selection is None:
        model_selection = {}
        
    # 参数解包
    threshold_p = model_selection.get('threshold_p_value', 0.1) 
    threshold_coef = model_selection.get('threshold_min_coef', 0.001) # Sigma单位
    force_positive_features = model_selection.get('force_positive_features', 
                                                ['volume', 'post', 'comment', 'view', 'share'])
    
    # 0. Crisis 硬过滤 (双重保险)
    if 'crisis' in combo_name:
        return False, "Crisis event excluded"

    # 1. 幅度过滤
    if abs(coef) < threshold_coef:
        return False, f"Impact too small ({abs(coef):.4f} < {threshold_coef})"
        
    # 2. 显著性过滤
    if p_val > threshold_p:
        return False, f"Not significant (p={p_val:.4f} > {threshold_p})"
        
    # 3. 业务逻辑过滤 (方向性)
    is_volume_metric = any(kw in feature for kw in force_positive_features)
    if is_volume_metric:
        if coef < 0:
            return False, f"Negative impact on volume metric ({coef:.4f})"

    return True, "Pass"

# ============================================================
# 3. 质量验证函数 (生成 quality_check_event_impact)
# ============================================================

def verify_step4_quality(df_pre, df_post, df_events, feature_list, output_dir):
    """
    Step 4 专项验证：检查冲击覆盖率和极值风险
    输出: quality_check_event_impact.csv
    """
    print("\n【Step 4 深度质量检测 (Post-Synthesis Validation)】")
    
    results = []
    check_window = 16 # 4小时窗口
    
    # 计算纯冲击成分
    df_shock_only = df_post - df_pre
    
    # 筛选有效时间范围内的事件
    sim_start = df_post.index.min()
    sim_end = df_post.index.max()
    valid_events = df_events[(df_events['timestamp'] >= sim_start) & 
                             (df_events['timestamp'] <= sim_end)]
    
    for feature in feature_list:
        if feature not in df_shock_only.columns: continue

        shock_series = df_shock_only[feature]
        max_impact = shock_series.max()
        min_impact = shock_series.min()
        
        # 状态判断
        if abs(max_impact) < 1e-9 and abs(min_impact) < 1e-9:
            results.append({
                'feature': feature, 'status': 'No Impact',
                'max_impact': 0, 'min_impact': 0,
                'event_response_rate': 0, 'max_sigma_jump': 0, 'is_safe': True
            })
            continue
            
        # 计算事件响应覆盖率
        active_events = 0
        total_events = 0
        
        # 采样检查事件响应 (加速)
        check_events = valid_events['timestamp']
        if len(check_events) > 500:
            check_events = check_events.sample(500, random_state=42)

        for t in check_events:
            try:
                locs = shock_series.index.get_indexer([t], method='nearest', tolerance=pd.Timedelta('30min'))
                if locs[0] == -1: continue
                
                start_loc = locs[0]
                end_loc = min(start_loc + check_window, len(shock_series))
                local_shock = shock_series.iloc[start_loc:end_loc]
                
                # 如果窗口内有明显非零值
                if local_shock.abs().max() > 1e-5:
                    active_events += 1
                total_events += 1
            except: continue
            
        response_rate = active_events / total_events if total_events > 0 else 0.0
        
        # 极值风险检查
        pre_std = df_pre[feature].std()
        if pre_std < 1e-6: pre_std = 1e-6
        max_sigma_jump = max(abs(max_impact), abs(min_impact)) / pre_std
        
        results.append({
            'feature': feature,
            'status': 'Active',
            'max_impact': round(max_impact, 4),
            'min_impact': round(min_impact, 4),
            'event_response_rate': round(response_rate, 2),
            'max_sigma_jump': round(max_sigma_jump, 2),
            'is_safe': max_sigma_jump < 20.0 # 20倍标准差阈值
        })
        
    # 保存报告
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        save_path = os.path.join(output_dir, 'quality_check_event_impact.csv')
        df_res.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        active_df = df_res[df_res['status']=='Active']
        print(f"   📊 平均事件响应率: {active_df['event_response_rate'].mean():.1%}")
        
        risky = active_df[~active_df['is_safe']]
        if not risky.empty:
            print(f"   ⚠️ 警告: {len(risky)} 个特征冲击幅度过大 (>20 std):")
            print(risky[['feature', 'max_sigma_jump']].head(3))
        else:
            print("   ✅ 极值检查通过 (Max Sigma Jump < 20)")

# ============================================================
# 4. 核心合成主函数 (含包络与埋点)
# ============================================================

def apply_event_shocks_step4_envelope(
    df_step3_input,       # Step 3 输出 (AR 残差)
    df_official_schedule, # 官方排期表
    decomp_params,        # 分解参数
    feature_list,
    model_selection=None, 
    output_dir='./result/step4_event_envelope'
):
    print("═" * 70)
    print("📌 第四步：官方运营事件冲击 (Envelope Strategy & Deep Debug)")
    print("═" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    

    feature_impact_adjustments = {
        # 🔴 红色警报：严重过冲 (Tail Diff > 200%)
        # 行动：大幅削减，建议打 5-6 折
        'comp_ratio_post': 0.5, 

        # 🟡 黄色警报：评论类普遍过冲 (Tail Diff 50%~80%)
        # 行动：因为是累加效应导致的，建议打 7 折
        'total_volume_comment': 0.7,
        'gini_comment': 0.7,
        'senti_symbol_comment': 0.7,
        'comp_ratio_comment': 0.7,
        'total_short_comment': 0.7,
        'total_long_comment': 0.7,
        'neg_ratio_comment': 0.7,
        'semantic_shift_comment': 0.7,

        # 🟢 绿色通行：表现良好 (Tail Diff < 20%)
        # 行动：保持原样 (默认 1.0)，显式写出来是为了方便管理
        'total_volume_post': 1.0,
        'gini_post': 1.0,
        'retweet_ratio_post': 1.0,
        'neg_ratio_post': 1.0,
        'vis_abs_redundancy_post': 1.0,
        'vis_concentration_post': 1.0
    }

    # --- 0. 配置缺省值 ---
    if model_selection is None:
        model_selection = {
            'threshold_p_value': 0.1,
            'threshold_min_coef': 0.001,
            'simulation_strength': 1.0 # 强度调节因子
        }
    
    # --- 1. 预处理排期 (强制过滤 Crisis) ---
    df_sch = df_official_schedule.copy()
    df_sch['timestamp'] = pd.to_datetime(df_sch['timestamp'])
    
    # 强制过滤 category == 'crisis'
    original_len = len(df_sch)
    df_sch = df_sch[df_sch['category'] != 'crisis']
    print(f"   🛡️ 已过滤 {original_len - len(df_sch)} 个 Crisis 事件")
        
    df_sch['combo'] = df_sch['official_author'] + '_' + df_sch['category']
    
    # 筛选时间范围
    sim_start = df_step3_input.index.min()
    sim_end = df_step3_input.index.max()
    df_sch = df_sch[(df_sch['timestamp'] >= sim_start) & (df_sch['timestamp'] <= sim_end)]
    
    print(f"   📅 有效模拟事件数: {len(df_sch)}")
    
    df_output = df_step3_input.copy()
    
    # 全局日志容器
    all_filter_logs = []
    all_modeling_stats = []
    
    # --- 2. 逐特征处理 ---
    for feature in feature_list:
        if feature not in decomp_params:
            continue
            
        feat_adj_factor = feature_impact_adjustments.get(feature, 1.0)
        
        # 打印一下日志，确保你知道它生效了
        if feat_adj_factor != 1.0:
            print(f"   🔧 微调生效: {feature} 冲击强度 x {feat_adj_factor}")
        feat_data = decomp_params[feature]
        coeffs_map = feat_data.get('coefficients', {})
        
        # 初始化双通道包络层 & 计数层
        n_points = len(df_step3_input)
        total_shock_accumulator = np.zeros(n_points)
        pos_envelope = np.zeros(n_points)  # 正向最大值层
        neg_envelope = np.zeros(n_points)  # 负向最小值层 (存负值)
        event_counts = np.zeros(n_points, dtype=int) # 堆积计数器
        
        # 特征级调试日志
        wave_debug_log = []
        stacking_log = []
        shock_magnitudes = []
        
        # 预计算组合决策 (生成 filter_logs)
        combo_decisions = {}
        for combo, param in coeffs_map.items():
            coef = param['coef']
            p_val = param.get('pvalue', 1.0)
            
            should_model, reason = check_modeling_criteria(
                feature, combo, coef, p_val, model_selection
            )
            
            combo_decisions[combo] = {'pass': should_model, 'param': param}
            
            if not should_model:
                all_filter_logs.append({
                    'feature': feature, 
                    'combo': combo, 
                    'coef': coef, 
                    'p_val': p_val, 
                    'reason': reason
                })

        # 计算基准波动率 (用于量纲还原)
        # [关键修正]: 分解系数是Sigma单位，合成需要还原为绝对值
        current_std = df_step3_input[feature].std()
        if current_std < 1e-6: current_std = 1e-6

        # --- 事件遍历 ---
        cnt_applied = 0
        
        for _, event in df_sch.iterrows():
            combo = event['combo']
            
            # 检查决策
            if combo not in combo_decisions or not combo_decisions[combo]['pass']:
                continue
                
            param = combo_decisions[combo]['param']
            
            # [量纲还原] Sigma Coef -> Absolute Impact
            raw_coef = param['coef']
            sim_strength = model_selection.get('simulation_strength', 1.0)
            # 核心公式：基础冲击 * 全局强度 * 特征专属微调
            actual_impact = raw_coef * current_std * sim_strength * feat_adj_factor
            
            # 波形参数
            d_type = param['decay_type']
            d_param = param['decay_param']
            
            # 定位时间
            try:
                locs = df_step3_input.index.get_indexer([event['timestamp']], method='nearest', tolerance=pd.Timedelta('30min'))
                if locs[0] == -1: continue
                idx_loc = locs[0]
            except: continue
            

            
            # ================= [新增] 参数锁死与保护 =================
            # 1. 修正衰减参数：防止指数衰减过慢
            # 对于指数分布，d_param 是 tau (时间尺度)，限制最大为 32 (约8小时半衰期)
            # 对于幂律分布，d_param 是 alpha (衰减速率)，通常 < 3，min(x, 32) 无副作用
            real_d_param = min(d_param, 32.0)
            
            # 2. 计算时长并应用硬截断
            if d_type == 'exponential':
                # 指数衰减：5倍 tau 约等于衰减到 0.6%
                duration = int(real_d_param * 5)
            else:
                # 幂律衰减：直接锁死 24 小时 (96 个点)
                # 幂律的长尾太长，通过计算衰减到 1% 的时间往往不靠谱
                duration = 96 
                
            # 3. 二次保护：绝对时长不超过 48 小时 (192点)
            duration = min(duration, 192)
            # =======================================================

            end_loc = min(idx_loc + duration, n_points)
            length = end_loc - idx_loc
            if length <= 0: continue
            
            # 生成波形
            wave = generate_decay_wave(length, actual_impact, d_type, d_param)
            
            # 如果是强制截断的幂律波形，对最后 4 个点做线性淡出，避免断崖
            if d_type == 'power' and length >= 10:
                fade_len = 4
                fade_factors = np.linspace(1, 0, fade_len)
                wave[-fade_len:] *= fade_factors
            
            # === [埋点 1] 波形末端检查 ===
            end_val_ratio = abs(wave[-1]) / (abs(actual_impact) + 1e-9)
            if end_val_ratio > 0.1: 
                wave_debug_log.append({
                    'time': str(event['timestamp']),
                    'end_ratio': round(end_val_ratio, 3)
                })

            # === [核心逻辑] 双通道包络 ===
            # 更新计数器
            event_counts[idx_loc:end_loc] += 1
            
            # 分通道取极值
            target_slice = slice(idx_loc, end_loc)
            slice_len = end_loc - idx_loc
            if len(wave) != slice_len:
                wave = wave[:slice_len]
            # [修复] 直接累加
            total_shock_accumulator[target_slice] += wave

            if actual_impact > 0:
                pos_envelope[target_slice] = np.maximum(pos_envelope[target_slice], wave)
            else:
                neg_envelope[target_slice] = np.minimum(neg_envelope[target_slice], wave)
                
            cnt_applied += 1
            shock_magnitudes.append(np.max(np.abs(wave)))
            
        # === [埋点 2] 堆积风险分析 ===
        max_overlap = np.max(event_counts) if len(event_counts) > 0 else 0
        
        # 合并双通道
        total_shock = pos_envelope + neg_envelope
        
        # 双通道注入
        # df_output[feature] = df_output[feature] + total_shock
        
        # 改用累加
        df_output[feature] = df_output[feature] + total_shock_accumulator
        
        # 记录 modeling_stats
        max_shock_val = np.max(np.abs(total_shock)) if len(total_shock) > 0 else 0
        all_modeling_stats.append({
            'feature': feature,
            'events_applied': cnt_applied,
            'max_overlap': max_overlap,
            'max_shock_val': max_shock_val,
            'avg_single_shock': np.mean(shock_magnitudes) if shock_magnitudes else 0,
            'wave_truncation_issues': len(wave_debug_log)
        })
        
        # 打印日志
        if cnt_applied > 0:
            print(f"   ⚡ {feature:<25}: 注入 {cnt_applied} 事件 | Max Overlap: {max_overlap} | Peak: {max_shock_val:.4f}")

    # --- 3. 保存所有输出文件 ---
    print("\n   💾 保存输出文件...")
    
    # 1. 最终数据
    df_output.to_csv(f'{output_dir}/step4_event_output.csv', encoding='utf-8-sig')
    
    # 2. 过滤日志 filter_logs_dropped
    if all_filter_logs:
        pd.DataFrame(all_filter_logs).to_csv(f'{output_dir}/filter_logs_dropped.csv', index=False, encoding='utf-8-sig')
        print(f"      📄 已生成 filter_logs_dropped.csv ({len(all_filter_logs)} 条记录)")
    
    # 3. 建模统计 modeling_stats
    if all_modeling_stats:
        pd.DataFrame(all_modeling_stats).to_csv(f'{output_dir}/modeling_stats.csv', index=False, encoding='utf-8-sig')
        print(f"      📄 已生成 modeling_stats.csv")

    # --- 4. 调用质量验证 ---
    # 生成 quality_check_event_impact.csv
    verify_step4_quality(df_step3_input, df_output, df_sch, feature_list, output_dir)
    print(f"      📄 已生成 quality_check_event_impact.csv")
    
    print(f"\n   ✅ Step 4 完成。结果已保存至 {output_dir}")
    return df_output

