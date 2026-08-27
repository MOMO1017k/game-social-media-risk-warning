import numpy as np
import pandas as pd
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import os
import json
import warnings

# 屏蔽可能的未来警告
warnings.filterwarnings('ignore')

def parse_ar_params(feature_name, feature_ar_data):
    """
    辅助函数：解析复杂格式的 AR 结果字典
    
    Args:
        feature_name: 特征名
        feature_ar_data: 字典，格式如 {'params': {...}, 'lags': [1,2], 'input_col': '...'}
    
    Returns:
        coeffs_array: 按顺序排列的 AR 系数数组 (缺失 lag 补 0)
        ar_order: 最大滞后阶数
    """
    # 1. 安全检查
    if not feature_ar_data or 'lags' not in feature_ar_data or not feature_ar_data['lags']:
        return np.array([]), 0
        
    params = feature_ar_data.get('params', {})
    lags = feature_ar_data.get('lags', [])
    input_col = feature_ar_data.get('input_col', '') # 获取原始列名用于拼凑 key
    
    # 2. 确定最大阶数 (处理稀疏滞后，例如只选了 lag 1 和 5)
    max_lag = max(lags)
    
    # 3. 初始化系数数组 (索引 i 对应 lag i+1)
    coeffs_array = np.zeros(max_lag)
    
    # 4. 填充系数
    # 尝试两种 key 格式：
    # 格式 A: "{input_col}.L{lag}" (Statsmodels 标准输出)
    # 格式 B: "L{lag}.{input_col}" (部分旧版本格式)
    for lag in lags:
        # 尝试构建可能的键名
        key_candidates = [
            f"{input_col}.L{lag}",
            f"L{lag}.{input_col}"
        ]
        
        found = False
        for key in key_candidates:
            if key in params:
                coeffs_array[lag-1] = params[key] # lag 1 存入 index 0
                found = True
                break
        
        if not found:
            print(f"   ⚠️ {feature_name}: 未找到 Lag {lag} 的系数 (Key 尝试: {key_candidates})")

    return coeffs_array, max_lag
from statsmodels.tsa.ar_model import AutoReg


def verify_step3_quality(df_input, df_output, ar_params_dict):
    """
    Step 3 专项验证：记忆性植入检查与参数回测
    """
    print("\n【Step 3 深度质量检测 (Parameter Recovery)】")
    
    results = []
    
    features = df_output.columns
    for feature in features:
        # ================= 改动点：适配复杂输入格式 =================
        # 使用之前定义的解析函数，将字典转换为稠密数组
        # 例如: {'L1': 0.5, 'L3': 0.2} -> [0.5, 0.0, 0.2]
        if feature in ar_params_dict:
            target_coeffs, max_lag = parse_ar_params(feature, ar_params_dict[feature])
        else:
            target_coeffs = np.array([])
            
        if len(target_coeffs) == 0:
            continue
        # ==========================================================
            
        # 1. 自相关性变化 (ACF Check)
        # 理论上 Step 3 的 ACF(1) 应该显著提升（对于正相关）或改变
        acf_in = df_input[feature].autocorr(lag=1)
        acf_out = df_output[feature].autocorr(lag=1)
        acf_change = acf_out - acf_in
        
        # 2. 参数回测 (Coefficient Recovery)
        # 用生成的数据反向训练一个 AR 模型，看能不能还原出参数
        # target_coeffs 是稠密数组 (包含0)，长度即为最大滞后阶数
        lags = len(target_coeffs)
        
        try:
            # 仅使用前 5000 个数据进行快速拟合，防止数据量过大变慢
            train_data = df_output[feature].values[:5000]
            
            # AutoReg fit: lags=int 意味着拟合 lag 1 到 lag k
            # 这与我们 target_coeffs 的结构 (稠密数组) 是一致的
            model = AutoReg(train_data, lags=lags, old_names=False)
            res = model.fit()
            
            # 提取拟合出的系数 (params[0]是const intercept, 后面才是phi)
            fitted_coeffs = res.params[1:] 
            
            # 计算系数误差 (MAE)
            if len(fitted_coeffs) == len(target_coeffs):
                coeff_error = np.mean(np.abs(fitted_coeffs - target_coeffs))
            else:
                coeff_error = 999.0
                
        except Exception as e:
            # print(f"Fit failed for {feature}: {e}")
            fitted_coeffs = []
            coeff_error = 999.0
            
        results.append({
            'feature': feature,
            'acf_input': acf_in,
            'acf_output': acf_out,
            'acf_delta': acf_change,
            'target_phi1': target_coeffs[0], # 目标第一阶系数
            'fitted_phi1': fitted_coeffs[0] if len(fitted_coeffs)>0 else 0,
            'coeff_recovery_mae': coeff_error
        })
        
    if not results:
        print("   ⚠️ 没有有效的 AR 特征进行验证")
        return pd.DataFrame()

    df_res = pd.DataFrame(results).set_index('feature')
    
    # --- 打印摘要 ---
    print(f"   📈 平均自相关提升: {df_res['acf_delta'].mean():.4f}")
    print(f"   🎯 系数还原误差 (MAE): {df_res['coeff_recovery_mae'].mean():.4f}")
    
    # 筛选还原失败的特征 (误差 > 0.1 视为偏差较大，对于合成数据通常允许宽松一点)
    bad_recovery = df_res[df_res['coeff_recovery_mae'] > 0.15]
    if not bad_recovery.empty:
        print(f"   ⚠️ 以下特征参数还原度较低 ({len(bad_recovery)}个):")
        print(bad_recovery[['target_phi1', 'fitted_phi1', 'coeff_recovery_mae']].head())
    else:
        print("   ✅ 所有特征的 AR 参数均成功还原 (Error < 0.15)")
        
    return df_res



def apply_ar_filter_step3(
    df_step2_input,   # 来自 Step 2 的输出 (df_with_volatility)
    ar_params_dict,   # 分解流程 Step 4 的 AR 结果字典
    feature_list,
    output_dir='./result/step3_ar'
):
    print("═" * 70)
    print("📌 第三步：叠加 AR(p) 短期线性自相关 (Inertia Injection)")
    print("═" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 容器
    df_ar_output = pd.DataFrame(index=df_step2_input.index)
    ar_info_log = {}
    
    # 统计计数
    stats_count = {'processed': 0, 'skipped': 0, 'unstable': 0}

    for feature in feature_list:
        input_series = df_step2_input[feature].values
        
        # 1. 获取并解析 AR 参数
        if feature in ar_params_dict:
            coeffs, ar_order = parse_ar_params(feature, ar_params_dict[feature])
        else:
            coeffs, ar_order = np.array([]), 0
        
        # 2. 检查系数有效性与稳定性
        is_stable = True
        
        if len(coeffs) == 0:
            # 无 AR 参数
            output_series = input_series
            stats_count['skipped'] += 1
        else:
            # 简单的平稳性检查 (充分非必要条件: sum(|phi|) < 1)
            # 对于合成数据，严格一点比较安全，防止长周期发散
            # 如果仅仅略大于1，可能导致数据缓慢飘走，建议在这里做个阻尼或者跳过
            sum_abs = np.sum(np.abs(coeffs))
            
            if sum_abs >= 1.0:
                print(f"   ⚠️ {feature}: AR 系数非平稳 (Sum Abs={sum_abs:.4f} >= 1.0)，已降级为无自相关")
                output_series = input_series
                is_stable = False
                stats_count['unstable'] += 1
                ar_order = 0 # 重置以便记录
                coeffs = []
            else:
                # 3. 应用 AR 滤波 (lfilter)
                # y[t] = phi_1*y[t-1] + phi_2*y[t-2] + ... + x[t]
                # 变换为: y[t] - phi_1*y[t-1] - ... = x[t]
                # Scipy lfilter: a[0]*y[t] + a[1]*y[t-1]... = b[0]*x[t]
                # 对应: a = [1, -phi_1, -phi_2, ...], b = [1]
                
                b = [1.0]
                a = np.concatenate(([1.0], -coeffs))
                
                # 注意：此处输入的是 Step 2 的带波动率残差 (Innovation)
                # lfilter 会将其作为白噪声输入，生成具有自相关的序列
                output_series = lfilter(b, a, input_series)
                stats_count['processed'] += 1

        df_ar_output[feature] = output_series
        
        # 记录日志 (转换 numpy array 为 list 以便 JSON 序列化)
        ar_info_log[feature] = {
            'order': int(ar_order), 
            'coeffs': list(coeffs) if len(coeffs) > 0 else [],
            'stable': bool(is_stable)
        }

    # 4. 验证 (ACF 检查)
    print("\n   📈 自相关性 (Lag-1) 变化检查 (Top 5 Features):")
    print(f"   {'Feature':<30} {'Step2(Raw)':<12} {'Step3(AR)':<12} {'Change'}")
    print("   " + "-" * 65)
    
    count = 0
    for feat in feature_list:
        if feat not in ar_info_log or ar_info_log[feat]['order'] == 0:
            continue
        
        acf2 = df_step2_input[feat].autocorr(lag=1)
        acf3 = df_ar_output[feat].autocorr(lag=1)
        print(f"   {feat:<30} {acf2:.4f}       {acf3:.4f}       {acf3-acf2:+.4f}")
        
        count += 1
        if count >= 5: break
        
    print(f"\n   📊 处理统计: 成功叠加 {stats_count['processed']}, 跳过/无参 {stats_count['skipped']}, 非平稳剔除 {stats_count['unstable']}")

    # 5. 可视化对比 (生成一张图，对比前后的 ACF 或 时序)
    plot_ar_effect(df_step2_input, df_ar_output, feature_list, output_dir)
    
    # 6. 质量验证
    df_quality_report = verify_step3_quality(
        df_step2_input, 
        df_ar_output, 
        ar_params_dict # 传入原始字典，函数内部会调用 parse_ar_params
    )
    if not df_quality_report.empty:
        df_quality_report.to_csv(f'{output_dir}/quality_check_ar_recovery.csv')
        
        # 计算汇总指标
        recovery_metrics = {
            'avg_acf_delta': float(df_quality_report['acf_delta'].mean()),
            'avg_coeff_mae': float(df_quality_report['coeff_recovery_mae'].mean()),
            'failed_recovery_count': int((df_quality_report['coeff_recovery_mae'] > 0.15).sum())
        }
    else:
        recovery_metrics = {}
        
    # 7. 保存
    df_ar_output.to_csv(f'{output_dir}/step3_ar_output.csv', encoding='utf-8-sig')
    try:
        df_ar_output.to_parquet(f'{output_dir}/step3_ar_output.parquet')
    except:
        pass
    
    with open(f'{output_dir}/step3_ar_params.json', 'w', encoding='utf-8') as f:
        json.dump(ar_info_log, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Step 3 完成。结果已保存至 {output_dir}")
    return df_ar_output

def plot_ar_effect(df_in, df_out, feature_list, output_dir):
    """
    绘制 AR 效果对比图 (选择前 3 个有 AR 参数的特征)
    """
    # 筛选出真正应用了 AR 的特征
    valid_features = []
    for f in feature_list:
        if not df_in[f].equals(df_out[f]):
            valid_features.append(f)
    
    if not valid_features:
        return

    n_plot = min(len(valid_features), 3)
    fig, axes = plt.subplots(n_plot, 2, figsize=(15, 4 * n_plot))
    if n_plot == 1: axes = [axes] # 统一维度

    for i in range(n_plot):
        feat = valid_features[i]
        
        # 时序图 (局部)
        ax_ts = axes[i][0] if n_plot > 1 else axes[0]
        ax_ts.plot(df_in[feat].iloc[:200], label='Step2 (Uncorrelated)', alpha=0.5, color='gray', lw=1)
        ax_ts.plot(df_out[feat].iloc[:200], label='Step3 (AR Injected)', alpha=0.8, color='blue', lw=1)
        ax_ts.set_title(f"{feat} - Time Series (First 200 pts)")
        ax_ts.legend()
        
        # ACF 图 (自相关图)
        ax_acf = axes[i][1] if n_plot > 1 else axes[1]
        
        # 手动计算简单的 ACF 用于绘图
        lags = range(11)
        acf_in = [df_in[feat].autocorr(lag=l) for l in lags]
        acf_out = [df_out[feat].autocorr(lag=l) for l in lags]
        
        ax_acf.plot(lags, acf_in, 'o--', label='Step2 ACF', color='gray')
        ax_acf.plot(lags, acf_out, 'o-', label='Step3 ACF', color='red')
        ax_acf.axhline(0, color='black', lw=0.5)
        ax_acf.set_title(f"{feat} - Autocorrelation Function")
        ax_acf.set_xlabel("Lag")
        ax_acf.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/step3_ar_check.png')
    plt.close()

# 示例调用 (假设环境已准备好)
# df_step3 = apply_ar_filter_step3(df_step2, ar_results, feature_list)