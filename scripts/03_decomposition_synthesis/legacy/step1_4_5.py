import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 核心工具函数 ====================

def check_autocorrelation(series, lags=10):
    """
    计算序列的短期自相关性指标
    返回: ACF数组, Ljung-Box p值(滞后5阶)
    """
    # 移除无效值
    valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_series) < lags * 2:
        return None, None

    # 计算 ACF
    acf_values = acf(valid_series, nlags=lags, fft=True)
    
    # 计算 Ljung-Box (白噪声检验), 取滞后 5 阶的 p 值
    # p < 0.05 表示拒绝白噪声假设（即存在显著自相关）
    lb_res = acorr_ljungbox(valid_series, lags=[5], return_df=True)
    lb_pvalue = lb_res['lb_pvalue'].iloc[0]
    
    return acf_values, lb_pvalue

def fit_ar_process(series, max_lags=5):
    """
    拟合 AR 模型提取自相关成分
    """
    valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 数据长度检查，避免报错
    if len(valid_series) <= max_lags + 10:
        print(f"   ⚠️ 数据过短 ({len(valid_series)}), 跳过 AR 拟合")
        return None

    # 使用 AIC 自动选择最佳 lag (1 ~ max_lags)
    model = AutoReg(valid_series, lags=max_lags, trend='c', old_names=False).fit()
    
    # 提取参数供合成使用
    # model.params 包含 'const' 和 'y.L1', 'y.L2' 等
    ar_params = model.params.to_dict()
    
    # 获取拟合值 (自相关部分)
    fitted_values = model.fittedvalues
    
    # 计算新残差 (原始 - 拟合)
    # 对齐索引
    new_resid = series - fitted_values
    
    # 填补因 Lag 产生的头部 NaN (保持长度一致，填0不影响后续 GARCH)
    new_resid = new_resid.fillna(0)
    
    return {
        'model': model,
        'params': ar_params,
        'fitted': fitted_values,
        'new_resid': new_resid,
        'lags_used': model.ar_lags
    }

# ==================== 可视化函数 ====================

def plot_acf_comparison(orig_series, new_series, feature_name, 
                       orig_stats, new_stats, output_dir):
    """
    绘制处理前后的 ACF 对比图
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    orig_acf, orig_p = orig_stats
    new_acf, new_p = new_stats
    
    if orig_acf is None or new_acf is None:
        return

    lags = len(orig_acf) - 1
    x = np.arange(lags + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图 1: 处理前 ACF
    axes[0].bar(x, orig_acf, color='steelblue', alpha=0.7, width=0.6)
    axes[0].axhline(0, color='black', linewidth=0.8)
    # 95% 置信区间近似线
    conf = 1.96 / np.sqrt(len(orig_series.dropna()))
    axes[0].axhline(conf, color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(-conf, color='red', linestyle='--', alpha=0.5)
    
    title_text = f"处理前: {feature_name}\nLjung-Box p={orig_p:.4f} "
    title_text += "(显著自相关" if orig_p < 0.05 else "(白噪声"
    title_text += ")"
    axes[0].set_title(title_text, fontsize=11)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[0].set_ylim(-0.5, 1.0)
    
    # 图 2: 处理后 ACF
    axes[1].bar(x, new_acf, color='green', alpha=0.7, width=0.6)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].axhline(conf, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(-conf, color='red', linestyle='--', alpha=0.5)
    
    title_text_new = f"处理后: AR过滤残差\nLjung-Box p={new_p:.4f} "
    title_text_new += "(Failed: 显著自相关" if new_p < 0.05 else "(Passed: 白噪声"
    title_text_new += ")"
    axes[1].set_title(title_text_new, fontsize=11)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylim(-0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature_name}_ACF对比.png', dpi=100)
    plt.close()

# ==================== 主流程 ====================

def run_step1_4_5_ar_filter(df_step4_output, feature_list, output_dir):
    """
    Step 1_4.5 主函数: 短期自相关过滤
    
    Parameters:
    -----------
    df_step4_output : pd.DataFrame
        Step 1_4 的输出数据 (包含 feature_event_adj_resid)
    feature_list : list
        特征列表
        
    Returns:
    --------
    df_output : pd.DataFrame
        包含新残差的数据 (增加 feature_ar_resid 列)
    ar_results : dict
        包含 AR 参数 (用于 Step 9 合成)
    summary_df : pd.DataFrame
        评估汇总表
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("📌 步骤1.4.5：短期自相关 (AR) 检测与提取")
    print("="*70)
    
    df_output = df_step4_output.copy()
    ar_results = {}
    summary_data = []
    
    for feature in feature_list:
        # 1. 确定输入列 (优先使用 Step 1_4 的去事件残差)
        col_name = f'{feature}_event_adj_resid'
        if col_name not in df_output.columns:
            # 回退到 Prophet 残差
            col_name = f'{feature}_prophet_resid'
            if col_name not in df_output.columns:
                print(f"⚠️ 跳过 {feature}: 未找到输入残差列")
                continue
                
        series = df_output[col_name]
        print(f"\n🔧 处理特征: {feature} (输入: {col_name})")
        
        # 2. 检测初始自相关性
        orig_acf, orig_lb_p = check_autocorrelation(series, lags=10)
        
        if orig_acf is None:
            print("   ⚠️ 数据不足，无法检测")
            continue
            
        print(f"   检测前: Lag-1 ACF = {orig_acf[1]:.4f}, LB p-value = {orig_lb_p:.4e}")
        
        # 3. 执行 AR 提取
        # 即使 p > 0.05 (已经是白噪声)，也可以尝试提取微弱的 AR 信息供合成使用
        # 这里默认 max_lags=5，依据 AIC 自动选择
        ar_res = fit_ar_process(series, max_lags=5)
        
        if ar_res is None:
            continue
            
        # 4. 保存结果
        new_resid = ar_res['new_resid']
        params = ar_res['params']
        lags_used = ar_res['lags_used']
        
        # 写入 DataFrame
        out_col = f'{feature}_ar_resid'
        df_output[out_col] = new_resid
        
        # 保存参数供合成
        ar_results[feature] = {
            'params': params,
            'lags': lags_used,
            'input_col': col_name
        }
        
        # 5. 再次检测 (评估改善)
        new_acf, new_lb_p = check_autocorrelation(new_resid, lags=10)
        
        print(f"   提取后: Lag-1 ACF = {new_acf[1]:.4f}, LB p-value = {new_lb_p:.4e}")
        print(f"   使用 Lags: {lags_used}, 参数数: {len(params)}")
        
        # 6. 绘图
        plot_acf_comparison(series, new_resid, feature, 
                           (orig_acf, orig_lb_p), (new_acf, new_lb_p), output_dir)
        
        # 7. 记录汇总
        summary_data.append({
            '特征': feature,
            '原始_Lag1_ACF': f"{orig_acf[1]:.4f}",
            '原始_LB_P值': f"{orig_lb_p:.4e}",
            '原始_状态': '❌显著相关' if orig_lb_p < 0.05 else '✅白噪声',
            '使用Lags': str(lags_used),
            '提取后_Lag1_ACF': f"{new_acf[1]:.4f}",
            '提取后_LB_P值': f"{new_lb_p:.4e}",
            '提取后_状态': '❌显著相关' if new_lb_p < 0.05 else '✅白噪声',
            '改善程度': f"{abs(orig_acf[1]) - abs(new_acf[1]):.4f}"
        })
        
    # 生成汇总表
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/AR过滤效果汇总.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print(f"✅ Step 1.4.5 完成，结果保存在: {output_dir}")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    return df_output, ar_results, summary_df