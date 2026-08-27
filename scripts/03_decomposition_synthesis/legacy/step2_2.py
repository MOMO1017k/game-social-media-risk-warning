import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import json
import os
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 辅助函数：波动率重采样
# ============================================================
def generate_bootstrap_volatility(hist_vol, n_samples, block_size=96):
    """
    终极保底方案：历史波动率分块重采样
    block_size=96 约等于 1 天 (15min * 4 * 24)，保留日内波动模式
    """
    # 移除无效值
    valid_vol = hist_vol[~np.isnan(hist_vol) & ~np.isinf(hist_vol)]
    if len(valid_vol) < block_size:
        return np.ones(n_samples) # 数据太少，退化为常数波动率
        
    generated_vol = []
    current_len = 0
    
    rng = np.random.default_rng()
    
    while current_len < n_samples:
        # 随机选一个起始点
        start_idx = rng.integers(0, len(valid_vol) - block_size)
        # 取出一个块
        block = valid_vol[start_idx : start_idx + block_size]
        generated_vol.append(block)
        current_len += len(block)
        
    # 拼接并裁剪
    final_vol = np.concatenate(generated_vol)[:n_samples]
    
    # 加上微小的随机扰动 (0.9 ~ 1.1)，防止完全重复
    noise = rng.uniform(0.9, 1.1, size=n_samples)
    return final_vol * noise


# ============================================================
# 引入 Tanh 饱和函数
# ============================================================
def smooth_tanh_saturation(x, limit):
    """
    双曲正切饱和：平滑地将数据限制在 [-limit, limit] 范围内
    - x 小的时候，近似线性 (y ≈ x)
    - x 大的时候，平滑逼近 limit
    """
    # 保护 limit 防止除零或无效
    limit = max(float(limit), 1e-6)
    
    # 核心公式: limit * tanh(x / limit)
    # tanh 的定义域是 (-inf, inf)，值域是 (-1, 1)
    # 所以结果严格限制在 (-limit, limit) 之间
    return limit * np.tanh(x / limit)
# ============================================================
# Core Logic (From Code B): Bootstrap Burn-in
# ============================================================
def _make_burnin_z_from_series(z_t: np.ndarray, burnin: int, rng: np.random.Generator) -> np.ndarray:
    """
    用 bootstrap 的方式从 z_t 中抽样生成 burn-in 段，保证与 z_t 同分布。
    """
    z_t = np.asarray(z_t, dtype=float)
    if burnin <= 0:
        return z_t

    z_t_clean = z_t[~np.isnan(z_t) & ~np.isinf(z_t)]
    # 如果有效数据太少，退化为标准正态
    if z_t_clean.size < 10:
        z_pre = rng.standard_normal(burnin)
    else:
        # 有放回抽样
        idx = rng.integers(0, z_t_clean.size, size=burnin)
        z_pre = z_t_clean[idx]

    return np.concatenate([z_pre, z_t], axis=0)

# ============================================================
# Core Logic (From Code B): Volatility Generators
# ============================================================
def generate_gjr_garch_volatility_from_z(
    z_full: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    unconditional_var: float | None = None,
    burnin: int = 500
):
    """
    GJR-GARCH(1,1,1) 生成器 (Code B逻辑：带持久性限制与极值保护)
    """
    z_full = np.asarray(z_full, dtype=float)
    total = z_full.size
    
    # 鲁棒性修正
    omega = float(max(omega, 1e-12))
    alpha = float(max(alpha, 0.0))
    gamma = float(gamma)  
    beta  = float(max(beta, 0.0))

    # 持续性检查与缩放 (Code B: 阈值 0.995)
    persistence = alpha + beta + 0.5 * gamma
    if persistence >= 0.995:
        scale = 0.995 / max(persistence, 1e-12)
        alpha *= scale
        beta  *= scale
        gamma *= scale
        persistence = alpha + beta + 0.5 * gamma

    # 无条件方差初始化
    if unconditional_var is None:
        denom = max(1.0 - persistence, 1e-6)
        unconditional_var = float(max(omega / denom, 1e-10))
    else:
        unconditional_var = float(max(unconditional_var, 1e-10))

    sigma_sq = np.zeros(total, dtype=float)
    eps      = np.zeros(total, dtype=float)

    # 初始状态
    sigma_sq[0] = unconditional_var
    eps[0] = np.sqrt(sigma_sq[0]) * z_full[0]

    for t in range(1, total):
        indicator = 1.0 if eps[t-1] < 0 else 0.0
        
        # GJR 核心公式
        sigma_sq[t] = (omega
                       + alpha * (eps[t-1]**2)
                       + gamma * (eps[t-1]**2) * indicator
                       + beta  * sigma_sq[t-1])
        
        # 极值保护 (Code B: Log Soft Clip的前置保护)
        sigma_sq[t] = min(max(sigma_sq[t], 1e-12), 1e8) 
        
        eps[t] = np.sqrt(sigma_sq[t]) * z_full[t]

    sigma = np.sqrt(sigma_sq)
    return sigma[burnin:], sigma_sq[burnin:]


def generate_garch_volatility_from_z(
    z_full: np.ndarray,
    omega: float,
    alpha_list,
    beta_list,
    unconditional_var: float | None = None,
    burnin: int = 500
):
    """
    通用 GARCH(p,q) 生成器 (Code B逻辑：支持高阶)
    """
    z_full = np.asarray(z_full, dtype=float)
    total = z_full.size

    omega = float(max(omega, 1e-12))
    alpha_list = [max(a, 0.0) for a in alpha_list]
    beta_list = [max(b, 0.0) for b in beta_list]

    p = len(alpha_list)
    q = len(beta_list)
    max_lag = max(p, q, 1)

    persistence = sum(alpha_list) + sum(beta_list)

    # 持续性限制 (Code B: 阈值 0.995)
    if persistence >= 0.995:
        scale = 0.995 / max(persistence, 1e-12)
        alpha_list = [a * scale for a in alpha_list]
        beta_list = [b * scale for b in beta_list]
        persistence = sum(alpha_list) + sum(beta_list)

    if unconditional_var is None:
        unconditional_var = omega / max(1.0 - persistence, 1e-6)
    unconditional_var = float(max(unconditional_var, 1e-10))

    sigma_sq = np.zeros(total, dtype=float)
    eps_sq = np.zeros(total, dtype=float)

    # 初始化
    sigma_sq[:max_lag] = unconditional_var
    eps_sq[:max_lag] = sigma_sq[:max_lag] * (z_full[:max_lag] ** 2)

    # 递推
    for t in range(max_lag, total):
        arch_term = 0.0
        for i in range(p):
            arch_term += alpha_list[i] * eps_sq[t - 1 - i]

        garch_term = 0.0
        for j in range(q):
            garch_term += beta_list[j] * sigma_sq[t - 1 - j]

        sigma_sq[t] = omega + arch_term + garch_term
        sigma_sq[t] = min(max(sigma_sq[t], 1e-12), 1e8)

        eps_sq[t] = sigma_sq[t] * (z_full[t] ** 2)

    sigma = np.sqrt(sigma_sq)
    return sigma[burnin:], sigma_sq[burnin:]


def generate_egarch_volatility_from_z(
    z_full: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    unconditional_var: float | None = None,
    burnin: int = 500
):
    """
    EGARCH(1,1) 生成器 (Code B逻辑：Beta限制与初始值保护)
    """
    z_full = np.asarray(z_full, dtype=float)
    total = z_full.size

    # Code B: Beta 保护
    if abs(beta) >= 0.995:
        beta = np.sign(beta) * 0.995

    # 计算经验 E|z|
    z_clean = z_full[~np.isnan(z_full) & ~np.isinf(z_full)]
    E_abs_z = float(np.mean(np.abs(z_clean))) if z_clean.size > 10 else np.sqrt(2 / np.pi)

    # 初始化 (E[log σ^2] = ω / (1 - β))
    if unconditional_var is None:
        unc_log_var = omega / (1.0 - beta)
    else:
        unc_log_var = np.log(max(unconditional_var, 1e-10))
    
    # Code B: 初始值限幅 [-10, 10]
    unc_log_var = np.clip(unc_log_var, -10, 10)

    log_sigma_sq = np.zeros(total, dtype=float)
    log_sigma_sq[0] = unc_log_var

    for t in range(1, total):
        z_prev = z_full[t-1]
        g_z = alpha * (np.abs(z_prev) - E_abs_z) + gamma * z_prev
        
        log_sigma_sq[t] = omega + beta * log_sigma_sq[t-1] + g_z
        
        # 强力裁剪防止溢出
        log_sigma_sq[t] = np.clip(log_sigma_sq[t], -20, 20)

    sigma_sq = np.exp(log_sigma_sq)
    sigma = np.sqrt(sigma_sq)
    return sigma[burnin:], sigma_sq[burnin:]


# ============================================================
# Core Logic (From Code B): Parameter Extraction
# ============================================================
def extract_garch_params_dynamic(garch_results, feature_name):
    """
    Code B+ 修改版：提取参数并应用 0.985 强阻尼 (Simulation Regularization)
    """
    if feature_name not in garch_results:
        return None

    result = garch_results[feature_name]
    params = result['params']
    model_name = str(result['model_name'])
    # params = model_result.params

    params_dict = {
        'model_type': model_name,
        'omega': float(params.get('omega', 1e-6)),
    }

    # ================= 提取原始参数 =================
    
    # --- 分支 A: EGARCH ---
    if 'EGARCH' in model_name:
        params_dict['model_class'] = 'EGARCH'
        params_dict['alpha'] = float(params.get('alpha[1]', 0.1))
        params_dict['gamma'] = float(params.get('gamma[1]', 0.0))
        params_dict['beta']  = float(params.get('beta[1]', 0.9))

    # --- 分支 B: GJR-GARCH ---
    elif 'GJR' in model_name or 'GJR-GARCH' in model_name:
        params_dict['model_class'] = 'GJR'
        params_dict['alpha'] = float(params.get('alpha[1]', 0.05))
        params_dict['gamma'] = float(params.get('gamma[1]', 0.0)) 
        params_dict['beta']  = float(params.get('beta[1]', 0.90))

    # --- 分支 C: 标准 GARCH (动态 p,q) ---
    else:
        params_dict['model_class'] = 'GARCH'
        alpha_list = []
        for i in range(1, 5): 
            key = f'alpha[{i}]'
            if key in params.keys(): alpha_list.append(float(params[key]))
            else: break
        
        beta_list = []
        for i in range(1, 5):
            key = f'beta[{i}]'
            if key in params.keys(): beta_list.append(float(params[key]))
            else: break

        if not alpha_list: alpha_list = [0.05]
        if not beta_list: beta_list = [0.90]

        params_dict['alpha_list'] = alpha_list
        params_dict['beta_list'] = beta_list

    # ================= 修改重点：强制阻尼逻辑 (0.985) =================
    # 针对 11 个月长周期，降低持续性上限，防止发散
    MAX_PERSISTENCE = 0.985
    
    current_persistence = 0.0
    cls = params_dict['model_class']

    if cls == 'EGARCH':
        # EGARCH 的持续性主要由 Beta 决定
        current_persistence = abs(params_dict['beta'])
        if current_persistence > MAX_PERSISTENCE:
            # print(f"   🔧 {feature_name} (EGARCH): Beta {current_persistence:.4f} -> {MAX_PERSISTENCE}")
            damping = MAX_PERSISTENCE / current_persistence
            params_dict['beta'] *= damping

    elif cls == 'GJR':
        # GJR Persistence = alpha + beta + 0.5 * gamma
        current_persistence = params_dict['alpha'] + params_dict['beta'] + 0.5 * params_dict['gamma']
        if current_persistence > MAX_PERSISTENCE:
            # print(f"   🔧 {feature_name} (GJR): Persistence {current_persistence:.4f} -> {MAX_PERSISTENCE}")
            damping = MAX_PERSISTENCE / current_persistence
            params_dict['alpha'] *= damping
            params_dict['beta']  *= damping
            params_dict['gamma'] *= damping

    else: # GARCH
        # GARCH Persistence = sum(alpha) + sum(beta)
        sum_alpha = sum(params_dict['alpha_list'])
        sum_beta = sum(params_dict['beta_list'])
        current_persistence = sum_alpha + sum_beta
        
        if current_persistence > MAX_PERSISTENCE:
            # print(f"   🔧 {feature_name} (GARCH): Persistence {current_persistence:.4f} -> {MAX_PERSISTENCE}")
            damping = MAX_PERSISTENCE / current_persistence
            params_dict['alpha_list'] = [a * damping for a in params_dict['alpha_list']]
            params_dict['beta_list']  = [b * damping for b in params_dict['beta_list']]

    # ================= 提取统计量 =================
    cond_vol = np.asarray(result['analysis']['cond_vol'], dtype=float)
    valid_vol = cond_vol[~np.isnan(cond_vol) & ~np.isinf(cond_vol)]
    
    if valid_vol.size > 10:
        params_dict['unconditional_var'] = float(np.mean(valid_vol ** 2))
        params_dict['vol_max'] = float(np.max(valid_vol))
        params_dict['vol_min'] = float(np.min(valid_vol))
        params_dict['vol_std'] = float(np.std(valid_vol))
    else:
        params_dict['unconditional_var'] = 1.0
        params_dict['vol_max'] = 5.0
        params_dict['vol_min'] = 0.1
        params_dict['vol_std'] = 1.0

    if 'scaling' in result:
        params_dict['scaling_mean'] = float(result['scaling']['mean'])
        params_dict['scaling_std']  = float(result['scaling']['std'])
    if 'params' in result:
        params_dict['garch_mu'] = float(result['params'].get('mu', 0.0))
   

    return params_dict
# ============================================================
# 质量验证
# ============================================================
def verify_step2_quality(df_noise, df_final, df_volatility, feature_list):
    """
    Step 2 专项质量验证：生成分特征的详细质量报告
    """
    print("\n【Step 2 深度质量检测 (Per-Feature)】")
    
    quality_records = []
    
    # 计算相关性矩阵差异 (全局指标，无法拆分到单特征，但可计算该特征的平均相关性偏差)
    corr_noise = df_noise.corr()
    corr_final = df_final.corr()
    corr_diff = np.abs(corr_final - corr_noise)
    # 计算每个特征与其他特征的相关性偏差的平均值
    mean_corr_diff_per_feature = corr_diff.mean(axis=1)

    for feature in feature_list:
        # 1. 提取序列
        s_noise = df_noise[feature]
        s_final = df_final[feature]
        
        # 2. 厚尾性 (Kurtosis Change)
        k_noise = s_noise.kurtosis()
        k_final = s_final.kurtosis()
        k_change = k_final - k_noise
        
        # 3. 极值合理性 (Max Sigma Check)
        # 检查相对于标准差的倍数
        std_val = s_final.std()
        max_abs_val = s_final.abs().max()
        sigma_ratio = max_abs_val / std_val if std_val > 1e-9 else 0.0
        
        # 4. 波动率聚集性 (Ljung-Box on squared residuals)
        # 这里集成 Ljung-Box 检验，不再在主函数里单独做
        s_final_clean = s_final[np.isfinite(s_final)]
        if len(s_final_clean) > 20:
            lb_res = acorr_ljungbox(s_final_clean ** 2, lags=[10], return_df=True)
            lb_p = float(lb_res['lb_pvalue'].iloc[0])
        else:
            lb_p = 1.0
            
        # 记录
        quality_records.append({
            'feature': feature,
            'kurtosis_noise': k_noise,
            'kurtosis_final': k_final,
            'kurtosis_increase': k_change,
            'max_sigma_ratio': sigma_ratio,
            'corr_diff_mean': mean_corr_diff_per_feature[feature],
            'lb_pvalue': lb_p,
            'is_heavy_tail': k_change > 0.1,  # 阈值可调
            'is_vol_clustered': lb_p < 0.05,
            'is_extreme_safe': sigma_ratio < 30.0 # 阈值可调
        })

    # 转为 DataFrame
    df_quality = pd.DataFrame(quality_records).set_index('feature')
    
    # 打印一些摘要
    print(f"   📊 峰度增加特征占比: {(df_quality['kurtosis_increase'] > 0).mean():.1%}")
    print(f"   📊 显著波动率聚集占比: {(df_quality['lb_pvalue'] < 0.05).mean():.1%}")
    print(f"   📏 最大 Sigma 倍数: {df_quality['max_sigma_ratio'].max():.2f}")
    
    # 找出潜在异常特征
    risky_features = df_quality[~df_quality['is_extreme_safe']].index.tolist()
    if risky_features:
        print(f"   ⚠️ 极值风险特征 ({len(risky_features)}个): {risky_features[:5]}...")
    else:
        print("   ✅ 所有特征极值均在安全范围内")

    return df_quality
# ============================================================
# Engineering Shell (From Code A): Main Orchestrator
# ============================================================
def inject_volatility_step2(
    df_synthetic_noise,
    garch_results,
    feature_list,
    risky_features=None,
    burnin=500,
    seed=42,
    output_dir=None
):
    """
    第二步：注入波动率
    架构：Code A (日志/报告/绘图/IO)
    内核：Code B (生成器/软截断/零均值还原)
    """
    os.makedirs(output_dir, exist_ok=True)
    rng_global = np.random.default_rng(seed)

    print("═" * 70)
    print("📌 第二步：注入 GARCH/EGARCH 条件波动率 (Soft Clipping Mode)")
    print("═" * 70)

    n_samples = len(df_synthetic_noise)
    time_index = df_synthetic_noise.index

    # 1. 提取参数 (使用 Code B 的动态提取器)
    print("\n【1. 提取模型参数】")
    all_params = {}
    model_summary = {'GARCH': 0, 'GJR': 0, 'EGARCH': 0, 'fallback': 0}

    for feature in feature_list:
        params = extract_garch_params_dynamic(garch_results, feature)
        # print(f"   {feature:<25} -> ", params if params else "Fallback")
        if params is None:
            params = {
                'model_class': 'EGARCH', 'model_type': 'Default',
                'omega': 0.01, 'alpha': 0.1, 'gamma': -0.05, 'beta': 0.9,
                'vol_max': 5.0, 'scaling_std': 1.0
            }
            model_summary['fallback'] += 1
        else:
            model_summary[params['model_class']] += 1
            
        all_params[feature] = params
        
        # 简单日志
        # if len(all_params) <= 5 or len(all_params) > len(feature_list) - 2:
        #     print(f"   {feature:<25} -> {params['model_class']}")

    print(f"\n   模型汇总: {model_summary}")

    # 2. 生成与注入
    print("\n【2. 生成波动率 & 注入 (Code B Kernel)】")
    volatility_dict = {}
    df_with_volatility = df_synthetic_noise.copy()

    for i, feature in enumerate(feature_list):
        params = all_params[feature]
        
        persistence = params.get('alpha', 0) + params.get('beta', 0) + 0.5 * params.get('gamma', 0)
        is_risky_feature = False

        # 强制指定某些字段走保底逻辑
        
        if any(k in feature for k in risky_features) or persistence > 0.98:
            is_risky_feature = True
            print(f"   ⚠️ {feature} 被标记为潜在风险特征 (Persistence: {persistence:.4f})")
            
        # if feature == "comp_ratio_post":is_risky_feature = False
        #准备历史波动率数据
        hist_vol_raw = np.array(garch_results[feature]['analysis']['cond_vol'])

        sigma = None # 初始化
        # === 分支 1: 启动保底方案 (Bootstrap) ===
        if is_risky_feature and len(hist_vol_raw) > 100:
            print(f"   🛡️ {feature} 启用历史重采样保底 (避免爆炸/零膨胀)")# 测试20260208
            sigma = generate_bootstrap_volatility(hist_vol_raw, n_samples)
            # print(f"   2️⃣ sigma: 均值={np.mean(sigma):.6f}, 最大={np.max(sigma):.6f}")# 测试20260208


        # === 分支 2: 正常 GARCH 递归 ===
        else:
            # print(f"    {feature} 启用正常GARCH生成器 (持久性: {persistence:.4f})")# 测试20260208

            # --- A. Bootstrap Burn-in (Code B) ---
            z_t = df_synthetic_noise[feature].to_numpy(dtype=float)

            # print(f"   1️⃣ z_t: 均值={np.mean(z_t):.6f}, 标准差={np.std(z_t):.6f}")# 测试20260208
            rng = np.random.default_rng(seed + i * 1000)
            z_full = _make_burnin_z_from_series(z_t, burnin, rng)

            # --- B. 调用生成器 (Code B) ---
            cls = params['model_class']
            args = {
                'z_full': z_full, 'omega': params['omega'], 
                'unconditional_var': params.get('unconditional_var'), 'burnin': burnin
            }
            
            if cls == 'GJR':
                sigma, _ = generate_gjr_garch_volatility_from_z(
                    alpha=params['alpha'], gamma=params['gamma'], beta=params['beta'], **args
                )
            elif cls == 'EGARCH':
                sigma, _ = generate_egarch_volatility_from_z(
                    alpha=params['alpha'], gamma=params['gamma'], beta=params['beta'], **args
                )
            else: # GARCH
                sigma, _ = generate_garch_volatility_from_z(
                    alpha_list=params['alpha_list'], beta_list=params['beta_list'], **args
                )

        # print(f"   2️⃣ sigma: 均值={np.mean(sigma):.6f}, 最大={np.max(sigma):.6f}")# 测试20260208
        # --- C. 截断策略: Soft Clipping (Code B 核心优势) ---
        # 逻辑：允许极值存在，但通过对数压缩防止其无限发散
        hist_max = params.get('vol_max', 10.0)
        limit_upper = max(hist_max * 4.0, 10.0) # 允许比历史最大值大4倍
        # 应用 Tanh 饱和
        sigma = smooth_tanh_saturation(sigma, limit_upper)
        volatility_dict[feature] = sigma
        
        if np.max(sigma) > limit_upper:
            # 软截断公式: limit + log(1 + (x - limit))
            mask = sigma > limit_upper
            # print(f"      🔧 {feature} 触发软截断 (Max: {np.max(sigma):.1f} -> {limit_upper:.1f}+log)")
            sigma[mask] = limit_upper + np.log1p(sigma[mask] - limit_upper)
        
        # print(f"   2️⃣ 截断sigma: 均值={np.mean(sigma):.6f}, 最大={np.max(sigma):.6f}")# 测试20260208

        volatility_dict[feature] = sigma

        # --- D. 注入与还原 (Code B 逻辑: 加 scaling_mean) ---
        # ε_scaled = σ_scaled * z_t
        eps_scaled = sigma * z_t
        
        # print(f"   3️⃣ eps_scaled (sigma*z): 均值={np.mean(eps_scaled):.6f}") # 测试20260208
        garch_mu = params.get('garch_mu', 0.0)
        scaled_with_mu = garch_mu + eps_scaled
        # print(f"   4️⃣ garch_mu = {garch_mu:.6f}")# 测试20260208
        # print(f"   5️⃣ scaled_with_mu (mu + sigma*z): 均值={np.mean(scaled_with_mu):.6f}")# 测试20260208
        # 还原: raw = scaled * std / 100 
        # 需要加入均值，因为数据并非零均值
        scaling_std = params.get('scaling_std', 1.0)
        scaling_mean = params.get('scaling_mean', 0.0)

        # eps_raw = eps_scaled * scaling_std / 100.0 + scaling_mean
        eps_raw = scaled_with_mu * scaling_std / 100.0 + scaling_mean
        # eps_raw = scaled_with_mu
        if 'redundancy' in feature:
            current_std = np.std(eps_scaled) + 1e-9
            correction_factor = 100.0 / current_std
            eps_scaled_final = eps_scaled * correction_factor
            eps_raw = (eps_scaled_final * scaling_std / 100.0) + scaling_mean
            # print(f"   注意: {feature} 是冗余特征，未加均值还原")# 测试20260208


        # print(f"   6️⃣ 最终 eps_raw:")# 测试20260208
        # print(f"      均值 = {np.mean(eps_raw):.6f}")
        # print(f"      标准差 = {np.std(eps_raw):.6f}")
        # print(f"      期望均值 ≈ {scaling_mean:.6f} (scaling_mean)")
        
        df_with_volatility[feature] = eps_raw

    # 结果转 DataFrame
    volatility_df = pd.DataFrame(volatility_dict, index=time_index)

    # 3. 质量验证 (保留 Code A 的 Ljung-Box 报告)
    print("\n【3. 质量验证】")
    # 调用新的验证函数，获取分特征报告
    df_quality_report = verify_step2_quality(
        df_synthetic_noise, 
        df_with_volatility, 
        volatility_df, 
        feature_list
    )
    
    # 保存分特征验证报告
    quality_csv_path = os.path.join(output_dir, 'quality_check_per_feature.csv')
    df_quality_report.to_csv(quality_csv_path, encoding='utf-8-sig')
    print(f"   📝 分特征质量报告已保存: {quality_csv_path}")

    # 计算全局汇总指标 (用于 JSON 报告)
    global_metrics = {
        'avg_kurtosis_increase': float(df_quality_report['kurtosis_increase'].mean()),
        'avg_corr_diff': float(df_quality_report['corr_diff_mean'].mean()),
        'max_sigma_ratio': float(df_quality_report['max_sigma_ratio'].max()),
        'vol_clustered_ratio': float((df_quality_report['lb_pvalue'] < 0.05).mean()),
        'heavy_tail_ratio': float((df_quality_report['kurtosis_increase'] > 0).mean())
    }

    # 4. 可视化 (修改版：逐个特征绘图)
    print("\n【4. 生成可视化 (逐特征)】")
    
    # 创建专门存放图片的子文件夹，防止文件过多杂乱
    plots_dir = output_dir
    # os.makedirs(plots_dir, exist_ok=True)
    
    for idx, feature in enumerate(feature_list):
        # 为每个特征创建一个独立的画布
        # 设置 figsize 为长条形，适合左右对比
        fig = plt.figure(figsize=(16, 6))
        
        # --- 左图: 波动率 (1行2列的第1个) ---
        ax1 = fig.add_subplot(1, 2, 1)
        vol_data = volatility_df[feature].iloc[:2000] # 只画前2000个点，避免过密
        ax1.plot(vol_data.index, vol_data.values, linewidth=1.0, color='red', alpha=0.8)
        ax1.set_title(f'{feature}\nConditional Volatility (Scaled)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        # 优化时间轴显示
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # --- 右图: 注入效果对比 (1行2列的第2个) ---
        ax2 = fig.add_subplot(1, 2, 2)
        
        # 画原始白噪声 (背景)
        noise_data = df_synthetic_noise[feature].iloc[:2000]
        ax2.plot(noise_data.index, noise_data.values, linewidth=0.5, alpha=0.4, color='gray', label='z_t (White Noise)')
        
        # 画注入后的残差 (前景)
        injected_data = df_with_volatility[feature].iloc[:2000]
        ax2.plot(injected_data.index, injected_data.values, linewidth=0.8, alpha=0.9, color='steelblue', label='ε_t (GARCH Raw)')
        
        ax2.set_title(f'{feature}\nVolatility Injection Effect', fontsize=12)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.tight_layout()
        
        # 处理文件名中可能存在的非法字符 (如 / 或空格)
        safe_fname = str(feature).replace('/', '_').replace('\\', '_').replace(' ', '_')
        save_path = os.path.join(plots_dir, f'step2_volatility_{safe_fname}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # 关键：画完一个关闭一个，释放内存
        
        if (idx + 1) % 5 == 0:
            print(f"   ...已保存 {idx + 1}/{len(feature_list)} 张图片")

    print(f"   ✅ 所有特征对比图已保存至: {plots_dir}")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/step2_volatility_injection.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 保存结果 (Code A 的 IO 逻辑)
    print("\n【5. 保存结果】")
    df_with_volatility.to_csv(f'{output_dir}/synthetic_with_volatility.csv', encoding='utf-8-sig')
    volatility_df.to_csv(f'{output_dir}/synthetic_volatility_scaled.csv', encoding='utf-8-sig')
    
    try:
        df_with_volatility.to_parquet(f'{output_dir}/synthetic_with_volatility.parquet')
        volatility_df.to_parquet(f'{output_dir}/synthetic_volatility_scaled.parquet')
        print("   ✅ Parquet 保存成功")
    except:
        pass

    # 生成 JSON 报告
    injection_report = {
        'metadata': {'n_samples': int(n_samples), 'burnin': int(burnin), 'seed': int(seed)},
        'strategy': {
            'clipping': 'Tanh Saturation (Strict Bound)', # 更新描述
            'damping': 'Max Persistence 0.985',           # 更新描述
            'rescaling': 'Zero-Mean (No Trend Addition)'},
        'model_summary': model_summary,
        'global_quality_metrics': global_metrics, # 使用新的汇总指标
        # 'clustering_check': {k: bool(v) for k, v in clustering_results.items()}
    }
    with open(f'{output_dir}/volatility_injection_report.json', 'w', encoding='utf-8') as f:
        json.dump(injection_report, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 完成。数据已保存至: {output_dir}")
    return df_with_volatility, volatility_df, injection_report


def run_step2_inject_volatility_v4(
    df_synthetic_noise,
    garch_results,
    feature_list,
    risky_features,
    burnin=500,
    seed=42,
    output_dir=None
):
    """
    流程封装Wrapper
    """
    return inject_volatility_step2(
        df_synthetic_noise, garch_results, feature_list, risky_features, burnin, seed, output_dir
    )