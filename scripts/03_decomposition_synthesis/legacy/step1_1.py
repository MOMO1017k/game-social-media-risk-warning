import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import nbinom, poisson, beta as beta_dist, kstest, chi2
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def get_feature_type(FEATURE_CONFIG, feature_name):
    """获取特征类型"""
    for ftype, features in FEATURE_CONFIG.items():
        if feature_name in features:
            return ftype
    return 'continuous'


# ==================== 分布检验函数 ====================

def fit_negative_binomial(data):
    """
    拟合负二项分布，返回参数(r, p)和拟合优度
    使用最大似然估计
    """
    data = data[data >= 0].astype(int)
    if len(data) == 0:
        return None, None, None, None
    
    mean_val = data.mean()
    var_val = data.var()
    
    # 矩估计初始值
    if var_val <= mean_val:
        # 方差小于等于均值，负二项不适用
        return None, None, None, "方差<=均值，不适合负二项分布"
    
    r_init = mean_val ** 2 / (var_val - mean_val)
    p_init = mean_val / var_val
    
    # 约束优化
    def neg_log_likelihood(params):
        r, p = params
        if r <= 0 or p <= 0 or p >= 1:
            return 1e10
        try:
            ll = np.sum(nbinom.logpmf(data, r, p))
            return -ll
        except:
            return 1e10
    
    try:
        result = minimize(neg_log_likelihood, [r_init, p_init], 
                         method='L-BFGS-B',
                         bounds=[(0.01, 1000), (0.001, 0.999)])
        r_fit, p_fit = result.x
        
        # KS检验
        ks_stat, ks_p = kstest(data, lambda x: nbinom.cdf(x, r_fit, p_fit))
        
        # 计算AIC
        k = 2  # 参数个数
        ll = -result.fun
        aic = 2 * k - 2 * ll
        
        return r_fit, p_fit, {'ks_stat': ks_stat, 'ks_p': ks_p, 'aic': aic, 'll': ll}, None
    except Exception as e:
        return None, None, None, str(e)


def fit_poisson(data):
    """
    拟合泊松分布，返回参数lambda和拟合优度
    """
    data = data[data >= 0].astype(int)
    if len(data) == 0:
        return None, None, None
    
    lambda_fit = data.mean()
    
    # KS检验
    ks_stat, ks_p = kstest(data, lambda x: poisson.cdf(x, lambda_fit))
    
    # 计算AIC
    ll = np.sum(poisson.logpmf(data, lambda_fit))
    aic = 2 * 1 - 2 * ll  # 1个参数
    
    # 离散系数 (判断是否过度离散)
    dispersion = data.var() / data.mean() if data.mean() > 0 else np.inf
    
    return lambda_fit, {'ks_stat': ks_stat, 'ks_p': ks_p, 'aic': aic, 'll': ll, 
                        'dispersion': dispersion}, None


def fit_beta(data):
    """
    拟合Beta分布，返回参数(a, b)和拟合优度
    """
    # 处理边界值
    data_clean = data[(data > 0) & (data < 1)]
    
    if len(data_clean) < 10:
        # 如果有效数据太少，尝试轻微收缩
        data_clean = np.clip(data, 0.001, 0.999)
    
    if len(data_clean) == 0:
        return None, None, None, "无有效数据"
    
    try:
        # MLE拟合
        a_fit, b_fit, loc, scale = beta_dist.fit(data_clean, floc=0, fscale=1)
        
        # KS检验
        ks_stat, ks_p = kstest(data_clean, lambda x: beta_dist.cdf(x, a_fit, b_fit))
        
        # 计算AIC
        ll = np.sum(beta_dist.logpdf(data_clean, a_fit, b_fit))
        aic = 2 * 2 - 2 * ll  # 2个参数
        
        return a_fit, b_fit, {'ks_stat': ks_stat, 'ks_p': ks_p, 'aic': aic, 'll': ll}, None
    except Exception as e:
        return None, None, None, str(e)


def fit_gmm(data, n_components=2):
    """
    拟合高斯混合模型，返回参数和拟合优度
    """
    data_clean = data[~np.isnan(data) & ~np.isinf(data)]
    
    if len(data_clean) < 30:
        return None, None, "数据不足"
    
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
        gmm.fit(data_clean.reshape(-1, 1))
        
        # BIC/AIC
        bic = gmm.bic(data_clean.reshape(-1, 1))
        aic = gmm.aic(data_clean.reshape(-1, 1))
        
        params = {
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.flatten().tolist(),
            'stds': np.sqrt(gmm.covariances_.flatten()).tolist(),
            'aic': aic,
            'bic': bic
        }
        
        return gmm, params, None
    except Exception as e:
        return None, None, str(e)


# ==================== 综合分布检验 ====================

def comprehensive_distribution_test(df, feature_name, FEATURE_CONFIG, output_dir):
    """
    综合分布检验：根据特征类型检验多种分布
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if feature_name not in df.columns:
        print(f"❌ 特征 {feature_name} 不存在")
        return None
    
    # 清洗数据
    series = df[feature_name].replace([np.inf, -np.inf], np.nan).dropna()
    data = series.values
    
    if len(data) < 30:
        print(f"⚠️ {feature_name}: 数据量不足 ({len(data)})")
        return None
    
    ftype = get_feature_type(FEATURE_CONFIG, feature_name)
    
    results = {
        'feature': feature_name,
        'type': ftype,
        'n_samples': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'zero_ratio': (data == 0).mean()
    }

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{feature_name} 分布检验', fontsize=20, fontweight='bold')
    
    # ========== 0. 原始数据分布 ==========
    axes[0, 0].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].set_title(f'原始分布\nN={len(data)}, μ={data.mean():.3f}, σ={data.std():.3f}')
    axes[0, 0].set_xlabel('值')
    axes[0, 0].set_ylabel('密度')
    
    # ========== 1. 泊松分布检验 ==========
    if ftype == 'count':
        lambda_pois, pois_metrics, pois_err = fit_poisson(data)
        
        if lambda_pois is not None:
            results['poisson'] = {
                'lambda': lambda_pois,
                **pois_metrics,
                'passed': pois_metrics['ks_p'] > 0.05
            }
            
            # 绑定绘图
            x_range = np.arange(0, min(int(np.percentile(data, 99)) + 5, 100))
            pois_pmf = poisson.pmf(x_range, lambda_pois)
            
            # 实际频率
            actual_freq = np.array([np.mean(data == x) for x in x_range])
            
            axes[0, 1].bar(x_range - 0.2, actual_freq, width=0.4, alpha=0.7, 
                          label='实际', color='steelblue')
            axes[0, 1].bar(x_range + 0.2, pois_pmf, width=0.4, alpha=0.7, 
                          label=f'Poisson(λ={lambda_pois:.2f})', color='orange')
            axes[0, 1].set_title(f'泊松分布\nKS p={pois_metrics["ks_p"]:.3e}, 离散系数={pois_metrics["dispersion"]:.2f}\n{"✅ 通过" if pois_metrics["ks_p"] > 0.05 else "❌ 不通过"}')
            axes[0, 1].legend()
            axes[0, 1].set_xlabel('计数值')
        else:
            axes[0, 1].text(0.5, 0.5, '泊松分布检验\n不适用', ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, '泊松分布检验\n(仅适用于计数型)', ha='center', va='center', fontsize=12)
        axes[0, 1].axis('off')
    
    # ========== 2. 负二项分布检验 ==========
    if ftype == 'count':
        r_nb, p_nb, nb_metrics, nb_err = fit_negative_binomial(data)
        
        if r_nb is not None:
            results['negative_binomial'] = {
                'r': r_nb,
                'p': p_nb,
                **nb_metrics,
                'passed': nb_metrics['ks_p'] > 0.05
            }
            
            # 绘图
            x_range = np.arange(0, min(int(np.percentile(data, 99)) + 5, 100))
            nb_pmf = nbinom.pmf(x_range, r_nb, p_nb)
            actual_freq = np.array([np.mean(data == x) for x in x_range])
            
            axes[0, 2].bar(x_range - 0.2, actual_freq, width=0.4, alpha=0.7, 
                          label='实际', color='steelblue')
            axes[0, 2].bar(x_range + 0.2, nb_pmf, width=0.4, alpha=0.7, 
                          label=f'NegBin(r={r_nb:.2f}, p={p_nb:.3f})', color='green')
            axes[0, 2].set_title(f'负二项分布\nKS p={nb_metrics["ks_p"]:.3e}\n{"✅ 通过" if nb_metrics["ks_p"] > 0.05 else "❌ 不通过"}')
            axes[0, 2].legend()
        else:
            axes[0, 2].text(0.5, 0.5, f'负二项分布检验\n{nb_err}', ha='center', va='center', fontsize=12)
            axes[0, 2].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, '负二项分布检验\n(仅适用于计数型)', ha='center', va='center', fontsize=12)
        axes[0, 2].axis('off')
    
    # ========== 3. Beta分布检验 ==========
    if ftype == 'ratio' or (data.min() >= 0 and data.max() <= 1):
        a_beta, b_beta, beta_metrics, beta_err = fit_beta(data)
        
        if a_beta is not None:
            results['beta'] = {
                'a': a_beta,
                'b': b_beta,
                **beta_metrics,
                'passed': beta_metrics['ks_p'] > 0.05
            }
            
            # 绘图
            x_range = np.linspace(0.001, 0.999, 200)
            beta_pdf = beta_dist.pdf(x_range, a_beta, b_beta)
            
            axes[1, 0].hist(data, bins=50, density=True, alpha=0.5, color='steelblue', label='实际')
            axes[1, 0].plot(x_range, beta_pdf, 'r-', linewidth=2, 
                           label=f'Beta(a={a_beta:.2f}, b={b_beta:.2f})')
            axes[1, 0].set_title(f'Beta分布\nKS p={beta_metrics["ks_p"]:.3e}\n{"✅ 通过" if beta_metrics["ks_p"] > 0.05 else "❌ 不通过"}')
            axes[1, 0].legend()
            axes[1, 0].set_xlim(0, 1)
        else:
            axes[1, 0].text(0.5, 0.5, f'Beta分布检验\n{beta_err}', ha='center', va='center', fontsize=12)
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'Beta分布检验\n(数据不在[0,1]区间)', ha='center', va='center', fontsize=12)
        axes[1, 0].axis('off')
    
    # ========== 4. GMM混合分布检验 ==========
    gmm_model, gmm_params, gmm_err = fit_gmm(data, n_components=2)
    
    if gmm_model is not None:
        results['gmm'] = gmm_params
        
        # 绘图
        x_range = np.linspace(data.min(), data.max(), 200)
        gmm_pdf = np.exp(gmm_model.score_samples(x_range.reshape(-1, 1)))
        
        axes[1, 1].hist(data, bins=50, density=True, alpha=0.5, color='steelblue', label='实际')
        axes[1, 1].plot(x_range, gmm_pdf, 'r-', linewidth=2, label='GMM(K=2)')
        
        # 标注各成分
        for i, (w, m, s) in enumerate(zip(gmm_params['weights'], 
                                          gmm_params['means'], 
                                          gmm_params['stds'])):
            comp_pdf = w * stats.norm.pdf(x_range, m, s)
            axes[1, 1].plot(x_range, comp_pdf, '--', linewidth=1.5, 
                           label=f'成分{i+1}: w={w:.2f}, μ={m:.2f}')
        
        axes[1, 1].set_title(f'GMM混合分布 (K=2)\nAIC={gmm_params["aic"]:.1f}, BIC={gmm_params["bic"]:.1f}')
        axes[1, 1].legend(fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, f'GMM检验\n{gmm_err}', ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    # ========== 5. Q-Q图 ==========
    stats.probplot(data, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot (vs Normal)')
    
    # ========== 6. 最优分布判定 ==========
    best_dist = None
    best_aic = np.inf
    
    if 'poisson' in results and results['poisson']['passed']:
        if results['poisson']['aic'] < best_aic:
            best_aic = results['poisson']['aic']
            best_dist = 'Poisson'
    
    if 'negative_binomial' in results and results['negative_binomial']['passed']:
        if results['negative_binomial']['aic'] < best_aic:
            best_aic = results['negative_binomial']['aic']
            best_dist = 'Negative Binomial'
    
    if 'beta' in results and results['beta']['passed']:
        if results['beta']['aic'] < best_aic:
            best_aic = results['beta']['aic']
            best_dist = 'Beta'
    
    if 'gmm' in results:
        if results['gmm']['aic'] < best_aic:
            best_aic = results['gmm']['aic']
            best_dist = 'GMM'
    
    results['best_distribution'] = best_dist
    results['best_aic'] = best_aic
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature_name}_分布检验.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print(f"📊 {feature_name} 分布检验结果")
    print(f"{'='*60}")
    print(f"  类型: {ftype}")
    print(f"  样本量: {len(data)}")
    print(f"  均值: {data.mean():.4f}, 标准差: {data.std():.4f}")
    print(f"  零值比例: {results['zero_ratio']:.2%}")
    
    if 'poisson' in results:
        print(f"\n  泊松分布: λ={results['poisson']['lambda']:.3f}")
        print(f"    KS p值: {results['poisson']['ks_p']:.3e} {'✅' if results['poisson']['passed'] else '❌'}")
        print(f"    离散系数: {results['poisson']['dispersion']:.3f}")
    
    if 'negative_binomial' in results:
        print(f"\n  负二项分布: r={results['negative_binomial']['r']:.3f}, p={results['negative_binomial']['p']:.4f}")
        print(f"    KS p值: {results['negative_binomial']['ks_p']:.3e} {'✅' if results['negative_binomial']['passed'] else '❌'}")
    
    if 'beta' in results:
        print(f"\n  Beta分布: a={results['beta']['a']:.3f}, b={results['beta']['b']:.3f}")
        print(f"    KS p值: {results['beta']['ks_p']:.3e} {'✅' if results['beta']['passed'] else '❌'}")
    
    if 'gmm' in results:
        print(f"\n  GMM混合分布:")
        print(f"    权重: {results['gmm']['weights']}")
        print(f"    均值: {results['gmm']['means']}")
    
    print(f"\n  🏆 最优分布: {best_dist} (AIC={best_aic:.1f})")
    
    return results


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import poisson, nbinom, beta as beta_dist, lognorm, norm
from sklearn.mixture import GaussianMixture
import matplotlib as mpl

# ── 全局中文设置 ──
mpl.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


def plot_distribution_analysis(data_1, feature_name, ftype, output_path=None):
    """
    单特征分布检验6子图 (用于图5.1、图5.2)
    
    Parameters
    ----------
    data : np.ndarray
        特征原始数据（已去除NaN/Inf）
    feature_name : str
        特征名称
    ftype : str
        特征类型：'count', 'ratio', 'continuous'
    output_path : str, optional
        保存路径
    """

    data = np.asarray(data_1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{feature_name} 分布检验', fontsize=20, fontweight='bold', y=1.02)
    
    data_clean = data[np.isfinite(data)]
    n = len(data_clean)
    zero_ratio = np.mean(data_clean == 0) * 100
    
    # ====== (0,0) 原始数据直方图 + 核密度 ======
    ax = axes[0, 0]
    ax.hist(data_clean, bins=60, density=True, alpha=0.6, 
            edgecolor='white', linewidth=0.5, color='#4E79A7', label='经验分布')
    
    # 叠加核密度估计
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data_clean, bw_method='scott')
        x_kde = np.linspace(data_clean.min(), np.percentile(data_clean, 99.5), 300)
        ax.plot(x_kde, kde(x_kde), color='#E15759', linewidth=2, label='KDE')
    except Exception:
        pass
    
    ax.set_title(f'经验分布\n'
                 f'$N$={n:,}, 零值率={zero_ratio:.1f}%\n'
                 f'$\\bar{{x}}$={data_clean.mean():.2f}, '
                 f'$\\gamma_1$={stats.skew(data_clean):.2f}',
                 fontsize=20)
    ax.set_xlabel('取值',fontsize=20)
    ax.set_ylabel('概率密度',fontsize=20)
    ax.legend(fontsize=20, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    
    # ====== (0,1) 泊松分布检验（仅计数型）======
    ax = axes[0, 1]
    if ftype == 'count':
        lambda_hat = data_clean.mean()
        D_disp = data_clean.var() / (lambda_hat + 1e-10)
        
        x_max = min(int(np.percentile(data_clean, 97)) + 3, 80)
        x_range = np.arange(0, x_max + 1)
        
        # 经验频率
        actual_freq = np.array([np.mean(data_clean == x) for x in x_range])
        pois_pmf = poisson.pmf(x_range, lambda_hat)
        
        width = 0.35
        ax.bar(x_range - width/2, actual_freq, width=width, alpha=0.7,
               color='#4E79A7', label='经验频率', edgecolor='white', linewidth=0.3)
        ax.bar(x_range + width/2, pois_pmf, width=width, alpha=0.7,
               color='#F28E2B', label=f'Poisson($\\lambda$={lambda_hat:.1f})')
        
        # K-S检验
        ks_stat, ks_p = stats.kstest(data_clean, 'poisson', args=(lambda_hat,))
        passed = '通过' if ks_p > 0.05 else '拒绝'
        
        ax.set_title(f'泊松分布拟合\n'
                     f'$D_{{\\mathrm{{disp}}}}$={D_disp:.2f}, '
                     f'$D_n$={ks_stat:.4f}\n'
                     f'K-S {passed} ($p$={ks_p:.2e})',
                     fontsize=12)
        ax.legend(fontsize=12, framealpha=0.8)
        ax.set_xlabel('计数值')
        ax.set_ylabel('频率')
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, '泊松分布检验\n仅适用于计数型特征',
                ha='center', va='center', fontsize=11, color='gray',
                transform=ax.transAxes)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # ====== (0,2) 对数正态分布拟合 ======
    ax = axes[0, 2]
    positive_data = data_clean[data_clean > 0]
    if len(positive_data) > 10:
        # 拟合对数正态
        shape, loc, scale = lognorm.fit(positive_data, floc=0)
        
        x_range = np.linspace(positive_data.min(), np.percentile(positive_data, 99), 300)
        ln_pdf = lognorm.pdf(x_range, shape, loc=loc, scale=scale)
        
        ax.hist(positive_data, bins=60, density=True, alpha=0.6,
                edgecolor='white', linewidth=0.5, color='#4E79A7', label='经验 ($x>0$)')
        ax.plot(x_range, ln_pdf, color='#59A14F', linewidth=2.5,
                label=f'LogNormal\n($\\mu_{{\\ln}}$={np.log(scale):.2f}, '
                      f'$\\sigma_{{\\ln}}$={shape:.2f})')
        
        ks_stat, ks_p = stats.kstest(positive_data, 'lognorm', args=(shape, loc, scale))
        aic_ln = -2 * np.sum(lognorm.logpdf(positive_data, shape, loc=loc, scale=scale)) + 2 * 2
        
        ax.set_title(f'对数正态分布拟合\n'
                     f'AIC={aic_ln:.1f}, $D_n$={ks_stat:.4f}\n'
                     f'K-S $p$={ks_p:.2e}',
                     fontsize=12)
        ax.legend(fontsize=12, framealpha=0.8)
    else:
        ax.text(0.5, 0.5, '正值样本不足\n无法拟合对数正态',
                ha='center', va='center', fontsize=11, color='gray',
                transform=ax.transAxes)
    ax.set_xlabel('取值', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.grid(True, alpha=0.2)
    
    # ====== (1,0) Beta分布检验（比率型）或负二项分布（计数型）======
    ax = axes[1, 0]
    if ftype == 'ratio':
        # Beta分布
        data_01 = data_clean[(data_clean > 0) & (data_clean < 1)]
        if len(data_01) > 10:
            a_hat, b_hat, bloc, bscale = beta_dist.fit(data_01, floc=0, fscale=1)
            x_range = np.linspace(0.001, 0.999, 300)
            beta_pdf = beta_dist.pdf(x_range, a_hat, b_hat)
            
            ax.hist(data_01, bins=50, density=True, alpha=0.6,
                    edgecolor='white', linewidth=0.5, color='#4E79A7',
                    label=f'经验 ($x \\in (0,1)$, n={len(data_01)})')
            ax.plot(x_range, beta_pdf, color='#B07AA1', linewidth=2.5,
                    label=f'Beta($\\alpha$={a_hat:.2f}, $\\beta$={b_hat:.2f})')
            
            ks_stat, ks_p = stats.kstest(data_01, 'beta', args=(a_hat, b_hat))
            ax.set_title(f'Beta分布拟合\n'
                         f'$D_n$={ks_stat:.4f}, K-S $p$={ks_p:.2e}',
                         fontsize=12)
            ax.legend(fontsize=12, framealpha=0.8)
            ax.set_xlim(-0.05, 1.05)
        else:
            ax.text(0.5, 0.5, 'Beta分布拟合\n(0,1)区间内样本不足',
                    ha='center', va='center', fontsize=11, color='gray',
                    transform=ax.transAxes)
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
    elif ftype == 'count':
        # 负二项分布
        try:
            # Method of moments for initial params
            mean_val = data_clean.mean()
            var_val = data_clean.var()
            if var_val > mean_val:
                p_hat = mean_val / var_val
                r_hat = mean_val * p_hat / (1 - p_hat)
                r_hat = max(r_hat, 0.1)
                p_hat = min(max(p_hat, 0.001), 0.999)
                
                x_max = min(int(np.percentile(data_clean, 97)) + 3, 80)
                x_range = np.arange(0, x_max + 1)
                actual_freq = np.array([np.mean(data_clean == x) for x in x_range])
                nb_pmf = nbinom.pmf(x_range, r_hat, p_hat)
                
                width = 0.35
                ax.bar(x_range - width/2, actual_freq, width=width, alpha=0.7,
                       color='#4E79A7', label='经验频率')
                ax.bar(x_range + width/2, nb_pmf, width=width, alpha=0.7,
                       color='#59A14F',
                       label=f'NegBin($r$={r_hat:.2f}, $p$={p_hat:.3f})')
                
                ks_stat, ks_p = stats.kstest(data_clean, 'nbinom', args=(r_hat, p_hat))
                ax.set_title(f'负二项分布拟合\n'
                             f'$D_n$={ks_stat:.4f}, K-S $p$={ks_p:.2e}',
                             fontsize=12)
                ax.legend(fontsize=12, framealpha=0.8)
            else:
                ax.text(0.5, 0.5, '方差≤均值\n负二项不适用',
                        ha='center', va='center', fontsize=11, color='gray',
                        transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'负二项拟合失败\n{str(e)[:30]}',
                    ha='center', va='center', fontsize=11, color='gray',
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, '不适用', ha='center', va='center',
                fontsize=11, color='gray', transform=ax.transAxes)
    ax.set_xlabel('取值')
    ax.set_ylabel('频率/密度')
    ax.grid(True, alpha=0.2)
    
    # ====== (1,1) GMM混合分布 (K=2) ======
    ax = axes[1, 1]
    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full',
                              n_init=5, random_state=42)
        gmm.fit(data_clean.reshape(-1, 1))
        
        x_range = np.linspace(data_clean.min(),
                              np.percentile(data_clean, 99.5), 300)
        log_prob = gmm.score_samples(x_range.reshape(-1, 1))
        gmm_pdf = np.exp(log_prob)
        
        ax.hist(data_clean, bins=60, density=True, alpha=0.5,
                edgecolor='white', linewidth=0.5, color='#4E79A7', label='经验分布')
        ax.plot(x_range, gmm_pdf, color='#E15759', linewidth=2.5, label='GMM ($K$=2)')
        
        # 各分量
        weights = gmm.weights_
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        
        colors_comp = ['#F28E2B', '#59A14F']
        for k in range(2):
            comp_pdf = weights[k] * norm.pdf(x_range, means[k], stds[k])
            ax.plot(x_range, comp_pdf, '--', linewidth=1.5, color=colors_comp[k],
                    label=f'分量{k+1}: $\\pi$={weights[k]:.2f}, '
                          f'$\\mu$={means[k]:.2f}')
        
        aic_gmm = gmm.aic(data_clean.reshape(-1, 1))
        bic_gmm = gmm.bic(data_clean.reshape(-1, 1))
        
        ax.set_title(f'高斯混合模型 ($K$=2)\n'
                     f'AIC={aic_gmm:.1f}, BIC={bic_gmm:.1f}',
                     fontsize=12)
        ax.legend(fontsize=12, framealpha=0.8, loc='upper right')
    except Exception as e:
        ax.text(0.5, 0.5, f'GMM拟合失败\n{str(e)[:30]}',
                ha='center', va='center', fontsize=11, color='gray',
                transform=ax.transAxes)
    ax.set_xlabel('取值', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.grid(True, alpha=0.2)
    
    # ====== (1,2) Q-Q图 ======
    ax = axes[1, 2]
    (osm, osr), (slope, intercept, r_val) = stats.probplot(data_clean, dist="norm")
    ax.scatter(osm, osr, s=8, alpha=0.4, color='#4E79A7', edgecolors='none')
    
    # 参考线
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'参考线 ($R^2$={r_val**2:.4f})')
    
    ax.set_title(f'正态Q-Q图\n$R^2$={r_val**2:.4f}', fontsize=12)
    ax.set_xlabel('理论分位数', fontsize=12)
    ax.set_ylabel('样本分位数', fontsize=12)
    ax.legend(fontsize=12, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'已保存: {output_path}')
    
    plt.show()
    plt.close()


# ==================== 批量检验所有特征 ====================

def run_step1_distribution_test(df_feature, feature_list=None, FEATURE_CONFIG = {}, output_dir = None):
    """
    步骤1：批量进行分布检验
    
    Parameters:
    -----------
    df_feature : pd.DataFrame, 特征数据
    feature_list : list, 要检验的特征列表，None则使用全部
    output_dir : str, 输出目录
    
    Returns:
    --------
    dict : 所有特征的检验结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if feature_list is None:
        feature_list = []
        for ftype, features in FEATURE_CONFIG.items():
            feature_list.extend(features)
    
    all_results = {}
    
    print("="*70)
    print("📌 步骤1：分布检验")
    print("="*70)
    
    for feature in feature_list:
        if feature in df_feature.columns:
            result = comprehensive_distribution_test(df_feature, feature, FEATURE_CONFIG, output_dir)
            if result:
                all_results[feature] = result
    
    # plot_distribution_analysis(df_feature['total_volume_post'].values, 'total_volume_post', 'count',
    #                 'fig5_1_total_volume_post.svg')
    # plot_distribution_analysis(df_feature['gini_post'].values, 'gini_post', 'ratio',
    #                         'fig5_2_gini_post.svg')
    for feature in feature_list:
        if feature in df_feature.columns:
            data = df_feature[feature].replace([np.inf, -np.inf], np.nan).dropna().values
            ftype = get_feature_type(FEATURE_CONFIG, feature)
            plot_distribution_analysis(data, feature, ftype, f'{output_dir}/{feature}_分布分析.svg')
    
    # 生成汇总表
    summary_data = []
    for feat, res in all_results.items():
        row = {
            '特征': feat,
            '类型': res['type'],
            '样本量': res['n_samples'],
            '零值比例': f"{res['zero_ratio']:.1%}",
            '最优分布': res.get('best_distribution', 'Unknown'),
            '最优AIC': res.get('best_aic', np.nan)
        }
        
        # 添加各分布检验结果
        if 'poisson' in res:
            row['泊松_D'] = f"{res['poisson']['ks_stat']:.4f}"
            row['泊松_p'] = f"{res['poisson']['ks_p']:.2e}"
            row['泊松_通过'] = '✅' if res['poisson']['passed'] else '❌'
        else:
            row['泊松_D'] = '-'
            row['泊松_p'] = '-'
            row['泊松_通过'] = '-'
        
        if 'negative_binomial' in res:
            row['负二项_D'] = f"{res['negative_binomial']['ks_stat']:.4f}"
            row['负二项_p'] = f"{res['negative_binomial']['ks_p']:.2e}"
            row['负二项_通过'] = '✅' if res['negative_binomial']['passed'] else '❌'
        else:
            row['负二项_D'] = '-'
            row['负二项_p'] = '-'
            row['负二项_通过'] = '-'
        
        if 'beta' in res:
            row['Beta_D'] = f"{res['beta']['ks_stat']:.4f}"
            row['Beta_p'] = f"{res['beta']['ks_p']:.2e}"
            row['Beta_通过'] = '✅' if res['beta']['passed'] else '❌'
        else:
            row['Beta_D'] = '-'
            row['Beta_p'] = '-'
            row['Beta_通过'] = '-'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/分布检验汇总.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("✅ 步骤1完成！")
    print(f"📁 结果保存至: {output_dir}")
    print("="*70)
    
    # 打印汇总表
    print("\n📋 分布检验汇总表:")
    print(summary_df.to_string(index=False))
    
    return all_results, summary_df
