# ===== NOTEBOOK CELL 210 =====
# ============================================================
# Cell: 自适应检测结果可视化 + crisis命中率 (重构版：PA / CA / CP)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 0. 数据读取与标签重构 ---
df_pic = pd.read_csv("./output/adaptive_detection_resultv5.csv")
df_pic['timestamp'] = pd.to_datetime(df_pic['timestamp'])

# 【核心修改】替换业务标签：AP -> PA (瞬时脉冲), CP -> CA (集体异常/中期冲击)
df_pic['pred_label'] = df_pic['pred_label'].replace({'AP': 'PA', 'CP': 'CA'})

# --- 1. 事件合并与统计 ---
print("="*60)
print(" 异常与变点检测结果统计 (PA / CA / CP)")
print("="*60)
vc = df_pic['pred_label'].value_counts()
drift_count = df_pic['is_baseline_drift'].sum() if 'is_baseline_drift' in df_pic.columns else 0

print(f"  总窗口: {len(df_pic)}")
print(f"  N(正常)={vc.get('N',0)}, PA(瞬时脉冲)={vc.get('PA',0)}, CA(中期冲击)={vc.get('CA',0)}, CP(长程漂移)={drift_count}")

# 1.1 提取 PA 和 CA 事件片段
df_pic['event_change'] = (df_pic['pred_label'] != df_pic['pred_label'].shift(1)).cumsum()
events = df_pic[df_pic['pred_label'] != 'N'].groupby(['pred_label', 'event_change']).agg(
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max'),
    duration_windows=('timestamp', 'count')
).reset_index().drop(columns=['event_change'])

# 1.2 提取真正的 CP (长程漂移) 事件片段
if 'is_baseline_drift' in df_pic.columns:
    df_pic['drift_change'] = (df_pic['is_baseline_drift'] != df_pic['is_baseline_drift'].shift(1)).cumsum()
    drift_events = df_pic[df_pic['is_baseline_drift'] == True].groupby('drift_change').agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        duration_windows=('timestamp', 'count')
    ).reset_index().drop(columns=['drift_change'])
    
    if len(drift_events) > 0:
        drift_events['pred_label'] = 'CP' # 赋予真正的变点标签
        events = pd.concat([events, drift_events], ignore_index=True)

events = events.sort_values('start_time').reset_index(drop=True)

print(f"\n  PA (脉冲) 事件: {len(events[events['pred_label']=='PA'])} 段")
print(f"  CA (冲击) 事件: {len(events[events['pred_label']=='CA'])} 段")
print(f"  CP (漂移) 事件: {len(events[events['pred_label']=='CP'])} 段")

# 1.3 Crisis命中率统计
if 'crisis_df' in locals() or 'crisis_df' in globals():
    crisis_df['t_crisis_start'] = pd.to_datetime(crisis_df['t_crisis_start'])
    crisis_df['t_crisis_end'] = pd.to_datetime(crisis_df['t_crisis_end'])

    events['in_crisis'] = False
    for idx, evt in events.iterrows():
        overlap = ((evt['start_time'] <= crisis_df['t_crisis_end']) & 
                   (evt['end_time'] >= crisis_df['t_crisis_start'])).any()
        events.at[idx, 'in_crisis'] = overlap

    for label in ['PA', 'CA', 'CP']:
        sub = events[events['pred_label'] == label]
        if len(sub) == 0:
            continue
        in_c = sub[sub['in_crisis']]
        out_c = sub[~sub['in_crisis']]
        print(f"\n  [{label}] 命中crisis: {len(in_c):<2} 段 ({in_c['duration_windows'].sum()/96:.1f}天)")
        print(f"  [{label}] crisis外: {len(out_c):<2} 段 ({out_c['duration_windows'].sum()/96:.1f}天)")
else:
    print("\n  ⚠️ 提示: 未找到 crisis_df，跳过命中率统计。")

# --- 2. 投票归因统计 (仅统计瞬时和短程冲击) ---
if 'dominant_model' in df_pic.columns:
    anom = df_pic[df_pic['pred_label'].isin(['PA', 'CA'])]
    if len(anom) > 0:
        print(f"\n[投票归因 (PA/CA)]")
        print(f"  主导子模型分布: {anom['dominant_model'].value_counts().to_dict()}")
        print(f"  主导信号类型分布: {anom['dominant_signal'].value_counts().to_dict()}")

# ============================================================
# 3. 时序全景可视化 (4 Panel 设计)
# ============================================================
# 判断是否有长程漂移数据决定画 3 张图还是 4 张图
has_trend = 'cp_trend_mean' in df_pic.columns
n_panels = 4 if has_trend else 3

fig, axes = plt.subplots(n_panels, 1, figsize=(20, 4 * n_panels), sharex=True)
if not isinstance(axes, np.ndarray):
    axes = [axes]

# 公共函数：绘制红色危机阴影
def draw_crisis_span(ax):
    if 'crisis_df' in locals() or 'crisis_df' in globals():
        for _, row in crisis_df.iterrows():
            ax.axvspan(row['t_crisis_start'], row['t_crisis_end'], alpha=0.12, color='red')

# (a) Panel 1: PA 检测 (原AP)
ax = axes[0]
if 'static_signal_z_max' in df_pic.columns:
    ax.plot(df_pic['timestamp'], df_pic['static_signal_z_max'], lw=0.5, alpha=0.7, color='steelblue', label='Static Z-max')
    draw_crisis_span(ax)
    
    pa_pts = df_pic[df_pic['pred_label'] == 'PA']
    ax.scatter(pa_pts['timestamp'], pa_pts['static_signal_z_max'], c='orange', s=10, zorder=5, label='PA (Pulse)')
    ax.set_ylabel('Static Z-max')
    ax.legend(loc='upper right')
    ax.set_title('PA 检测: 瞬时脉冲异常 (多维Z-score极值)')

# (b) Panel 2: CA 检测 (原突发CP)
ax = axes[1]
if 'cp_density' in df_pic.columns:
    ax.plot(df_pic['timestamp'], df_pic['cp_density'], lw=1, color='blue', alpha=0.8, label='Anomaly Density')
    draw_crisis_span(ax)
    
    ca_pts = df_pic[df_pic['pred_label'] == 'CA']
    ax.scatter(ca_pts['timestamp'], ca_pts['cp_density'], c='red', marker='^', s=20, zorder=5, label='CA (Collective)')
    
    ax.axhline(0.95, color='darkred', ls='--', alpha=0.5, label='CA 密度阈值 (0.95)')
    ax.set_ylabel('Anomaly Density')
    ax.legend(loc='upper right')
    ax.set_title('CA 检测: 中期冲击 / 集体异常 (288窗口高水位密度)')

# (c) Panel 3: CP 检测 (长程漂移) - 新增
if has_trend:
    ax = axes[2]
    ax.plot(df_pic['timestamp'], df_pic['cp_trend_mean'], lw=2.5, color='purple', label='28天均值')
    draw_crisis_span(ax)
    
    if 'is_baseline_drift' in df_pic.columns:
        cp_pts = df_pic[df_pic['is_baseline_drift'] == True]
        ax.scatter(cp_pts['timestamp'], cp_pts['cp_trend_mean'], c='magenta', marker='s', s=15, zorder=5, label='CP')
    
    # 估算阈值线画出辅助参考 (取漂移发生的最低均值点，若无漂移则不画)
    if drift_count > 0:
        drift_thresh = df_pic.loc[df_pic['is_baseline_drift'] == True, 'cp_trend_mean'].min()
        ax.axhline(drift_thresh, color='purple', ls=':', lw=2, alpha=0.8, label='阈值参考')
        
    ax.set_ylabel('趋势均值')
    ax.legend(loc='upper left')
    ax.set_title('长程漂移检测')

# (d) Panel 4: 自适应切换时间线
# ax = axes[-1]

# # 1. 直接绘制 CSV 中原生存储的正确阶跃线
# if 'baseline_version' in df_pic.columns:
#     ax.step(df_pic['timestamp'], df_pic['baseline_version'], 
#             where='pre', color='green', lw=2.5, label='Baseline Version')

# # 2. 动态获取检测器实例以提取事件时间戳进行打线
# det_obj = None
# if 'detector4' in locals():
#     det_obj = locals()['detector4']
# elif 'detector' in locals():
#     det_obj = locals()['detector']

# if det_obj and hasattr(det_obj, 'cp_events'):
#     for evt in det_obj.cp_events:
#         # 绘制立案确认线 (粉色/红色虚线)
#         if evt.get('confirmed_tidx'):
#             ts_row = df_pic[df_pic['time_idx'] == evt['confirmed_tidx']]
#             if len(ts_row) > 0:
#                 evt_type = evt.get('drift_type', 'CA')
#                 c = 'magenta' if evt_type == 'Incremental' else 'red'
#                 label_name = 'CP (Drift) Triggered' if evt_type == 'Incremental' else 'CA (Shock) Confirmed'
#                 ax.axvline(ts_row['timestamp'].iloc[0], color=c, ls='--', alpha=0.7, label=label_name)
        
#         # 绘制实际切换线 (绿色虚线)
#         if evt.get('switched_tidx'):
#             ts_row = df_pic[df_pic['time_idx'] == evt['switched_tidx']]
#             if len(ts_row) > 0:
#                 ax.axvline(ts_row['timestamp'].iloc[0], color='green', ls='--', alpha=0.9, lw=2, label='Baseline Switched')

# # 3. 坐标系与图例设置
# ax.set_ylabel('Version')
# ax.set_yticks(range(int(df_pic['baseline_version'].max() + 2) if 'baseline_version' in df_pic.columns else 5))
# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), loc='upper left')
# ax.set_title('状态机操作：底层基线自适应切换时间线')
# (d) Panel 4: 自适应切换时间线
# (d) Panel 4: 自适应切换时间线
ax = axes[-1]

# 1. 动态获取状态机对象
det_obj = None
if 'detector5' in locals():
    det_obj = locals()['detector5']
elif 'detector' in locals():
    det_obj = locals()['detector']

# 2. 严格基于 switched_tidx 重构基线版本阶跃线
# 消除并行验证期带来的 3 天时间差，确保版本跳变严格锚定实际切换点
df_pic['plot_version'] = 0
if det_obj and hasattr(det_obj, 'cp_events'):
    for evt in det_obj.cp_events:
        if evt.get('switched_tidx'):
            # 仅在真正执行切换的时刻之后，版本号予以递增
            mask = df_pic['time_idx'] >= evt['switched_tidx']
            df_pic.loc[mask, 'plot_version'] += 1

    # 绘制阶跃线 (使用 where='post' 确保阶跃垂直线精准对齐时间戳)
    ax.step(df_pic['timestamp'], df_pic['plot_version'], 
            where='post', color='green', lw=2.5, label='TFT模型版本')
    
    # 3. 绘制垂直参考状态线
    for evt in det_obj.cp_events:
        # 绘制立案确认线 (粉色/红色虚线)
        if evt.get('confirmed_tidx'):
            ts_row = df_pic[df_pic['time_idx'] == evt['confirmed_tidx']]
            if len(ts_row) > 0:
                evt_type = evt.get('drift_type', 'CA')
                c = 'magenta' if evt_type == 'Incremental' else 'red'
                label_name = 'CP触发' if evt_type == 'Incremental' else 'CA确认'
                ax.axvline(ts_row['timestamp'].iloc[0], color=c, ls='--', alpha=0.7, label=label_name)
        
        # 绘制实际切换线 (绿色虚线)
        if evt.get('switched_tidx'):
            ts_row = df_pic[df_pic['time_idx'] == evt['switched_tidx']]
            if len(ts_row) > 0:
                ax.axvline(ts_row['timestamp'].iloc[0], color='green', ls='--', alpha=0.9, lw=2, label='版本切换')

else:
    # 降级兼容：若无内存对象，依赖原始 CSV 绘制
    if 'baseline_version' in df_pic.columns:
        ax.step(df_pic['timestamp'], df_pic['baseline_version'], 
                where='post', color='green', lw=2.5, label='TFT模型版本')

# 4. 坐标系与图例收敛设置
ax.set_ylabel('版本')
max_v = df_pic['plot_version'].max() if 'plot_version' in df_pic.columns else 4
ax.set_yticks(range(int(max_v) + 2))

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')
ax.set_title('TFT模型自适应切换时间线')

# 动态获取检测器实例以提取真实的切换时间
# det_obj = None
# if 'detector4' in locals():
#     det_obj = locals()['detector4']
# elif 'detector' in locals():
#     det_obj = locals()['detector']

# # 方案：利用 detector 的历史记录，在 df_pic 中重构正确的 baseline_version
# if det_obj and hasattr(det_obj, 'cp_events'):
#     df_pic['fixed_version'] = 0
#     for evt in det_obj.cp_events:
#         if evt.get('switched_tidx'):
#             # 找到发生切换的那个时间点的 index
#             switch_mask = df_pic['time_idx'] >= evt['switched_tidx']
#             # 将该时间点之后的所有版本号 +1
#             df_pic.loc[switch_mask, 'fixed_version'] += 1
    
#     # 绘制重构后的真实阶跃线
#     ax.step(df_pic['timestamp'], df_pic['fixed_version'], where='post', color='green', lw=2.5, label='Baseline Version (Reconstructed)')
    
#     # 绘制垂直参考线
#     for evt in det_obj.cp_events:
#         if evt.get('confirmed_tidx'):
#             ts_row = df_pic[df_pic['time_idx'] == evt['confirmed_tidx']]
#             if len(ts_row) > 0:
#                 evt_type = evt.get('drift_type', 'CA')
#                 c = 'magenta' if evt_type == 'Incremental' else 'red'
#                 label_name = 'CP (Drift) Triggered' if evt_type == 'Incremental' else 'CA (Shock) Confirmed'
#                 ax.axvline(ts_row['timestamp'].iloc[0], color=c, ls='--', alpha=0.7, label=label_name)
        
#         if evt.get('switched_tidx'):
#             ts_row = df_pic[df_pic['time_idx'] == evt['switched_tidx']]
#             if len(ts_row) > 0:
#                 ax.axvline(ts_row['timestamp'].iloc[0], color='green', ls='--', alpha=0.9, lw=2, label='Baseline Switched')

# else:
#     # 兼容处理：如果没有内存对象，说明是在单独跑CSV，只能画CSV里的残留错误线
#     if 'baseline_version' in df_pic.columns:
#         ax.step(df_pic['timestamp'], df_pic['baseline_version'], where='post', color='green', lw=2, label='Baseline Version (Raw CSV)')

# ax.set_ylabel('Version')
# ax.set_yticks(range(int(df_pic['fixed_version'].max() + 2))) # 强制 Y 轴显示整数刻度 (0,1,2,3,4)
# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), loc='upper left')
# ax.set_title('状态机操作：底层基线自适应切换时间线')

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)
plt.suptitle('自适应滚动检测结果全景 (三级多尺度架构)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('./output/adaptive_detection_panorama_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 4. 归因热力图 (仅保留 PA/CA) ---
if 'trigger_features' in df_pic.columns:
    anom = df_pic[df_pic['pred_label'].isin(['PA', 'CA'])].copy()
    if len(anom) > 0:
        feat_counts = {}
        for feats_str in anom['trigger_features'].dropna():
            for f in feats_str.split('|'):
                if f:
                    feat_counts[f] = feat_counts.get(f, 0) + 1
        
        if feat_counts:
            fc_df = pd.Series(feat_counts).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(fc_df) * 0.4)))
            colors = ['#e74c3c' if 'comment' in f else '#2980b9' for f in fc_df.index]
            fc_df.plot.barh(ax=ax, color=colors)
            ax.set_xlabel('触发次数')
            ax.set_title('PA/CA 触发特征频率 (红=comment, 蓝=post)')
            plt.tight_layout()
            plt.savefig('./output/trigger_feature_frequency_v5.svg', dpi=200, bbox_inches='tight')
            plt.show()

print("\n✅ 可视化完成 (文件已保存为 _v5.png 结尾)")
# ===== NOTEBOOK CELL 211 =====
# ============================================================
# 新增模块：危机事件全生命周期与多维预警信号耦合分析
# ============================================================

# 确保时间字段已转换为标准 datetime 对象
crisis_df['t_crisis_start'] = pd.to_datetime(crisis_df['t_crisis_start'])
crisis_df['t_crisis_peak'] = pd.to_datetime(crisis_df['t_crisis_peak'])
crisis_df['t_official_resp'] = pd.to_datetime(crisis_df['t_official_resp'])
crisis_df['t_crisis_end'] = pd.to_datetime(crisis_df['t_crisis_end'])
crisis_df = crisis_df[crisis_df['t_crisis_start'] >= pd.to_datetime('2025-03-01')] # 过滤掉过早的危机事件
# 数据容器
analysis_records = []

for idx, row in crisis_df.iterrows():
    c_id = row['crisis_id']
    lvl = row['crisis_level']
    t_s = row['t_crisis_start']
    t_p = row['t_crisis_peak']
    t_r = row['t_official_resp']
    t_e = row['t_crisis_end']

    # 提取当前危机生命周期内的检测信号
    mask = (df_pic['timestamp'] >= t_s) & (df_pic['timestamp'] <= t_e)
    sub_df = df_pic[mask]

    pa_df = sub_df[sub_df['pred_label'] == 'PA']
    ca_df = sub_df[sub_df['pred_label'] == 'CA']
    cp_df = sub_df[sub_df['is_baseline_drift'] == True] if 'is_baseline_drift' in sub_df.columns else pd.DataFrame()

    # 信号时序统计
    first_pa = pa_df['timestamp'].min() if not pa_df.empty else pd.NaT
    first_ca = ca_df['timestamp'].min() if not ca_df.empty else pd.NaT
    first_cp = cp_df['timestamp'].min() if not cp_df.empty else pd.NaT

    # 首次预警点计算 (综合 PA 与 CA)
    early_signals = pd.concat([pa_df['timestamp'], ca_df['timestamp']])
    t_detect = early_signals.min() if not early_signals.empty else pd.NaT

    # 落点阶段判定与前置时间计算
    phase = '未成功报警'
    delta_t = pd.NaT
    delta_t_min = np.nan
    if pd.notna(t_detect):
        delta_t = t_r - t_detect
        delta_t_min = delta_t.total_seconds() / 60.0
        if t_detect < t_p:
            phase = '酝酿期'
        elif t_detect < t_r:
            phase = '爆发期'
        else:
            phase = '消退期'

    analysis_records.append({
        'crisis_id': c_id,
        'level': lvl,
        't_start': t_s,
        't_peak': t_p,
        't_resp': t_r,
        't_end': t_e,
        'has_pa': not pa_df.empty,
        'has_ca': not ca_df.empty,
        'has_cp': not cp_df.empty,
        'first_pa': first_pa,
        'first_ca': first_ca,
        't_detect': t_detect,
        'delta_t_min': delta_t_min,
        'phase': phase
    })

res_df = pd.DataFrame(analysis_records)

# --- 文本报告输出 ---
print("\n" + "="*60)
print(" 危机事件全生命周期与预警信号耦合量化分析")
print("="*60)

# 一、预警时效性与前置拦截能力分析
print("\n[一] 预警时效性与前置拦截能力分析")
detected_df = res_df[res_df['phase'] != '漏警 (FN)']
hit_rate = len(detected_df) / len(res_df) * 100
avg_advance = detected_df['delta_t_min'].mean()
phase_dist = res_df['phase'].value_counts()

print(f"  全局危机检出率: {hit_rate:.1f}% ({len(detected_df)}/{len(res_df)})")
print(f"  平均提前响应时间 (相对于官方响应): {avg_advance:.1f} 分钟")
print(f"  首发预警落点分布: ")
for p, count in phase_dist.items():
    print(f"    - {p}: {count} 件 ({count/len(res_df)*100:.1f}%)")

# 二、分层预警类型与事件生命周期的耦合规律
print("\n[二] 分层预警类型与事件生命周期的耦合规律")
pre_pa = res_df[(res_df['phase'] == '酝酿期') & res_df['has_pa']]
mid_ca = res_df[(res_df['phase'] == '爆发期') | (res_df['phase'] == '酝酿期 (Pre)')]
mid_ca_triggered = mid_ca[mid_ca['has_ca']]
post_cp = res_df[res_df['has_cp']]

print(f"  酝酿期 PA 检出比例: {len(pre_pa)}/{len(res_df[res_df['phase'] == '酝酿期 (Pre)'])} (反映系统对早期微弱信号的敏感度)")
print(f"  发酵至 CA 状态跃迁比例: {len(mid_ca_triggered)}/{len(res_df)} (反映中后期集群异常的确认情况)")
print(f"  触发长程重构 (CP) 的事件数: {len(post_cp)} 件 (反映不可逆分布改变的发生频率)")

# 三、危机等级 (Level) 对系统敏感度的非线性影响
print("\n[三] 危机等级 (Level) 对系统敏感度的影响")
grouped_lvl = res_df.groupby('level').agg(
    total=('crisis_id', 'count'),
    detected=('t_detect', lambda x: x.notna().sum()),
    ca_triggered=('has_ca', 'sum'),
    cp_triggered=('has_cp', 'sum')
)
for lvl, r in grouped_lvl.iterrows():
    print(f"  Level {lvl}: 共 {r['total']} 件 | 检出 {r['detected']} 件 | CA 触发 {r['ca_triggered']} 件 | CP 触发 {r['cp_triggered']} 件")

# 四、漏警事件 (False Negative) 的精细化归因
print("\n[四] 漏警事件 (False Negative) 分析")
fn_df = res_df[res_df['phase'] == '漏警 (FN)']
if fn_df.empty:
    print("  系统在全量危机事件中未出现漏警。")
else:
    print(f"  漏警事件数: {len(fn_df)} 件。需结合相位平滑或高频波动率约束做进一步归因分析。")
    for _, fn_r in fn_df.iterrows():
        print(f"    - Crisis ID {fn_r['crisis_id']} (Level {fn_r['level']}): 历时 {(fn_r['t_end'] - fn_r['t_start']).total_seconds()/3600:.1f} 小时")

# ============================================================
# 绘制：危机事件生命周期与多维预警映射甘特图
# ============================================================
fig, ax = plt.subplots(figsize=(14, max(6, len(res_df) * 0.5)))

# 设置 Y 轴标签
y_labels = []
for i, row in res_df.iterrows():
    y_pos = i
    y_labels.append(f"危机事件 {row['crisis_id']}\n(Lvl {row['level']})")
    
    # 时段区间长度计算
    len_pre = row['t_peak'] - row['t_start']
    len_mid = row['t_resp'] - row['t_peak']
    len_post = row['t_end'] - row['t_resp']
    
    # 绘制生命周期背景色块
    ax.barh(y_pos, len_pre, left=row['t_start'], color='#2ecc71', alpha=0.3, edgecolor='none', label='酝酿期' if i == 0 else "")
    ax.barh(y_pos, len_mid, left=row['t_peak'], color='#e74c3c', alpha=0.3, edgecolor='none', label='爆发期 ' if i == 0 else "")
    ax.barh(y_pos, len_post, left=row['t_resp'], color='#95a5a6', alpha=0.3, edgecolor='none', label='消退期' if i == 0 else "")
    
    # 绘制关键时间节点虚线
    ax.vlines(row['t_start'], ymin=y_pos-0.4, ymax=y_pos+0.4, color='green', linestyle=':', lw=1.5)
    ax.vlines(row['t_peak'], ymin=y_pos-0.4, ymax=y_pos+0.4, color='red', linestyle='--', lw=1.5)
    ax.vlines(row['t_resp'], ymin=y_pos-0.4, ymax=y_pos+0.4, color='black', linestyle='-', lw=1.5)
    
    # 提取并绘制对应的预警信号散点
    mask = (df_pic['timestamp'] >= row['t_start']) & (df_pic['timestamp'] <= row['t_end'])
    sub_df = df_pic[mask]
    
    pa_pts = sub_df[sub_df['pred_label'] == 'PA']
    ca_pts = sub_df[sub_df['pred_label'] == 'CA']
    cp_pts = sub_df[sub_df['is_baseline_drift'] == True] if 'is_baseline_drift' in sub_df.columns else pd.DataFrame()
    
    if not pa_pts.empty:
        ax.scatter(pa_pts['timestamp'], [y_pos]*len(pa_pts), color='orange', marker='o', s=40, zorder=3, label='PA (脉冲)' if i == 0 else "")
    if not ca_pts.empty:
        ax.scatter(ca_pts['timestamp'], [y_pos]*len(ca_pts), color='red', marker='^', s=70, zorder=4, label='CA (冲击)' if i == 0 else "")
    if not cp_pts.empty:
        ax.scatter(cp_pts['timestamp'], [y_pos]*len(cp_pts), color='purple', marker='s', s=70, zorder=5, label='CP (漂移)' if i == 0 else "")

ax.set_yticks(range(len(res_df)))
ax.set_yticklabels(y_labels)
# ax.set_xlabel("时间 ")
ax.set_title("事件生命周期与多维预警映射关系 (甘特图)", fontsize=20)

# 整理图例以防重复
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# 额外添加竖线图例说明
line_start = plt.Line2D([0], [0], color='green', linestyle=':', lw=1.5, label='起始时间')
line_peak = plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='峰值时间')
line_resp = plt.Line2D([0], [0], color='black', linestyle='-', lw=1.5, label='官方响应事件')
by_label.update({l.get_label(): l for l in [line_start, line_peak, line_resp]})

ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.xticks(rotation=30)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

output_file = './output/crisis_gantt_mapping_v5.svg'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
plt.show()
print(f"✅ 甘特图已保存至 {output_file}")
# ===== NOTEBOOK CELL 213 =====
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import pandas as pd

# 预设字体大小与样式，适应学术论文排版
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# 动态计算高度，额外增加空间以容纳图例
fig, ax = plt.subplots(figsize=(14, max(8, len(res_df) * 0.7)))

# 学术配色字典
colors = {
    'pre': '#AED6F1',        
    'mid': '#F5B7B1',        
    'post': '#E5E7E9',       
    'line_start': '#2980B9', 
    'line_peak': '#C0392B',  
    'line_resp': '#2C3E50',  
    'pa': '#F39C12',         
    'ca': '#E74C3C',         
    'cp': '#8E44AD'          
}

y_labels = []
for i, row in res_df.iterrows():
    y_pos = i
    y_labels.append(f"危机事件 {row['crisis_id']-5}\n(Lvl {row['level']})")
    
    # 时段区间长度计算
    len_pre = row['t_peak'] - row['t_start']
    len_mid = row['t_resp'] - row['t_peak']
    len_post = row['t_end'] - row['t_resp']
    
    # 绘制生命周期背景色块
    ax.barh(y_pos, len_pre, left=row['t_start'], color=colors['pre'], alpha=0.7, edgecolor='none', label='酝酿期' if i == 0 else "")
    ax.barh(y_pos, len_mid, left=row['t_peak'], color=colors['mid'], alpha=0.7, edgecolor='none', label='爆发期' if i == 0 else "")
    ax.barh(y_pos, len_post, left=row['t_resp'], color=colors['post'], alpha=0.7, edgecolor='none', label='消退期' if i == 0 else "")
    
    # 绘制关键时间节点虚线
    ax.vlines(row['t_start'], ymin=y_pos-0.45, ymax=y_pos+0.45, color=colors['line_start'], linestyle=':', lw=2.5)
    ax.vlines(row['t_peak'], ymin=y_pos-0.45, ymax=y_pos+0.45, color=colors['line_peak'], linestyle='--', lw=2.5)
    ax.vlines(row['t_resp'], ymin=y_pos-0.45, ymax=y_pos+0.45, color=colors['line_resp'], linestyle='-', lw=2.5)
    
    # 提取并绘制对应的预警信号散点
    mask = (df_pic['timestamp'] >= row['t_start']) & (df_pic['timestamp'] <= row['t_end'])
    sub_df = df_pic[mask]
    
    pa_pts = sub_df[sub_df['pred_label'] == 'PA']
    ca_pts = sub_df[sub_df['pred_label'] == 'CA']
    cp_pts = sub_df[sub_df['is_baseline_drift'] == True] if 'is_baseline_drift' in sub_df.columns else pd.DataFrame()
    
    if not pa_pts.empty:
        ax.scatter(pa_pts['timestamp'], [y_pos]*len(pa_pts), color=colors['pa'], marker='o', s=80, zorder=3)
    if not ca_pts.empty:
        ax.scatter(ca_pts['timestamp'], [y_pos]*len(ca_pts), color=colors['ca'], marker='^', s=120, zorder=4)
    if not cp_pts.empty:
        ax.scatter(cp_pts['timestamp'], [y_pos]*len(cp_pts), color=colors['cp'], marker='s', s=120, zorder=5)

ax.set_yticks(range(len(res_df)))
ax.set_yticklabels(y_labels)
# ax.set_title("事件生命周期与多维预警映射关系", pad=20)

# 整理背景色块的图例句柄
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# 创建显式的代理图例句柄
proxy_start = Line2D([0], [0], color=colors['line_start'], linestyle=':', lw=2.5, label='起始时间')
proxy_peak = Line2D([0], [0], color=colors['line_peak'], linestyle='--', lw=2.5, label='峰值时间')
proxy_resp = Line2D([0], [0], color=colors['line_resp'], linestyle='-', lw=2.5, label='官方响应时间')

proxy_pa = Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['pa'], markersize=10, label='点异常')
proxy_ca = Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['ca'], markersize=12, label='集体异常')
proxy_cp = Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['cp'], markersize=12, label='变点')

# 合并所有图例对象
for item in [proxy_start, proxy_peak, proxy_resp, proxy_pa, proxy_ca, proxy_cp]:
    by_label[item.get_label()] = item

# 按照指定逻辑顺序组织图例输出内容
legend_order = [
    '酝酿期', '爆发期', '消退期', 
    '起始时间', '峰值时间', '官方响应时间', 
    '点异常', '集体异常', '变点'
]
ordered_handles = [by_label[k] for k in legend_order if k in by_label]
ordered_labels = [k for k in legend_order if k in by_label]

# 图例排版
ax.legend(ordered_handles, ordered_labels, loc='upper left', ncol=3, framealpha=0.95, edgecolor='black', borderpad=0.8)

# 限定X轴起点为3月1日（年份依照实际数据集调整，此处设为2025）
ax.set_xlim(left=pd.to_datetime('2025-03-01'))

# 动态扩展 y 轴的顶部空间
ylim_bottom, ylim_top = ax.get_ylim()
ax.set_ylim(ylim_bottom, ylim_top + 1.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.xticks(rotation=30)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

output_file = './output/crisis_gantt_mapping_v5.1.svg'
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='svg')
plt.show()
print(f"✅ 甘特图已保存至 {output_file}")
# ===== NOTEBOOK CELL 214 =====
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import pandas as pd

# 预设字体大小与样式，适应学术论文排版
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# 动态计算高度，额外增加空间以容纳图例
fig, ax = plt.subplots(figsize=(14, max(8, len(res_df) * 0.7)))

# 学术配色字典
colors = {
    'pre': '#AED6F1',        
    'mid': '#F5B7B1',        
    'post': '#E5E7E9',       
    'line_start': '#2980B9', 
    'line_peak': '#C0392B',  
    'line_resp': '#2C3E50',  
    'pa': '#F39C12',         
    'ca': '#E74C3C',         
    'cp': '#8E44AD'          
}

y_labels = []
for i, row in res_df.iterrows():
    y_pos = i
    y_labels.append(f"危机事件 {row['crisis_id']-5}\n(Lvl {row['level']})")
    
    # 时段区间长度计算
    len_pre = row['t_peak'] - row['t_start']
    len_mid = row['t_resp'] - row['t_peak']
    len_post = row['t_end'] - row['t_resp']
    
    # 绘制生命周期背景色块
    ax.barh(y_pos, len_pre, left=row['t_start'], color=colors['pre'], alpha=0.7, edgecolor='none', label='酝酿期' if i == 0 else "")
    ax.barh(y_pos, len_mid, left=row['t_peak'], color=colors['mid'], alpha=0.7, edgecolor='none', label='爆发期' if i == 0 else "")
    ax.barh(y_pos, len_post, left=row['t_resp'], color=colors['post'], alpha=0.7, edgecolor='none', label='消退期' if i == 0 else "")
    
    # 绘制关键时间节点虚线
    ax.vlines(row['t_start'], ymin=y_pos-0.45, ymax=y_pos+0.45, color=colors['line_start'], linestyle=':', lw=2.5)
    ax.vlines(row['t_peak'], ymin=y_pos-0.45, ymax=y_pos+0.45, color=colors['line_peak'], linestyle='--', lw=2.5)
    ax.vlines(row['t_resp'], ymin=y_pos-0.45, ymax=y_pos+0.45, color=colors['line_resp'], linestyle='-', lw=2.5)
    
    # 提取并绘制对应的预警信号散点
    mask = (df_pic['timestamp'] >= row['t_start']) & (df_pic['timestamp'] <= row['t_end'])
    sub_df = df_pic[mask]
    
    pa_pts = sub_df[sub_df['pred_label'] == 'PA']
    ca_pts = sub_df[sub_df['pred_label'] == 'CA']
    
    # 核心修改点：增加多重掩码过滤，确保CP期间仅在状态不为PA或CA时才赋值并渲染CP颜色
    if 'is_baseline_drift' in sub_df.columns:
        drift_mask = (sub_df['is_baseline_drift'] == True) & (sub_df['pred_label'].isin(['N']))
        cp_pts = sub_df[drift_mask]
    else:
        cp_pts = pd.DataFrame()
    
    if not pa_pts.empty:
        ax.scatter(pa_pts['timestamp'], [y_pos]*len(pa_pts), color=colors['pa'], marker='o', s=80, zorder=3)
    if not ca_pts.empty:
        ax.scatter(ca_pts['timestamp'], [y_pos]*len(ca_pts), color=colors['ca'], marker='^', s=120, zorder=4)
    if not cp_pts.empty:
        ax.scatter(cp_pts['timestamp'], [y_pos]*len(cp_pts), color=colors['cp'], marker='s', s=60, zorder=5,alpha=0.5)

ax.set_yticks(range(len(res_df)))
ax.set_yticklabels(y_labels)

# 整理背景色块的图例句柄
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# 创建显式的代理图例句柄
proxy_start = Line2D([0], [0], color=colors['line_start'], linestyle=':', lw=2.5, label='起始时间')
proxy_peak = Line2D([0], [0], color=colors['line_peak'], linestyle='--', lw=2.5, label='峰值时间')
proxy_resp = Line2D([0], [0], color=colors['line_resp'], linestyle='-', lw=2.5, label='官方响应时间')

proxy_pa = Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['pa'], markersize=10, label='点异常')
proxy_ca = Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['ca'], markersize=12, label='集体异常')
proxy_cp = Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['cp'], markersize=12, label='变点')

# 合并所有图例对象
for item in [proxy_start, proxy_peak, proxy_resp, proxy_pa, proxy_ca, proxy_cp]:
    by_label[item.get_label()] = item

# 按照指定逻辑顺序组织图例输出内容
legend_order = [
    '酝酿期', '爆发期', '消退期', 
    '起始时间', '峰值时间', '官方响应时间', 
    '点异常', '集体异常', '变点'
]
ordered_handles = [by_label[k] for k in legend_order if k in by_label]
ordered_labels = [k for k in legend_order if k in by_label]

# 图例排版
ax.legend(ordered_handles, ordered_labels, loc='upper left', ncol=3, framealpha=0.95, edgecolor='black', borderpad=0.8)

# 限定X轴起点为3月1日（年份依照实际数据集调整，此处设为2025）
ax.set_xlim(left=pd.to_datetime('2025-03-01'))

# 动态扩展 y 轴的顶部空间
ylim_bottom, ylim_top = ax.get_ylim()
ax.set_ylim(ylim_bottom, ylim_top + 1.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.xticks(rotation=30)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

output_file = './output/crisis_gantt_mapping_v5.2.svg'
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='svg')
plt.show()
print(f"✅ 甘特图已保存至 {output_file}")
# ===== NOTEBOOK CELL 215 =====
import matplotlib.pyplot as plt

# 预设字体大小与样式，适应学术论文排版
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'legend.fontsize': 13
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# (a) 数据源敏感度分布 数据配置
# ==========================================
labels_a = ['评论特征主导\n(comment_pc1)', '广场特征主导\n(post_pc2)']
sizes_a = [860, 173]
# 采用冷色系（蓝/青）代表数据源模态
colors_a = ['#5DADE2', '#48C9B0'] 

# ==========================================
# (b) 偏差机制主导分布 数据配置
# ==========================================
labels_b = ['关键特征重排序\n(vsn_rank_shift)', '特征空间贡献平移\n(vsn_js)', '时序注意力偏移\n(att_kl)']
sizes_b = [902, 130, 1]
# 采用暖色系（橙/黄/红）代表内部表征机制
colors_b = ['#EB984E', '#F4D03F', '#EC7063']

# 绘制左侧子图 (a)
wedges_a, texts_a, autotexts_a = axes[0].pie(
    sizes_a, 
    labels=labels_a, 
    autopct='%1.1f%%', 
    startangle=140,
    colors=colors_a, 
    wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2),
    pctdistance=0.75, 
    textprops={'fontsize': 14}
)
axes[0].set_title('(a) 数据源敏感度分布', pad=20)

# 绘制右侧子图 (b)
wedges_b, texts_b, autotexts_b = axes[1].pie(
    sizes_b, 
    labels=labels_b, 
    autopct='%1.1f%%', 
    startangle=140,
    colors=colors_b, 
    wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2),
    pctdistance=0.75, 
    textprops={'fontsize': 14}
)
axes[1].set_title('(b) 偏差机制主导分布', pad=20)

# 针对占比极小的数据（如0.1%），将其数值标签向外侧微调以防重叠
autotexts_b[2].set_position((1.1, 0.1))

# 调整全局布局与主标题
plt.suptitle('图 6.3 预警窗口底层特征多维投票归因分布', y=1.02, fontsize=20)
plt.figtext(0.5, -0.05, 
            "注：左侧展示了系统对不同模态数据源的响应差异，右侧揭示了导致预警触发的核心内部表征变化机理。", 
            ha="center", fontsize=14, color="#333333")

plt.tight_layout()

# 导出高清矢量图
output_file = './output/attribution_doughnut_chart.svg'
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='svg')
plt.show()

print(f"图表渲染完成，已导出至: {output_file}")
