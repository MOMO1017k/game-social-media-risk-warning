# ===== NOTEBOOK CELL 1 =====
# ==================== 单元格 1: 导入库 ====================
"""
TFT异常检测和变点检测主流程
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import json

# ===== NOTEBOOK CELL 3 =====
import sys
import warnings
warnings.filterwarnings('ignore')

# 检查版本
print("📦 检查环境版本...")
import torch
# import lightning.pytorch as 
import lightning.pytorch as pl
try:
    import pytorch_forecasting as pf
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ PyTorch Lightning: {pl.__version__}")
    print(f"✅ PyTorch Forecasting: {pf.__version__}")
    
    # 检查版本兼容性
    pl_major = int(pl.__version__.split('.')[0])
    if pl_major >= 2:
        print("⚠️  警告: PyTorch Lightning 2.x 可能存在兼容性问题")
        print("   建议版本: pytorch-lightning<2.0.0")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    # print("请运行: pip install pytorch-forecasting pytorch-lightning==1.9.5")
    sys.exit(1)
# ===== NOTEBOOK CELL 5 =====

# ==================== 单元格 2: 加载数据 ====================
"""
加载模拟数据、运营日历、真实数据
"""
import os
# 数据路径（根据实际情况修改）

DATA_PATH = r'C:\tongji\0 code\00_data\03_TFT'
RAW_DATA_PATH = r'C:\tongji\0 code\00_data\01_EDA'


# 加载数据
print("📂 加载数据...")



transform_info = pd.read_csv(DATA_PATH +os.sep+ '变换信息汇总.csv')

df_official_pretrain = pd.read_csv(RAW_DATA_PATH +os.sep+ 'temp_all_official_standart_time_type_pretrain.csv')# 无crisis官方运营日历
df_official_real = pd.read_csv(RAW_DATA_PATH +os.sep+ 'temp_all_official_standart_time_type.csv')# 真实官方运营日历
df_features = pd.read_csv(RAW_DATA_PATH +os.sep+ 'Time_Series_Features_2024-12-05 00 to 2025-11-27 00_15min.csv')# 真实数据集
df_raw = pd.read_csv(RAW_DATA_PATH +os.sep+ 'temp_all_standart_time_2cleaned.csv')# 真实原始数据


# ===== NOTEBOOK CELL 7 =====

# %%
df_normal = pd.read_csv(r'C:\tongji\0 code\00_data\02_stimulate\g_step6_inverse'+os.sep+'synthetic_physical.csv')
df_normal['timestamp'] = pd.to_datetime(df_normal['timestamp'] )
# ★ 注意: 不再 set_index, 保持 timestamp 为列
# df_normal = df_normal.set_index('timestamp')

crisis_df =  pd.read_csv(r'C:\tongji\0 code\00_data\01_EDA\0_crisis_event'+os.sep+'crisis_event_pool.csv')
# print('df_normal',df_normal.columns)
# abnormal_point = sorted(list(pd.to_datetime(df_official[df_official['category'] == "crisis"]['timestamp'])))[1:]
# crisis_events = sorted(list(df_official[df_official['category'] == "crisis"]['timestamp']))
# del crisis_events[4]
# del crisis_events[0:2]

# %%

# ===== NOTEBOOK CELL 8 =====
# df_synthetic['timestamp'] = pd.to_datetime(df_synthetic['timestamp'])
# df_synthetic_labels['timestamp'] = pd.to_datetime(df_synthetic_labels['timestamp'])
df_official_pretrain['timestamp'] = pd.to_datetime(df_official_pretrain['timestamp'])
df_official_real['timestamp'] = pd.to_datetime(df_official_real['timestamp'])
df_features.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
# ===== NOTEBOOK CELL 9 =====
transform_info.drop(columns=['类型','原始偏度','变换偏度','偏度改善','状态','零值指示','规则覆盖'], inplace=True)
# ===== NOTEBOOK CELL 11 =====
# 导入预训练官方事件
df_official_pretrain['event'] = df_official_pretrain['official_author']+' ' + df_official_pretrain['category']
# df_official_pretrain.drop(columns=['official_author','category'], inplace=True)
df_official_pretrain = df_official_pretrain[['event','timestamp']]
df_official_pretrain['timestamp'] = pd.to_datetime(df_official_pretrain['timestamp'])
# ===== NOTEBOOK CELL 12 =====
# 导入真实官方事件
df_official_real['event'] = df_official_real['official_author']+' ' + df_official_real['category']
# df_official_real.drop(columns=['official_author','category'], inplace=True)
df_official_real = df_official_real[['event','timestamp']]
df_official_real['timestamp'] = pd.to_datetime(df_official_real['timestamp'])
# ===== NOTEBOOK CELL 13 =====
# 导入真实特征集
# df_features['timestamp'] = pd.to_datetime(df_features.index)
df_features['comp_ratio_post'] = 1-df_features['comp_ratio_post']
df_features['comp_ratio_comment'] = 1-df_features['comp_ratio_comment']
df_features.rename(columns={'origin_ratio': 'retweet_ratio_post'}, inplace=True)
df_features = df_features.drop(['total_medium_post','total_medium_comment','unique_users_post', 'unique_users_comment', 'semantic_shift_cross'],axis=1) 
# ===== NOTEBOOK CELL 18 =====
# %% [markdown]
# # TFT + LightGBM 舆情异常检测
# ## 第一阶段：数据处理

# %% 
# 导入必要的库
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from TFT_utils import load_config, setup_logger
from TFT_transform import (
    TFTDataProcessor, 
    FeatureTransformer,
    TFTDataset
)
from target_builder import TargetBuilder
from tft_full_period_utils import create_held_out_split, compute_baseline_from_held_out, verify_baseline_bias


# 设置日志
logger = setup_logger("TFT_Main")

# %%
# 加载配置
config = load_config("TFT_config.yaml")
print("配置加载成功")
print(f"跳过的变换方法: {config['data']['skip_transforms']}")
# print(f"静态属性映射: {config['data']['static_suffix_mapping']}")

# ===== NOTEBOOK CELL 22 =====

processor = TFTDataProcessor(config)

tft_dataset = processor.fit_transform(
    df_synthetic=df_normal,
    transform_info=transform_info,
    df_official=df_official_pretrain       # ← 无 crisis 日历
)
df_full = tft_dataset.data.copy()

# ===== NOTEBOOK CELL 23 =====

tb = TargetBuilder(n_components_post=2, n_components_comment=2)
tb.fit(df_full)
df_full = tb.transform(df_full)

# 验证目标列已生成
for target_col in ['comment_pc1', 'post_pc2']:
    assert target_col in df_full.columns, f"❌ {target_col} 未生成!"
    print(f"✅ {target_col}: median={df_full[target_col].median():.4f}, "
          f"P95={df_full[target_col].quantile(0.95):.4f}")

# 打印loadings
tb.print_loadings(top_k=3)
# ===== NOTEBOOK CELL 27 =====
# ==================== Step 3~6: Held-out + Baseline + 保存 ====================

from tft_full_period_utils import create_held_out_split, compute_baseline_from_held_out, verify_baseline_bias

# --- Step 3: 创建 Held-out ---
held_out_mask, held_mask, split_info = create_held_out_split(
        df_full,
        time_col='timestamp',
        block_days=4,  # Held-out长度（天）
        min_block_windows=96,   # ≥ 24h, TFT encoder 最小长度
    )
df_train = df_full[~held_mask].copy()
df_held_out = df_full[held_out_mask].copy()
held_out_time_indices = set(df_full.loc[held_mask, 'time_idx'].values)
print(f"训练集: {len(df_train)}, Held-out: {held_mask.sum()}")

# ===== NOTEBOOK CELL 32 =====
from TFT_tft_engine import TFTEngine

# ===== NOTEBOOK CELL 33 =====


# --- Step 4: 训练 TFT ---
engine_PC = TFTEngine('TFT_config.yaml')

for model_name in engine_PC.config['tft_models']:

    print(f"  训练 [{model_name}]  ({len(df_train)} windows)")
    print(f"  target = {engine_PC.config['tft_models'][model_name]['target']}")
    print(f"  {'─'*50}")
    # if engine_PC.config['tft_models'][model_name]['target'] == "comment_pc1":
    #     continue
    # else:
    engine_PC.build_and_fit(
        model_name=model_name,
        df=df_train,
        max_epochs=300,
        quiet_end_idx=None
    )


# ===== NOTEBOOK CELL 35 =====


# --- Step 6: 冻结 + 保存 ---
import pickle, os

SAVE_DIR = "./checkpoints/pretrained_planPC"

for model_name in engine_PC.models:
    engine_PC.freeze_layers(model_name)

engine_PC.save(f"{SAVE_DIR}/tft_engine")
tb.save(f"{SAVE_DIR}/target_builder.pkl")

os.makedirs(SAVE_DIR, exist_ok=True)
with open(f"{SAVE_DIR}/processor.pkl", 'wb') as f:
    pickle.dump(processor, f)

print(f"\n🎯 全部保存到: {SAVE_DIR}")

# ===== NOTEBOOK CELL 36 =====
# 保存 processor (部署时需要)
import pickle, os
proc_path = os.path.join(SAVE_DIR, "processor.pkl")
with open(proc_path, 'wb') as f:
    pickle.dump(processor, f)
print(f"  ✅ Processor saved to {proc_path}")

# 保存 split_info (可追溯)
info_path = os.path.join(SAVE_DIR, "split_info.pkl")
with open(info_path, 'wb') as f:
    pickle.dump(split_info, f)
# ===== NOTEBOOK CELL 40 =====

# --- Step 5: 推理全量 + Held-out Baseline ---
results_PC = {}
for model_name in engine_PC.models:
    print(f"\n推理: {model_name}")
    res = engine_PC.analyze_rolling(model_name, df_full, baseline_end_idx=None)
    baseline = compute_baseline_from_held_out(res, held_out_time_indices)
    engine_PC.baselines[model_name] = baseline
    results_PC[model_name] = res
    print(f"  Baseline: residual μ={baseline['residual_mean']:.4f}, "
          f"σ={baseline['residual_std']:.4f}, P95={baseline['residual_p95']:.4f}")
