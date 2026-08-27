
# ============================================================
# Cell: 自适应滚动检测 — 完整运行
# ============================================================
import types, io, contextlib, copy, os, pickle
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pytorch_forecasting import TimeSeriesDataSet
from TFT_tft_engine import TFTEngine
from target_builder import TargetBuilder
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("TFTDataProcessor").setLevel(logging.WARNING)
logging.getLogger("TargetBuilder").setLevel(logging.WARNING)

CKPT = "./checkpoints/pretrained_planPC"
tb = TargetBuilder.load(f"{CKPT}/target_builder.pkl")
engine_PC = TFTEngine.load(f"{CKPT}/tft_engine", config_path="TFT_config.yaml")
with open(f"{CKPT}/processor.pkl", 'rb') as f:
    processor = pickle.load(f)

df_features = df_features[df_features['timestamp']<pd.to_datetime('2025-11-13')].copy()

def analyze_rolling_patched(self, model_name, df, baseline_end_idx=None, predict=True):
    model = self.models[model_name]
    train_dataset = self.datasets[model_name]
    df_inf = df.copy()
    for col in self.known_categoricals:
        df_inf[col] = df_inf[col].astype(str)
        
    if len(df_inf) <= train_dataset.max_encoder_length:
        return {}
        
    inference_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, df_inf, predict=False, stop_randomization=True)
        
    if len(inference_dataset) == 0:
        return {}
        
    dataloader = inference_dataset.to_dataloader(
        train=False, batch_size=512, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    result_buffer = {}
    with torch.no_grad():
        for x, y in dataloader:
            x = {k: v.to(device) for k, v in x.items()}
            targets = y[0].cpu().numpy().flatten()
            raw_out = model(x)
            preds = raw_out.prediction.cpu().numpy().squeeze(axis=1)
            p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]
            time_idx = x['decoder_time_idx'].cpu().numpy().flatten()
            interp = model.interpret_output(raw_out, reduction="none")
            batch_att = interp['attention'].cpu().numpy()
            batch_vsn = interp['encoder_variables'].cpu().numpy()
            for i in range(len(time_idx)):
                t_id = int(time_idx[i])
                result_buffer[t_id] = {
                    "target_true": targets[i], "pred_p50": p50[i],
                    "residual": targets[i] - p50[i], "divergence": p90[i] - p10[i],
                    "attention": batch_att[i], "vsn": batch_vsn[i]}
    if not result_buffer:
        return {}
    sorted_t = sorted(result_buffer.keys())
    df_s = pd.DataFrame({
        "time_idx": sorted_t,
        "target_true": [result_buffer[t]["target_true"] for t in sorted_t],
        "pred_p50": [result_buffer[t]["pred_p50"] for t in sorted_t],
        "residual": [result_buffer[t]["residual"] for t in sorted_t],
        "divergence": [result_buffer[t]["divergence"] for t in sorted_t]})
    full_att = np.stack([result_buffer[t]["attention"] for t in sorted_t])
    full_vsn = np.stack([result_buffer[t]["vsn"] for t in sorted_t])
    result = {"metrics": df_s, "attention": full_att, "vsn": full_vsn}
    if baseline_end_idx is not None:
        bl = self._build_baseline(df_s, full_att, full_vsn, baseline_end_idx)
        result["baseline"] = bl
        self.baselines[model_name] = bl
    return result

engine_PC.analyze_rolling = types.MethodType(analyze_rolling_patched, engine_PC)

QUIET_START = pd.to_datetime('2025-02-18 00:00:00')
QUIET_END   = pd.to_datetime('2025-02-25 00:00:00')

mask_quiet = (df_features['timestamp'] >= QUIET_START) & (df_features['timestamp'] < QUIET_END)
df_quiet_raw = df_features[mask_quiet].copy().fillna(0)

processor_real = copy.deepcopy(processor)
processor_real.feature_transformer.fit(df_quiet_raw)

dataset_real = processor_real.transform(df_features.fillna(0), df_official_real)
df_real_all = dataset_real.data.copy()
df_real_all.fillna(0, inplace=True)
df_real_all.replace([np.inf, -np.inf], 0, inplace=True)
df_real_all = tb.transform(df_real_all)
if 'label' not in df_real_all.columns:
    df_real_all['label'] = 'N'
df_real_all = df_real_all.sort_values('time_idx').reset_index(drop=True)

# =====================================================================
# 初始化系统 7
# =====================================================================
classifier7 = StatisticalClassifier7(
    ap_k=5.0, 
    cp_base_k=3.0, 
    cp_density_thresh=0.95,
    cp_window=288, 
    cp_min_periods=96, 
    ap_min_triggers=4,
    cp_trend_window=2688,
    cp_trend_k=2.0
)

detector7 = AdaptiveRollingDetector7(
    engine=engine_PC,
    tb=tb,
    processor=processor_real,
    classifier2=classifier7,
    # 核心变动：推进至 24，实现更高频率的状态机判定，缓解末端聚集现象
    buffer_size=672, step_size=24, min_required=None,
    cp_confirm_windows=192,
    retrace_windows=288, accumulate_min=1344,
    finetune_epochs=100, finetune_lr_scale=0.005, finetune_batch_size=32, finetune_grad_clip=0.5, 
    held_out_days=2, held_out_min_windows=96,
    switch_parallel_windows=288, switch_parallel_extend=480,
    switch_recovery_threshold=0.85, switch_stability_threshold=0.85,
    cooldown_windows=672, ap_warmup_windows=96, ap_warmup_relax=1.2,
    inference_batch_size=256
)

# =====================================================================
# 执行运行
# =====================================================================
df_result7 = detector7.run(
    df_real_all=df_real_all,
    df_features_raw=df_features.fillna(0),
    df_official=df_official_real,
    quiet_start=QUIET_START,
    quiet_end=QUIET_END,
    output_path="./output/adaptive_detection_resultv7.csv"
)
