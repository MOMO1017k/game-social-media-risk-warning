"""
自适应滚动检测器 (V7)：
  滚动TFT推理 → 分类器判定 → CP确认 → 回溯288窗口 → 积累微调 → 分层切换
"""

import os, copy, pickle, tempfile, io, contextlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import logging

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("TFTDataProcessor").setLevel(logging.WARNING)
logging.getLogger("TargetBuilder").setLevel(logging.WARNING)

# =====================================================================
# 分类器7: 纯统计阈值 AP/CP 判定
# =====================================================================
class StatisticalClassifier7:
    """
    基于冷启动期校准的多维Z-score分类器。
    包含方差下界约束与严格的时空归因隔离。
    """
    
    AP_ROLL_FEATURES = [
        'model_model_comment_pc1_residual_ratio_rollmax_8',
        'model_model_post_pc2_residual_ratio_rollmax_8',
        'model_model_comment_pc1_vsn_rank_shift_rollmax_8',
        'model_model_post_pc2_vsn_rank_shift_rollmax_8',
        'model_model_comment_pc1_att_kl_rollmax_8',
        'model_model_post_pc2_att_kl_rollmax_8',
        'model_model_comment_pc1_vsn_js_rollmax_8',
        'model_model_post_pc2_vsn_js_rollmax_8',
    ]
    
    CP_STATIC_FEATURES = [
        'model_model_comment_pc1_vsn_rank_shift',
        'model_model_post_pc2_vsn_rank_shift',
        'model_model_comment_pc1_att_kl',
        'model_model_post_pc2_att_kl',
    ]
    
    def __init__(self, ap_k=5.0, ap_min_triggers=4, cp_base_k=3.0, 
                 cp_density_thresh=0.95, cp_window=288, cp_min_periods=96, 
                 cp_trend_window=2688, cp_trend_k=2.0):
        self.ap_k = ap_k
        self.cp_base_k = cp_base_k
        self.cp_density_thresh = cp_density_thresh
        self.cp_window = cp_window
        self.cp_min_periods = cp_min_periods
        self.ap_min_triggers = ap_min_triggers
        self.cp_trend_window = cp_trend_window
        self.cp_trend_k = cp_trend_k
        
        self.ap_thresholds = {}
        self.cp_base_thresh = None
        self.cp_vol_thresh = None
        self.cp_trend_thresh = None 
        self._z_params = {} 
        self._calibrated = False
        self._ap_relax_factor = 1.0
    
    def calibrate(self, df_base):
        for feat in self.AP_ROLL_FEATURES:
            if feat in df_base.columns:
                self.ap_thresholds[feat] = df_base[feat].mean() + self.ap_k * df_base[feat].std()
        
        min_std_floors = {
            'model_model_comment_pc1_vsn_rank_shift': 0.05,
            'model_model_post_pc2_vsn_rank_shift': 0.05,
            'model_model_comment_pc1_att_kl': 0.1,
            'model_model_post_pc2_att_kl': 0.1,
            'model_model_comment_pc1_vsn_js': 0.02,
            'model_model_post_pc2_vsn_js': 0.02
        }

        for feat in self.CP_STATIC_FEATURES:
            if feat in df_base.columns:
                actual_std = df_base[feat].std()
                floor_std = min_std_floors.get(feat, 1e-4)
                safe_std = max(actual_std, floor_std)
                self._z_params[feat] = {'mean': df_base[feat].mean(), 'std': safe_std}
        
        z_max_vals = self._compute_z_max(df_base)
        self.cp_base_thresh = z_max_vals.mean() + self.cp_base_k * z_max_vals.std()
        self.cp_vol_thresh = z_max_vals.std() * 1.5
        self.cp_trend_thresh = z_max_vals.mean() + self.cp_trend_k * z_max_vals.std() 
        
        self._calibrated = True
        
    def set_ap_relax_factor(self, factor):
        self._ap_relax_factor = factor
    
    def _compute_z_max(self, df):
        z_vals = []
        for feat, params in self._z_params.items():
            if feat in df.columns:
                z = (df[feat] - params['mean']) / params['std']
                z_vals.append(z)
        if z_vals:
            return pd.concat(z_vals, axis=1).max(axis=1)
        return pd.Series(0.0, index=df.index)
    
    def predict(self, df):
        assert self._calibrated, "Calibration required before prediction"
        df = df.copy()
        
        z_max = self._compute_z_max(df)
        df['static_signal_z_max'] = z_max
        is_high = (z_max > self.cp_base_thresh).astype(float)
        df['cp_density'] = is_high.rolling(self.cp_window, min_periods=self.cp_min_periods).mean()
        df['cp_volatility'] = z_max.rolling(self.cp_window, min_periods=self.cp_min_periods).std()
        
        df['cp_trend_mean'] = z_max.rolling(self.cp_trend_window, min_periods=self.cp_window).mean()
        df['is_baseline_drift'] = df['cp_trend_mean'] > self.cp_trend_thresh
        
        trigger_matrix = pd.DataFrame(index=df.index)
        for feat in self.AP_ROLL_FEATURES:
            if feat in df.columns and feat in self.ap_thresholds:   
                effective_thresh = self.ap_thresholds[feat] * self._ap_relax_factor
                trigger_matrix[feat] = (df[feat] > effective_thresh).astype(int)
        df['ap_trigger_count'] = trigger_matrix.sum(axis=1)

        df['pred_label'] = 'N'
        mask_cp = (df['cp_density'] > self.cp_density_thresh) & (df['cp_volatility'] < self.cp_vol_thresh)
        df.loc[mask_cp, 'pred_label'] = 'CP'
        
        mask_ap = (df['ap_trigger_count'] >= self.ap_min_triggers) & (~mask_cp)
        df.loc[mask_ap, 'pred_label'] = 'AP'
        
        # --- 归因时间点定位 ---
        df['attribution_time'] = pd.NaT
        df.loc[mask_ap, 'attribution_time'] = df.loc[mask_ap, 'timestamp']
        
        ca_attr_times = df['timestamp'].shift(self.cp_window)
        df.loc[mask_cp, 'attribution_time'] = ca_attr_times.loc[mask_cp]
        
        cp_attr_times = df['timestamp'].shift(1344) 
        drift_mask = (df['is_baseline_drift'] == True)
        # 隔离约束：防止 CP 的宏观归因时间覆盖 PA 的微观爆发时间
        empty_time_mask = df['attribution_time'].isna()
        df.loc[drift_mask & empty_time_mask, 'attribution_time'] = cp_attr_times.loc[drift_mask & empty_time_mask]

        df['dominant_model'] = 'N/A'
        df['dominant_signal'] = 'N/A'
        df['trigger_features'] = ''
        
        anom_mask = df['pred_label'] != 'N'
        if anom_mask.any():
            for idx in df[anom_mask].index:
                triggers = [f for f in trigger_matrix.columns if trigger_matrix.loc[idx, f] == 1]
                if not triggers:
                    continue

                comment_v = sum(1 for f in triggers if 'comment_pc1' in f)
                post_v = sum(1 for f in triggers if 'post_pc2' in f)
                dom_model = 'comment_pc1' if comment_v >= post_v else 'post_pc2'
                
                signal_votes = {}
                for sig_type in ['residual_ratio', 'vsn_rank_shift', 'att_kl', 'vsn_js']:
                    signal_votes[sig_type] = sum(1 for f in triggers if sig_type in f)
                dom_signal = max(signal_votes, key=signal_votes.get)
                
                df.loc[idx, 'dominant_model'] = dom_model
                df.loc[idx, 'dominant_signal'] = dom_signal
                df.loc[idx, 'trigger_features'] = '|'.join(triggers)
        
        return df


# =====================================================================
# 自适应滚动检测器 7
# =====================================================================
class AdaptiveRollingDetector7:
    
    NORMAL = 'NORMAL'
    CP_TENTATIVE = 'CP_TENTATIVE'
    ACCUMULATING = 'ACCUMULATING'
    SWITCHING = 'SWITCHING'
    
    def __init__(self, engine, tb, processor, classifier2,
                 buffer_size=672, step_size=24, min_required=None,
                 cp_confirm_windows=192, retrace_windows=288, accumulate_min=1344,
                 finetune_epochs=100, finetune_lr_scale=0.005, finetune_batch_size=32, finetune_grad_clip=0.5, 
                 held_out_days=2, held_out_min_windows=96,
                 switch_parallel_windows=288, switch_parallel_extend=480,
                 switch_recovery_threshold=0.85, switch_stability_threshold=0.85,
                 cooldown_windows=672, ap_warmup_windows=96, ap_warmup_relax=1.2,
                 inference_batch_size=256):
        
        self.engine = engine
        self.tb = tb
        self.processor = processor
        self.classifier2 = classifier2
        
        self.buffer_size = buffer_size
        self.step_size = step_size
        ds_0 = list(engine.datasets.values())[0]
        self.min_encoder = ds_0.max_encoder_length
        self.min_pred = ds_0.max_prediction_length
        self.min_required = min_required or (self.min_encoder + self.min_pred + 1)
        
        self.cp_confirm_windows = cp_confirm_windows
        self.retrace_windows = retrace_windows
        self.accumulate_min = accumulate_min
        
        self.finetune_epochs = finetune_epochs
        self.finetune_lr_scale = finetune_lr_scale
        self.finetune_batch_size = finetune_batch_size
        self.finetune_grad_clip = finetune_grad_clip
        self.held_out_days = held_out_days
        self.held_out_min_windows = held_out_min_windows
        
        self.switch_parallel_windows = switch_parallel_windows
        self.switch_parallel_extend = switch_parallel_extend
        self.switch_recovery_threshold = switch_recovery_threshold 
        self.switch_stability_threshold = switch_stability_threshold
        
        self.cooldown_windows = cooldown_windows
        self.ap_warmup_windows = ap_warmup_windows
        self.ap_warmup_relax = ap_warmup_relax
        self.inference_batch_size = inference_batch_size
        
        self.state = self.NORMAL
        self.baseline_version = 0
        self.cp_consec = 0
        self.cp_confirmed_tidx = None
        self.retrace_start_tidx = None
        self.accumulate_buffer_tidx = []
        self.cooldown_until_tidx = -1
        
        self._new_engine = None
        self._new_processor = None
        self._new_classifier2 = None
        self._parallel_preds_old = []
        self._parallel_preds_new = []
        self._parallel_count = 0
        self._switch_extended = False
        
        self._ap_warmup_remaining = 0
        self.cp_events = []
        self.switch_history = []
        self.last_baseline_update_tidx = -999999
        
    def save_state(self, path="./checkpoints/adaptive_state_v7"):
        import json
        os.makedirs(path, exist_ok=True)
        
        state_dict = {
            'state': self.state,
            'baseline_version': self.baseline_version,
            'cp_consec': self.cp_consec,
            'cp_confirmed_tidx': self.cp_confirmed_tidx,
            'retrace_start_tidx': self.retrace_start_tidx,
            'accumulate_buffer_tidx': self.accumulate_buffer_tidx,
            'cooldown_until_tidx': self.cooldown_until_tidx,
            'ap_warmup_remaining': self._ap_warmup_remaining,
            'parallel_count': self._parallel_count,
            'switch_extended': self._switch_extended,
            'cp_events': self.cp_events,
            'switch_history': self.switch_history,
        }
        
        with open(os.path.join(path, 'state.json'), 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        
        version_path = os.path.join(path, f'v{self.baseline_version}')
        os.makedirs(version_path, exist_ok=True)
        
        self.engine.save(os.path.join(version_path, 'engine'))
        with open(os.path.join(version_path, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        with open(os.path.join(version_path, 'classifier2.pkl'), 'wb') as f:
            pickle.dump(self.classifier2, f)
        
        if self.state == self.SWITCHING and self._new_engine is not None:
            new_path = os.path.join(path, f'v{self.baseline_version + 1}_pending')
            os.makedirs(new_path, exist_ok=True)
            self._new_engine.save(os.path.join(new_path, 'engine'))
            with open(os.path.join(new_path, 'processor.pkl'), 'wb') as f:
                pickle.dump(self._new_processor, f)
            with open(os.path.join(new_path, 'classifier2.pkl'), 'wb') as f:
                pickle.dump(self._new_classifier2, f)
    
    def load_state(self, path="./checkpoints/adaptive_state_v7"):
        import json
        state_file = os.path.join(path, 'state.json')
        if not os.path.exists(state_file):
            return False
        
        with open(state_file, 'r') as f:
            state_dict = json.load(f)
        
        self.state = state_dict['state']
        self.baseline_version = state_dict['baseline_version']
        self.cp_consec = state_dict['cp_consec']
        self.cp_confirmed_tidx = state_dict.get('cp_confirmed_tidx')
        self.retrace_start_tidx = state_dict.get('retrace_start_tidx')
        self.accumulate_buffer_tidx = state_dict.get('accumulate_buffer_tidx', [])
        self.cooldown_until_tidx = state_dict.get('cooldown_until_tidx', -1)
        self._ap_warmup_remaining = state_dict.get('ap_warmup_remaining', 0)
        self._parallel_count = state_dict.get('parallel_count', 0)
        self._switch_extended = state_dict.get('switch_extended', False)
        self.cp_events = state_dict.get('cp_events', [])
        self.switch_history = state_dict.get('switch_history', [])
        
        version_path = os.path.join(path, f'v{self.baseline_version}')
        if os.path.exists(version_path):
            from TFT_tft_engine import TFTEngine
            self.engine = TFTEngine.load(os.path.join(version_path, 'engine'), config_path='TFT_config.yaml')
            with open(os.path.join(version_path, 'processor.pkl'), 'rb') as f:
                self.processor = pickle.load(f)
            with open(os.path.join(version_path, 'classifier2.pkl'), 'rb') as f:
                self.classifier2 = pickle.load(f)
        
        if self.state == self.SWITCHING:
            new_path = os.path.join(path, f'v{self.baseline_version + 1}_pending')
            if os.path.exists(new_path):
                self._new_engine = TFTEngine.load(os.path.join(new_path, 'engine'), config_path='TFT_config.yaml')
                with open(os.path.join(new_path, 'processor.pkl'), 'rb') as f:
                    self._new_processor = pickle.load(f)
                with open(os.path.join(new_path, 'classifier2.pkl'), 'rb') as f:
                    self._new_classifier2 = pickle.load(f)
        return True
       
    def run(self, df_real_all, df_features_raw, df_official,
            quiet_start=None, quiet_end=None, output_path=None, resume=False):
        df = df_real_all.sort_values('time_idx').reset_index(drop=True)
        
        if resume and os.path.exists("./checkpoints/adaptive_state_v7/state.json"):
            self.load_state("./checkpoints/adaptive_state_v7")

        if quiet_start and quiet_end:
            quiet_mask = (df['timestamp'] >= quiet_start) & (df['timestamp'] < quiet_end)
            quiet_start_idx = df[df['timestamp'] >= quiet_start].index[0]
        else:
            quiet_end_ts = df['timestamp'].min() + pd.Timedelta(days=7)
            quiet_mask = df['timestamp'] < quiet_end_ts
            quiet_start_idx = 0
        
        pre_ctx = max(0, quiet_start_idx - self.min_required)
        bl_end = df[quiet_mask].index[-1] + 1
        df_for_bl = df.iloc[pre_ctx:bl_end].copy().reset_index(drop=True)
        bl_end_tidx = int(df.loc[quiet_mask, 'time_idx'].max())
        
        self._patch_engine_batch_size(self.engine)

        for mn in self.engine.models:
            res = self.engine.analyze_rolling(mn, df_for_bl, baseline_end_idx=bl_end_tidx)
            if 'baseline' not in res and 'metrics' in res:
                bl = self.engine._build_baseline(
                    res['metrics'], res['attention'], res['vsn'], bl_end_tidx)
                self.engine.baselines[mn] = bl
        
        df_quiet_for_cal = self._extract_tft_batch(df_for_bl, engine=self.engine)
        if len(df_quiet_for_cal) > 0:
            cal_mask = df_quiet_for_cal['time_idx'] <= bl_end_tidx
            self.classifier2.calibrate(df_quiet_for_cal[cal_mask])
        else:
            self.classifier2.calibrate(df[quiet_mask])
        
        initial_end = min(quiet_start_idx + self.buffer_size, len(df))
        buffer_df = df.iloc[pre_ctx:initial_end].copy().reset_index(drop=True)
        
        quiet_start_tidx = int(df.loc[quiet_mask, 'time_idx'].min())
        output_tidx = set(
            buffer_df.loc[buffer_df['time_idx'] >= quiet_start_tidx, 'time_idx']
            .astype(int).values)
        
        df_cold = self._extract_tft_batch(buffer_df, engine=self.engine, output_tidx_set=output_tidx)
        all_classified_results = []
        processed_tidx = set(df_cold['time_idx'].astype(int).values) if len(df_cold) > 0 else set()
    
        remaining_start = initial_end
        total_remaining = len(df) - remaining_start
        n_batches = (total_remaining + self.step_size - 1) // self.step_size
        
        accumulated_tft_features = df_cold.copy() if len(df_cold) > 0 else pd.DataFrame()

        for batch_i in tqdm(range(n_batches), desc="自适应滚动检测"):
            batch_start = remaining_start + batch_i * self.step_size
            batch_end = min(batch_start + self.step_size, len(df))
            
            if batch_start >= len(df):
                break
            
            new_rows = df.iloc[batch_start:batch_end]
            new_tidx = set(new_rows['time_idx'].astype(int).values) - processed_tidx
            
            if not new_tidx:
                continue
            
            buffer_df = pd.concat([buffer_df, new_rows], ignore_index=True)
            if len(buffer_df) > self.buffer_size:
                buffer_df = buffer_df.iloc[-self.buffer_size:].reset_index(drop=True)
            
            if len(buffer_df) < self.min_required:
                continue
         
            if self._ap_warmup_remaining > 0:
                self.classifier2.set_ap_relax_factor(self.ap_warmup_relax)
                self._ap_warmup_remaining -= len(new_tidx)
                if self._ap_warmup_remaining <= 0:
                    self.classifier2.set_ap_relax_factor(1.0)
       
            if self.state == self.SWITCHING and self._new_engine is not None:
                df_feat_old = self._extract_tft_batch(buffer_df, engine=self.engine, output_tidx_set=new_tidx)
                df_feat_new = self._extract_tft_batch(buffer_df, engine=self._new_engine, output_tidx_set=new_tidx)
                
                df_feat = df_feat_old
                
                if len(df_feat_old) > 0 and len(df_feat_new) > 0:
                    pred_old = self.classifier2.predict(df_feat_old)['pred_label'].values
                    pred_new = self._new_classifier2.predict(df_feat_new)['pred_label'].values
                   
                    self._parallel_preds_old.extend(pred_old.tolist())
                    self._parallel_preds_new.extend(pred_new.tolist())
                    self._parallel_count += len(pred_old)
            else:
                df_feat = self._extract_tft_batch(buffer_df, engine=self.engine, output_tidx_set=new_tidx)   
            
            if len(df_feat) == 0:
                continue
            
            processed_tidx.update(df_feat['time_idx'].astype(int).values)

            accumulated_tft_features = pd.concat([accumulated_tft_features, df_feat], ignore_index=True)
            max_history_needed = self.classifier2.cp_trend_window + self.step_size + 100
            if len(accumulated_tft_features) > max_history_needed:
                accumulated_tft_features = accumulated_tft_features.iloc[-max_history_needed:].reset_index(drop=True)
            
            df_classified_all = self.classifier2.predict(accumulated_tft_features)
            
            df_classified = df_classified_all[df_classified_all['time_idx'].isin(new_tidx)].copy()
            df_classified['baseline_version'] = self.baseline_version
            df_classified['state'] = self.state
            all_classified_results.append(df_classified)
            
            current_tidx = int(new_rows['time_idx'].iloc[-1])
            self._process_batch(df_classified, current_tidx, df, df_features_raw, df_official)     
                   
            # 按日统计输出逻辑
            eval_frequency = 96 // self.step_size
            if (batch_i + 1) % eval_frequency == 0:
                ts_date = new_rows['timestamp'].iloc[-1].strftime('%Y-%m-%d')
                n_ap = (df_classified['pred_label'] == 'AP').sum()
                n_cp = (df_classified['pred_label'] == 'CP').sum()
                drift_flag = df_classified['is_baseline_drift'].any()
                
                print(f"  [{ts_date}] state={self.state:<12} | v={self.baseline_version} | "
                      f"AP={n_ap:<2} CP={n_cp:<2} Drift={drift_flag}")
        
        if all_classified_results:
            df_final = pd.concat(all_classified_results, ignore_index=True)
            df_final = df_final.drop_duplicates(subset='time_idx', keep='last')
            df_final = df_final.sort_values('time_idx').reset_index(drop=True)
        else:
            df_final = pd.DataFrame()

        if output_path and len(df_final) > 0:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            df_final.to_csv(output_path, index=False)
        
        self.save_state()
        return df_final
    
    def _process_batch(self, df_classified, current_tidx, df_full, df_features_raw, df_official):
        labels = df_classified['pred_label'].values
        is_drift = df_classified['is_baseline_drift'].values
        
        if self.state == self.SWITCHING:
            self._check_parallel_switch(current_tidx)
            return

        if current_tidx < self.cooldown_until_tidx:
            return

        trend_window = self.classifier2.cp_trend_window
        is_trend_safe = (current_tidx - self.last_baseline_update_tidx) >= trend_window

        if is_drift.any() and is_trend_safe:
            retrace_start = max(0, current_tidx - self.accumulate_min)
            retrace_mask = (df_full['time_idx'] >= retrace_start) & (df_full['time_idx'] <= current_tidx)
            df_retrace = df_full[retrace_mask].copy()
            
            ts_start = df_retrace['timestamp'].min()
            ts_end = df_retrace['timestamp'].max()
            raw_mask = (df_features_raw['timestamp'] >= ts_start) & (df_features_raw['timestamp'] <= ts_end)
            df_raw_retrace = df_features_raw[raw_mask].copy()
            
            success = self._finetune(df_retrace, df_raw_retrace, df_official)
            
            if success:
                self.state = self.SWITCHING
                self._parallel_preds_old = []
                self._parallel_preds_new = []
                self._parallel_count = 0
                self._switch_extended = False
                
                self.cp_events.append({
                    'confirmed_tidx': current_tidx,
                    'retrace_tidx': retrace_start,
                    'retrace_windows': self.accumulate_min,
                    'drift_type': 'Incremental',
                    'switched_tidx': None,
                    'switch_type': None,
                    'baseline_version': self.baseline_version,
                })
            else:
                self.state = self.NORMAL
                self._reset_partial_state()
                self.cooldown_until_tidx = current_tidx + self.cooldown_windows
            return 

        if self.state == self.NORMAL:
            n_cp = (labels == 'CP').sum()
            if n_cp > 0:
                self.cp_consec += n_cp
                if self.cp_consec >= 1:
                     self.state = self.CP_TENTATIVE
            else:
                self.cp_consec = max(0, self.cp_consec - 1)
        
        elif self.state == self.CP_TENTATIVE:
            n_cp = (labels == 'CP').sum()
            if n_cp > 0:
                self.cp_consec += n_cp
            else:
                self.cp_consec = max(0, self.cp_consec - 1)
            
            if self.cp_consec >= self.cp_confirm_windows:
                self.cp_confirmed_tidx = current_tidx
                self.retrace_start_tidx = current_tidx  
                self.accumulate_buffer_tidx = []
                self.state = self.ACCUMULATING
      
            if self.cp_consec <= 0:
                self.state = self.NORMAL
                self.cp_consec = 0
                
        elif self.state == self.ACCUMULATING:
            batch_tidx = df_classified['time_idx'].astype(int).tolist()
            self.accumulate_buffer_tidx.extend(batch_tidx)
            accumulated = len(set(self.accumulate_buffer_tidx))
            
            if accumulated >= self.accumulate_min:
                retrace_mask = (df_full['time_idx'] >= self.retrace_start_tidx) & (df_full['time_idx'] <= current_tidx)
                df_retrace = df_full[retrace_mask].copy()
                
                ts_start = df_retrace['timestamp'].min()
                ts_end = df_retrace['timestamp'].max()
                raw_mask = (df_features_raw['timestamp'] >= ts_start) & (df_features_raw['timestamp'] <= ts_end)
                df_raw_retrace = df_features_raw[raw_mask].copy()
                
                success = self._finetune(df_retrace, df_raw_retrace, df_official)
                
                if success:
                    self.state = self.SWITCHING
                    self._parallel_preds_old = []
                    self._parallel_preds_new = []
                    self._parallel_count = 0
                    self._switch_extended = False
                    
                    self.cp_events.append({
                        'confirmed_tidx': self.cp_confirmed_tidx,
                        'retrace_tidx': self.retrace_start_tidx,
                        'retrace_windows': self.accumulate_min,
                        'drift_type': 'Sudden',
                        'switched_tidx': None,
                        'switch_type': None,
                        'baseline_version': self.baseline_version,
                    })
                else:
                    self.state = self.NORMAL
                    self._reset_partial_state()
                    self.cooldown_until_tidx = current_tidx + self.cooldown_windows

    def _check_parallel_switch(self, current_tidx):
        required_windows = (self.switch_parallel_extend 
                            if self._switch_extended 
                            else self.switch_parallel_windows)
        
        if self._parallel_count < required_windows:
            return
        
        pred_old = np.array(self._parallel_preds_old[-required_windows:])
        pred_new = np.array(self._parallel_preds_new[-required_windows:])
        
        if len(pred_old) == 0 or len(pred_new) == 0:
            return
        
        old_n_ratio = (pred_old == 'N').mean()
        new_n_ratio = (pred_new == 'N').mean()
        
        current_drift_type = self.cp_events[-1].get('drift_type', 'Sudden') if self.cp_events else 'Sudden'
        
        if current_drift_type == 'Sudden':
            if old_n_ratio >= self.switch_recovery_threshold:
                self._finalize_switch(current_tidx, 'aborted', agreement=0.0, kappa=0.0)
            elif new_n_ratio >= self.switch_stability_threshold and new_n_ratio >= old_n_ratio:
                self._execute_full_switch(current_tidx, agreement=0.0, kappa=0.0)
            else:
                self._handle_unstable_new_model(current_tidx, new_n_ratio)
                
        elif current_drift_type == 'Incremental':
            if new_n_ratio >= self.switch_stability_threshold:
                self._execute_full_switch(current_tidx, agreement=0.0, kappa=0.0)
            else:
                self._handle_unstable_new_model(current_tidx, new_n_ratio)

    def _handle_unstable_new_model(self, current_tidx, new_n_ratio):
        if not self._switch_extended:
            self._switch_extended = True
        else:
            self._finalize_switch(current_tidx, 'aborted', agreement=0.0, kappa=0.0)
    
    def _execute_full_switch(self, current_tidx, agreement, kappa):
        self.engine = self._new_engine
        self.processor = self._new_processor
        self.classifier2 = self._new_classifier2
        self._finalize_switch(current_tidx, 'full', agreement, kappa)
    
    def _execute_partial_switch(self, current_tidx, agreement, kappa):
        self.processor = self._new_processor
        self.classifier2.ap_thresholds = copy.deepcopy(self._new_classifier2.ap_thresholds)
        self.classifier2.cp_base_thresh = self._new_classifier2.cp_base_thresh
        self.classifier2.cp_vol_thresh = self._new_classifier2.cp_vol_thresh
        self.classifier2._z_params = copy.deepcopy(self._new_classifier2._z_params)
        
        for mn in self._new_engine.baselines:
            self.engine.baselines[mn] = self._new_engine.baselines[mn]
        self._finalize_switch(current_tidx, 'partial', agreement, kappa)
    
    def _finalize_switch(self, current_tidx, switch_type, agreement, kappa):
        if switch_type == 'full':
            self.baseline_version += 1
            self.last_baseline_update_tidx = current_tidx
        
        self.cooldown_until_tidx = current_tidx + self.cooldown_windows
        self._ap_warmup_remaining = self.ap_warmup_windows
        self.classifier2.set_ap_relax_factor(self.ap_warmup_relax)
        
        if self.cp_events:
            self.cp_events[-1]['switched_tidx'] = current_tidx
            self.cp_events[-1]['switch_type'] = switch_type
        
        self.switch_history.append({
            'tidx': current_tidx,
            'type': switch_type,
            'agreement': agreement,
            'kappa': kappa,
            'new_version': self.baseline_version,
        })
        
        self.state = self.NORMAL
        self._reset_partial_state()
    
    def _reset_partial_state(self):
        self.cp_consec = 0
        self.cp_confirmed_tidx = None
        self.retrace_start_tidx = None
        self.accumulate_buffer_tidx = []
        self._new_engine = None
        self._new_processor = None
        self._new_classifier2 = None
        self._parallel_preds_old = []
        self._parallel_preds_new = []
        self._parallel_count = 0
        self._switch_extended = False

    def _finetune(self, df_retrace, df_raw_retrace, df_official):
        from TFT_tft_engine import TFTEngine
        from tft_full_period_utils import compute_baseline_from_held_out
        
        processor_new = copy.deepcopy(self.processor)
        processor_new.feature_transformer.fit(df_raw_retrace.fillna(0))
 
        dataset_new = processor_new.transform(df_raw_retrace.fillna(0), df_official)
        df_tft = dataset_new.data.copy()

        df_tft.fillna(0, inplace=True)
        df_tft.replace([np.inf, -np.inf], 0, inplace=True)
        df_tft = self.tb.transform(df_tft)
        if 'label' not in df_tft.columns:
            df_tft['label'] = 'N'

        with tempfile.TemporaryDirectory() as tmp:
            self.engine.save(tmp)
            engine_new = TFTEngine.load(tmp, config_path='TFT_config.yaml')
        
        self._patch_engine_batch_size(engine_new)
        self._patch_engine_early_stopping(engine_new)
        
        held_out_windows = max(self.held_out_min_windows, int(len(df_tft) * 0.15))
        held_out_windows = min(held_out_windows, len(df_tft) // 3)

        split_idx = len(df_tft) - held_out_windows
        df_train_ft = df_tft.iloc[:split_idx].copy()
        df_held = df_tft.iloc[split_idx:].copy()
        held_out_tidx = set(df_held['time_idx'].values)

        if len(df_train_ft) < self.min_required * 2:
            return False
        
        for mn in engine_new.models:
            for p in engine_new.models[mn].parameters():
                p.requires_grad = True
            
            cfg = engine_new.config['tft_models'][mn]
            orig_lr = cfg.get('learning_rate', 0.03)
            cfg['learning_rate'] = orig_lr * self.finetune_lr_scale
            if 'gradient_clip_val' not in cfg:
                cfg['gradient_clip_val'] = self.finetune_grad_clip

            with contextlib.redirect_stdout(io.StringIO()):
                engine_new.build_and_fit(mn, df_train_ft, max_epochs=self.finetune_epochs)
            
            cfg['learning_rate'] = orig_lr
            
            res = engine_new.analyze_rolling(mn, df_tft, baseline_end_idx=None)
            if 'metrics' not in res:
                continue
            if held_out_tidx:
                bl = compute_baseline_from_held_out(res, held_out_tidx)
            else:
                bl = engine_new._build_baseline(
                    res['metrics'], res['attention'], res['vsn'], int(df_tft['time_idx'].max()))
            engine_new.baselines[mn] = bl

        classifier2_new = copy.deepcopy(self.classifier2)

        if len(df_held) > self.min_required:
            df_held_tft = self._extract_tft_batch(df_held, engine=engine_new)
            if len(df_held_tft) > 10:
                classifier2_new.calibrate(df_held_tft)
            else:
                classifier2_new = copy.deepcopy(self.classifier2)
        else:
            classifier2_new = copy.deepcopy(self.classifier2)
        
        self._new_engine = engine_new
        self._new_processor = processor_new
        self._new_classifier2 = classifier2_new
        
        return True

    def _patch_engine_batch_size(self, engine):
        inference_bs = self.inference_batch_size
        
        def patched_analyze(model_name, df, baseline_end_idx=None, predict=True):
            model = engine.models[model_name]
            train_dataset = engine.datasets[model_name]
            df_inf = df.copy()
            for col in engine.known_categoricals:
                df_inf[col] = df_inf[col].astype(str)
            
            if len(df_inf) <= train_dataset.max_encoder_length:
                return {}
                
            from pytorch_forecasting import TimeSeriesDataSet
            inference_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset, df_inf, predict=False, stop_randomization=True)
            
            if len(inference_dataset) == 0:
                return {}
            
            dataloader = inference_dataset.to_dataloader(
                train=False, batch_size=inference_bs, num_workers=0, pin_memory=True)
            
            import torch
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
                            "target_true": targets[i],
                            "pred_p50": p50[i],
                            "residual": targets[i] - p50[i],
                            "divergence": p90[i] - p10[i],
                            "attention": batch_att[i],
                            "vsn": batch_vsn[i]
                        }
            
            if not result_buffer:
                return {}
            
            sorted_t = sorted(result_buffer.keys())
            df_s = pd.DataFrame({
                "time_idx": sorted_t,
                "target_true": [result_buffer[t]["target_true"] for t in sorted_t],
                "pred_p50": [result_buffer[t]["pred_p50"] for t in sorted_t],
                "residual": [result_buffer[t]["residual"] for t in sorted_t],
                "divergence": [result_buffer[t]["divergence"] for t in sorted_t]
            })
            full_att = np.stack([result_buffer[t]["attention"] for t in sorted_t])
            full_vsn = np.stack([result_buffer[t]["vsn"] for t in sorted_t])
            
            result = {"metrics": df_s, "attention": full_att, "vsn": full_vsn}
            
            if baseline_end_idx is not None:
                bl = engine._build_baseline(df_s, full_att, full_vsn, baseline_end_idx)
                result["baseline"] = bl
                engine.baselines[model_name] = bl
            
            return result
        
        engine.analyze_rolling = patched_analyze
        
    def _patch_engine_early_stopping(self, engine):
        import types
        import torch
        import pandas as pd
        import lightning.pytorch as pl
        from lightning.pytorch.loggers import CSVLogger
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.data.encoders import NaNLabelEncoder

        ft_batch_size = self.finetune_batch_size

        def patched_build_and_fit(self_engine, model_name: str, df: pd.DataFrame, 
                                  val_ratio: float = 0.2, max_epochs: int = -1,
                                  quiet_end_idx: int = None):
            
            cfg = self_engine.config['tft_models'][model_name]
            pl.seed_everything(cfg.get('seed', 42))

            df_inf = df.copy()
            for col in self_engine.known_categoricals:
                df_inf[col] = df_inf[col].astype(str)

            if quiet_end_idx is not None:
                time_col = self_engine.global_cfg['time_column']
                df_for_training = df_inf[df_inf[time_col] <= quiet_end_idx].copy()
            else:
                df_for_training = df_inf

            time_col = self_engine.global_cfg['time_column']
            time_steps = df_for_training[time_col].sort_values().unique()
            split_idx = int(len(time_steps) * (1 - val_ratio))
            cutoff_time = time_steps[split_idx]
            
            train_df = df_for_training[df_for_training[time_col] <= cutoff_time]
            max_lookback = cfg.get('max_encoder_length', 48)
            val_start_time = time_steps[max(0, split_idx - max_lookback)]
            val_df = df_for_training[df_for_training[time_col] >= val_start_time]

            categorical_encoders = {
                name: NaNLabelEncoder(add_nan=True) for name in self_engine.known_categoricals
            }
            for g_id in self_engine.global_cfg['group_ids']:
                categorical_encoders[g_id] = NaNLabelEncoder(add_nan=True)

            training = TimeSeriesDataSet(
                train_df,
                time_idx=self_engine.global_cfg['time_column'],
                target=cfg['target'],
                group_ids=self_engine.global_cfg['group_ids'],
                min_encoder_length=cfg.get('min_encoder_length'),
                max_encoder_length=cfg.get('max_encoder_length'),
                max_prediction_length=cfg.get('max_prediction_length', 1),
                time_varying_unknown_reals=self_engine.unknown_reals,
                time_varying_known_reals=self_engine.known_reals,
                time_varying_known_categoricals=self_engine.known_categoricals,
                categorical_encoders=categorical_encoders,
                target_normalizer=GroupNormalizer(groups=self_engine.global_cfg['group_ids'], transformation=None),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True
            )

            validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

            train_dataloader = training.to_dataloader(train=True, batch_size=ft_batch_size, num_workers=0, pin_memory=True)
            val_dataloader = validation.to_dataloader(train=False, batch_size=ft_batch_size, num_workers=0, pin_memory=True)

            model = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=cfg.get('learning_rate', 0.03),
                hidden_size=cfg.get('hidden_size', 16),
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=8,
                output_size=len(self_engine.quantiles),
                loss=QuantileLoss(quantiles=self_engine.quantiles),
                reduce_on_plateau_patience=4
            )

            logger = CSVLogger("lightning_logs", name=model_name, flush_logs_every_n_steps=10)
            self_engine.log_dirs[model_name] = logger.log_dir

            early_stop_callback = EarlyStopping(
                monitor="val_loss", patience=8, min_delta=1e-3, mode="min", verbose=False
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, 
                dirpath=logger.log_dir, filename="best_model"
            )

            trainer_kwargs = {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "enable_model_summary": False,
                "enable_checkpointing": True,
                "callbacks": [early_stop_callback, checkpoint_callback],
                "logger": logger,
                "enable_progress_bar": True
            }

            if max_epochs == -1:
                trainer = pl.Trainer(**trainer_kwargs)
            elif max_epochs != 1:
                trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)
            elif max_epochs == 1:
                trainer_kwargs.update({"limit_train_batches": 5, "limit_val_batches": 5})
                trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)

            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            if checkpoint_callback.best_model_path:
                best_model = TemporalFusionTransformer.load_from_checkpoint(
                    checkpoint_callback.best_model_path, weights_only=False
                )
                model.load_state_dict(best_model.state_dict())

            self_engine.plot_training_history(model_name)
            self_engine.datasets[model_name] = training
            self_engine.models[model_name] = model

        engine.build_and_fit = types.MethodType(patched_build_and_fit, engine)

    def _extract_tft_batch(self, df_buffer, engine=None, output_tidx_set=None):
        if engine is None:
            engine = self.engine
        
        tft_results = {}
        for mn in engine.models:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                res = engine.analyze_rolling(mn, df_buffer, baseline_end_idx=None)
            tft_results[mn] = res
        
        if not any('metrics' in v for v in tft_results.values()):
            return pd.DataFrame()
        
        df_merged = df_buffer[['time_idx', 'timestamp']].copy()
        
        for model_name in tft_results:
            res = tft_results[model_name]
            if 'metrics' not in res:
                continue
            
            metrics = res['metrics']
            attention = res['attention']
            vsn = res['vsn']
            baseline = engine.baselines.get(model_name, {})
            
            if not baseline:
                continue
            
            n = len(metrics)
            
            residual_abs = metrics['residual'].abs().values
            baseline_p95 = max(
                abs(baseline.get('residual_p95', 0)),
                abs(baseline.get('residual_p5', 0)),
                baseline.get('residual_std', 1e-8) * 1.645, 1e-8)
            sig_residual = np.clip(residual_abs / baseline_p95, 0, 20)
            
            att_base = baseline.get('att_mean')
            sig_att_kl = np.zeros(n)
            if att_base is not None:
                for i in range(n):
                    p = attention[i].flatten()
                    q = att_base.flatten()
                    p = p / (p.sum() + 1e-10) + 1e-10
                    q = q / (q.sum() + 1e-10) + 1e-10
                    p, q = p / p.sum(), q / q.sum()
                    sig_att_kl[i] = np.sum(p * np.log(p / q))
            
            vsn_base = baseline.get('vsn_mean')
            sig_vsn_js = np.zeros(n)
            if vsn_base is not None:
                for i in range(n):
                    v_i = np.abs(vsn[i].flatten()) + 1e-10
                    v_b = np.abs(vsn_base.flatten()) + 1e-10
                    p_v, q_v = v_i / v_i.sum(), v_b / v_b.sum()
                    m_v = 0.5 * (p_v + q_v)
                    sig_vsn_js[i] = (
                        0.5 * np.sum(p_v * np.log(p_v / m_v)) +
                        0.5 * np.sum(q_v * np.log(q_v / m_v)))
            
            sig_vsn_rank = np.zeros(n)
            if vsn_base is not None:
                baseline_rank = np.argsort(np.argsort(-np.abs(vsn_base.flatten())))
                for i in range(n):
                    curr_rank = np.argsort(np.argsort(-np.abs(vsn[i].flatten())))
                    corr, _ = spearmanr(baseline_rank, curr_rank)
                    sig_vsn_rank[i] = 1 - corr if not np.isnan(corr) else 1.0
            
            prefix = f'model_{model_name}'
            tft_df = pd.DataFrame({
                'time_idx': metrics['time_idx'].values,
                f'{prefix}_residual_ratio': sig_residual,
                f'{prefix}_att_kl': np.clip(sig_att_kl, 0, 20),
                f'{prefix}_vsn_js': np.clip(sig_vsn_js, 0, 5),
                f'{prefix}_vsn_rank_shift': np.clip(sig_vsn_rank, 0, 2),
            }).drop_duplicates(subset='time_idx', keep='last')
            
            n_before = len(df_merged)
            df_merged = df_merged.merge(tft_df, on='time_idx', how='left')
            if len(df_merged) != n_before:
                new_cols = [c for c in tft_df.columns if c != 'time_idx']
                df_merged = df_merged.drop(columns=new_cols).iloc[:n_before]
                for col in new_cols:
                    val_map = dict(zip(tft_df['time_idx'], tft_df[col]))
                    df_merged[col] = df_merged['time_idx'].map(val_map)
        
        base_tft_cols = [c for c in df_merged.columns if c.startswith('model_') and 'roll' not in c]
        for col in base_tft_cols:
            for k in [4, 8]:
                df_merged[f'{col}_rollmax_{k}'] = (df_merged[col].rolling(k, min_periods=1).max())
                df_merged[f'{col}_rollmean_{k}'] = (df_merged[col].rolling(k, min_periods=1).mean())
        
        df_merged = df_merged.replace([np.inf, -np.inf], 0).fillna(0)
        
        keep_cols = ['time_idx', 'timestamp'] + [c for c in df_merged.columns if c.startswith('model_')]
        df_merged = df_merged[[c for c in keep_cols if c in df_merged.columns]]
        
        if output_tidx_set is not None:
            df_merged = df_merged[df_merged['time_idx'].isin(output_tidx_set)]
        
        return df_merged
