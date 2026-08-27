import yaml
import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger # [20260217新增] 用于记录训练日志
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
from TFT_utils import ensure_dir
import os


from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from lightning.pytorch.callbacks import EarlyStopping

class TFTEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.global_cfg = self.config.get('global', {})
        self.quantiles = self.global_cfg.get('quantiles', [0.1, 0.5, 0.9])
        
        self.known_reals = ['hour','dayofweek', 'month', 'day', 'is_weekend', 'is_holiday', 'time_slot']
        self.known_categoricals = ['event_code']
        # self.unknown_reals = ['total_volume_post', 'total_volume_comment',
        #     'gini_post','gini_comment', 
        #     'senti_symbol_post', 'senti_symbol_comment',
        #     'comp_ratio_post', 'comp_ratio_comment', 
        #     'total_short_post', 'total_long_post', 
        #     'total_short_comment','total_long_comment', 
        #     'neg_ratio_post', 'neg_ratio_comment',
        #     'semantic_shift_post', 'semantic_shift_comment',
        #     'retweet_ratio_post','vis_abs_redundancy_post', 'vis_concentration_post',]
        self.unknown_reals = [
            'total_volume_post', 'total_volume_comment',
            'gini_post', 'gini_comment',
            'senti_symbol_post', 'senti_symbol_comment',
            'comp_ratio_post', 'comp_ratio_comment',
            'total_short_post', 'total_long_post',
            'total_short_comment', 'total_long_comment',
            'neg_ratio_post', 'neg_ratio_comment',
            'semantic_shift_post', 'semantic_shift_comment',
            'retweet_ratio_post', 'vis_abs_redundancy_post', 'vis_concentration_post',
            # 复合目标（TargetBuilder生成）
             'post_pc2','comment_pc1',
            # 'feature_energy', 'max_abs_z',
            # 'pc1_score', 'pc2_score', 'pc3_score', 'pc4_score', 'pc5_score',
        ]
        self.datasets: Dict[str, TimeSeriesDataSet] = {}
        self.models: Dict[str, TemporalFusionTransformer] = {}
        self.baselines: Dict[str, Dict] = {}  # 加这行
        self.log_dirs = {}  # [20260217新增] 记录每个模型的日志路径
        

    def build_and_fit(self, model_name: str, df: pd.DataFrame, 
                      val_ratio: float = 0.2, max_epochs: int = -1,
                      quiet_end_idx: int = None):  # 新增参数
        """
        构建并训练模型
        
        Parameters
        ----------
        quiet_end_idx : int, optional
            平静期结束的time_idx。如果传入，只用time_idx <= quiet_end_idx的数据训练。
            这确保TFT只学习正常模式，不接触异常数据。
        """
        if model_name not in self.config['tft_models']:
            raise ValueError(f"Model {model_name} not found in config")
        
        cfg = self.config['tft_models'][model_name]
        pl.seed_everything(cfg.get('seed', 42))

        df = df.copy()
        for col in self.known_categoricals:
            df[col] = df[col].astype(str)

        # ===== 新增：如果指定了平静期，只用平静期数据 =====
        if quiet_end_idx is not None:
            time_col = self.global_cfg['time_column']
            df_quiet = df[df[time_col] <= quiet_end_idx].copy()
            print(f"[Quiet Mode] Using only quiet period data: "
                  f"time_idx <= {quiet_end_idx}, {len(df_quiet)} rows "
                  f"(out of {len(df)} total)")
            df_for_training = df_quiet
        else:
            df_for_training = df
        # ===== 新增结束 =====

        # 时间轴切分（在平静期内部切分）
        time_col = self.global_cfg['time_column']
        time_steps = df_for_training[time_col].sort_values().unique()
        split_idx = int(len(time_steps) * (1 - val_ratio))
        cutoff_time = time_steps[split_idx]
        
        train_df = df_for_training[df_for_training[time_col] <= cutoff_time]

        max_lookback = cfg.get('max_encoder_length', 48)
        val_start_time = time_steps[max(0, split_idx - max_lookback)]
        val_df = df_for_training[df_for_training[time_col] >= val_start_time]

        print(f"Dataset Split ({'Quiet Period' if quiet_end_idx else 'Full Data'}):")
        print(f"  -> Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        print(f"  -> Train cutoff: {cutoff_time}")


        # --- [修改2] 构建 Dataset 时配置 categorical_encoders ---
        # 为所有的类别变量配置 "add_nan=True"，允许遇到新类别
        categorical_encoders = {
            name: NaNLabelEncoder(add_nan=True) 
            for name in self.known_categoricals
        }

        for g_id in self.global_cfg['group_ids']:
            categorical_encoders[g_id] = NaNLabelEncoder(add_nan=True)


        

        # 2. 构建训练 Dataset
        training = TimeSeriesDataSet(
            train_df,
            time_idx=self.global_cfg['time_column'],
            target=cfg['target'],
            group_ids=self.global_cfg['group_ids'],
            min_encoder_length=cfg.get('min_encoder_length'),
            max_encoder_length=cfg.get('max_encoder_length'),
            max_prediction_length=cfg.get('max_prediction_length', 1), # 强制为1
            
            time_varying_unknown_reals=self.unknown_reals, # 包含target
            time_varying_known_reals=self.known_reals, # 数值型
            time_varying_known_categoricals=self.known_categoricals, # 类别型
            categorical_encoders=categorical_encoders, # 自定义的 encoders

            # static_categoricals=self.global_cfg.get('static_categoricals', []),
            # target_normalizer=GroupNormalizer(groups=self.global_cfg['group_ids'], transformation="softplus"),
            target_normalizer=GroupNormalizer(groups=self.global_cfg['group_ids'], transformation=None),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )


        # 3. 构建验证 Dataset
        # 使用 from_dataset 确保共享 Scaler 和类别编码
        validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)

        # 创建 DataLoaders
        train_dataloader = training.to_dataloader(train=True, batch_size=512, num_workers=0, pin_memory=True)
        val_dataloader = validation.to_dataloader(train=False, batch_size=512, num_workers=0, pin_memory=True)

        # 4. 模型初始化
        model = TemporalFusionTransformer.from_dataset(
            training,
            # optimizer="ranger",
            learning_rate=cfg.get('learning_rate', 0.03),
            hidden_size=cfg.get('hidden_size', 16),
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=len(self.quantiles),
            loss=QuantileLoss(quantiles=self.quantiles),
            reduce_on_plateau_patience=4
        )

        
        
        # 5. 训练器配置

        # [20260217修改] 配置 CSVLogger
        logger = CSVLogger("lightning_logs", name=model_name, flush_logs_every_n_steps=10)
        self.log_dirs[model_name] = logger.log_dir

        if max_epochs == -1: # 正式训练
            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu", 
                devices=1,
                callbacks=[EarlyStopping(
                    monitor="val_loss", min_delta=1e-3, patience=8, verbose=True,mode="min"
                    )], # 早停机制
                # val_check_interval=0.5, # 每个 epoch 验证两次，获取更细粒度的训练曲线
                enable_model_summary=True,# 在训练开始前，自动打印模型的层级结构表
                enable_checkpointing=True, # 是否自动保存模型权重文件
                logger=logger, # 是否启用实验追踪日志-20260217
                enable_progress_bar=True # Lightning 自带训练进度条
            )
        elif max_epochs !=1: # 正式训练
            trainer = pl.Trainer(
                max_epochs=max_epochs, 
                accelerator="gpu" if torch.cuda.is_available() else "cpu", 
                devices=1,
                callbacks=[EarlyStopping(
                    monitor="val_loss", min_delta=1e-3, patience=5, verbose=True,mode="min"
                    )], # 早停机制
                # val_check_interval=0.5, 
                enable_model_summary=True,# 在训练开始前，自动打印模型的层级结构表
                enable_checkpointing=True, # 是否自动保存模型权重文件
                logger=logger, # 是否启用实验追踪日志-20260217
                enable_progress_bar=True # Lightning 自带训练进度条
            )
        elif max_epochs == 1: # 测试
            trainer = pl.Trainer(
                max_epochs=max_epochs, 
                limit_train_batches=5,
                limit_val_batches=5,
                accelerator="gpu" if torch.cuda.is_available() else "cpu", 
                devices=1,
                callbacks=[EarlyStopping(monitor="val_loss", patience=3, verbose=True)], # 早停机制
                enable_model_summary=True,# 在训练开始前，自动打印模型的层级结构表
                enable_checkpointing=True, # 是否自动保存模型权重文件
                logger=logger, # 是否启用实验追踪日志-20260217
                enable_progress_bar=True # Lightning 自带训练进度条
            )



        # 6. 执行训练 (传入 val_dataloaders)
        trainer.fit(
            model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader
        )

        # [20260217新增] 画图
        self.plot_training_history(model_name)
        
        # 7. 保存状态 (通常保存 full dataset 的定义以便后续全量推理，或者保存 training)
        self.datasets[model_name] = training
        self.models[model_name] = model  # 训练结束后保存模型引用
        print(f"Model [{model_name}] built and trained successfully.")

    def plot_training_history(self, model_name):
        """[20260217新增] 读取日志并绘制 Loss 曲线"""
        log_dir = self.log_dirs.get(model_name)
        if not log_dir: return
        
        metrics_path = f"{log_dir}/metrics.csv"
        df_metrics = pd.read_csv(metrics_path)
        
        # 聚合 epoch 级别的 loss
        epoch_metrics = df_metrics.groupby("epoch")[["train_loss_epoch", "val_loss"]].mean()
        print(epoch_metrics)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_metrics.index, epoch_metrics["train_loss_epoch"], label="Train Loss")
        plt.plot(epoch_metrics.index, epoch_metrics["val_loss"], label="Val Loss", linestyle="--")
        plt.title(f"Training History: {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{log_dir}/loss_curve.png")
        plt.close()
        print(f"[TFT Engine] Loss curve saved to {log_dir}/loss_curve.png")

        
    def analyze_rolling(self, model_name: str, df: pd.DataFrame,
                        baseline_end_idx: int = None, predict = True) -> Dict:
        """
        执行滚动预测
        
        Parameters
        ----------
        baseline_end_idx : int, optional
            如果传入，用time_idx <= baseline_end_idx的预测结果建立正常基线。
            基线包含残差和不确定性的统计分布，用于后续偏差信号归一化。
        """
        print(f"\n[Analysis] Model: {model_name}")
        
        model = self.models[model_name]
        train_dataset = self.datasets[model_name]

        if not isinstance(train_dataset, TimeSeriesDataSet):
            raise TypeError(
                f"self.datasets['{model_name}'] is not TimeSeriesDataSet"
            )
        
        df_inf = df.copy()
        for col in self.known_categoricals:
            df_inf[col] = df_inf[col].astype(str)

        inference_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset, df_inf, predict=False, stop_randomization=True
        )

        if len(inference_dataset) == 0:
            print("[Analysis] WARNING: Dataset is empty!")
            return {}

        dataloader = inference_dataset.to_dataloader(
            train=False, batch_size=512, num_workers=0, pin_memory=True
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        encoder_vars = model.encoder_variables
        print(f"[Analysis] VSN features: {encoder_vars}")

        scalar_results = {
            "time_idx": [], "target_true": [], 
            "pred_p50": [], "residual": [], "divergence": []
        }
        tensor_attention = []
        tensor_vsn = []

        with torch.no_grad():
            iterator = tqdm(dataloader, desc=f"Inference {model_name}", unit="batch")
            for batch_idx, (x, y) in enumerate(iterator):
                x = {k: v.to(device) for k, v in x.items()}
                targets = y[0].cpu().numpy().flatten()
                
                raw_out = model(x)
                preds = raw_out.prediction.cpu().numpy().squeeze(axis=1)
                p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]
                time_idx = x['decoder_time_idx'].cpu().numpy().flatten()
                
                scalar_results["time_idx"].extend(time_idx)
                scalar_results["target_true"].extend(targets)
                scalar_results["pred_p50"].extend(p50)
                scalar_results["residual"].extend(targets - p50)
                scalar_results["divergence"].extend(p90 - p10)
                
                interpretation = model.interpret_output(raw_out, reduction="none")
                tensor_attention.append(interpretation['attention'].cpu().numpy())
                tensor_vsn.append(interpretation['encoder_variables'].cpu().numpy())

                if batch_idx == 0:
                    print(f"   Pred: {preds.shape}, ATT: {tensor_attention[0].shape}, "
                          f"VSN: {tensor_vsn[0].shape}")

        df_scalars = pd.DataFrame(scalar_results)
        full_attention = np.concatenate(tensor_attention, axis=0)
        full_vsn = np.concatenate(tensor_vsn, axis=0)
            # ★ 验证无重复
        # n_dup = df_scalars['time_idx'].duplicated().sum()
        # if n_dup > 0:
        #     print(f"[Analysis] WARNING: {n_dup} duplicate time_idx, deduplicating...")
        #     dup_mask = ~df_scalars['time_idx'].duplicated(keep='last')
        #     df_scalars = df_scalars[dup_mask].reset_index(drop=True)
        #     full_attention = full_attention[dup_mask.values]
        #     full_vsn = full_vsn[dup_mask.values]

        print(f"[Analysis] Scalars: {df_scalars.shape}, "
              f"ATT: {full_attention.shape}, VSN: {full_vsn.shape}")

        result = {
            "metrics": df_scalars,
            "attention": full_attention,
            "vsn": full_vsn
        }

        # ===== 新增：建立基线 =====
        if baseline_end_idx is not None:
            baseline = self._build_baseline(
                df_scalars, full_attention, full_vsn, baseline_end_idx
            )
            result["baseline"] = baseline
            self.baselines[model_name] = baseline
            print(f"[Baseline] Built from time_idx <= {baseline_end_idx}: "
                  f"residual μ={baseline['residual_mean']:.4f}, "
                  f"σ={baseline['residual_std']:.4f}")

        return result
    # ================================================================
    # 新增: 基线建立
    # ================================================================
    def _build_baseline(self, df_scalars, full_attention, full_vsn,
                        baseline_end_idx) -> Dict:
        """从平静期数据建立正常行为基线"""
        mask = df_scalars['time_idx'].values <= baseline_end_idx
        
        if mask.sum() < 100:
            print(f"[Baseline] WARNING: only {mask.sum()} samples in quiet period")
        
        quiet_residuals = df_scalars.loc[mask, 'residual'].values
        quiet_divergence = df_scalars.loc[mask, 'divergence'].values
        quiet_att = full_attention[mask]
        quiet_vsn = full_vsn[mask]
        
        # 用鲁棒统计量（应对冷启动期可能的少量异常）
        baseline = {
            # 残差分布
            'residual_mean': np.median(quiet_residuals),  # 用中位数更鲁棒
            'residual_std': np.median(np.abs(
                quiet_residuals - np.median(quiet_residuals)
            )) * 1.4826,  # MAD → σ
            'residual_p5': np.percentile(quiet_residuals, 5),
            'residual_p95': np.percentile(quiet_residuals, 95),
            
            # 不确定性分布
            'divergence_mean': np.median(quiet_divergence),
            'divergence_std': np.median(np.abs(
                quiet_divergence - np.median(quiet_divergence)
            )) * 1.4826,
            
            # 注意力权重基线
            'att_mean': np.mean(quiet_att, axis=0),
            'att_std': np.std(quiet_att, axis=0) + 1e-8,
            
            # VSN门控基线
            'vsn_mean': np.mean(quiet_vsn, axis=0),
            'vsn_std': np.std(quiet_vsn, axis=0) + 1e-8,
            
            # 元信息
            'n_samples': int(mask.sum()),
            'baseline_end_idx': int(baseline_end_idx),
        }
        
        return baseline

    # ================================================================
    # 新增: 参数冻结
    # ================================================================
    def freeze_layers(self, model_name: str):
        """
        冻结TFT约75%的参数
        冻结：LSTM主体、注意力层、静态编码器、GRN内部
        保留：输入投影、VSN门控softmax、输出头
        """
        model = self.models[model_name]
        
        # 先冻结全部
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻需要微调的部分
        unfrozen_keywords = [
            'output_layer',           # 分位数输出头
            'target_proj',            # 目标变量投影
            'prescalers',             # 输入投影/缩放
            'softmax',                # VSN门控softmax
        ]
        
        total, unfrozen = 0, 0
        for name, param in model.named_parameters():
            total += param.numel()
            if any(kw in name for kw in unfrozen_keywords):
                param.requires_grad = True
                unfrozen += param.numel()
        
        print(f"[Freeze] {model_name}: "
              f"{total - unfrozen}/{total} params frozen "
              f"({(total - unfrozen) / total * 100:.1f}%), "
              f"{unfrozen} params trainable")
    
    def unfreeze_all(self, model_name: str):
        """解冻所有参数"""
        for param in self.models[model_name].parameters():
            param.requires_grad = True
        print(f"[Unfreeze] {model_name}: all params trainable")

    # ================================================================
    # save/load: 增加baselines的保存加载
    # ================================================================
    def save(self, save_dir: str = "./checkpoints/tft_engine"):
        ensure_dir(save_dir)
        
        meta = {
            'config': self.config,
            'model_names': list(self.models.keys()),
            'log_dirs': self.log_dirs,
            'baselines': self.baselines,  # 新增
        }
        
        for name, model in self.models.items():
            ckpt_path = os.path.join(save_dir, f"{name}.ckpt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✅ Model '{name}' saved")
            
            if name in self.datasets:
                ds_path = os.path.join(save_dir, f"{name}_dataset.pkl")
                with open(ds_path, 'wb') as f:
                    pickle.dump(self.datasets[name], f)
        
        meta_path = os.path.join(save_dir, "engine_meta.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"\n🎯 TFT Engine saved to: {save_dir}")

    @classmethod
    def load(cls, save_dir: str = "./checkpoints/tft_engine", 
             config_path: str = "TFT_config.yaml"):
        meta_path = os.path.join(save_dir, "engine_meta.pkl")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        engine = cls(config_path)
        engine.log_dirs = meta.get('log_dirs', {})
        engine.baselines = meta.get('baselines', {})  # 新增
        
        for name in meta['model_names']:
            ckpt_path = os.path.join(save_dir, f"{name}.ckpt")
            ds_path = os.path.join(save_dir, f"{name}_dataset.pkl")
            
            if not os.path.exists(ckpt_path) or not os.path.exists(ds_path):
                print(f"  ⚠️ '{name}' files missing, skipping")
                continue
            
            with open(ds_path, 'rb') as f:
                dataset_obj = pickle.load(f)
            engine.datasets[name] = dataset_obj
            
            cfg = engine.config['tft_models'][name]
            model = TemporalFusionTransformer.from_dataset(
                dataset_obj,
                learning_rate=cfg.get('learning_rate', 0.03),
                hidden_size=cfg.get('hidden_size', 16),
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=8,
                output_size=len(engine.quantiles),
                loss=QuantileLoss(quantiles=engine.quantiles),
                reduce_on_plateau_patience=4
            )
            
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            
            engine.models[name] = model
            print(f"  ✅ '{name}' restored")
        
        print(f"\n🎯 TFT Engine loaded from: {save_dir}")
        return engine