"""
target_builder.py — 分组PCA版（post/comment 分别做PCA）
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle, os


class TargetBuilder:
    # ── 按 post / comment 分组 ──
    POST_FEATURES = [
        'total_volume_post', 'gini_post', 'senti_symbol_post',
        'comp_ratio_post', 'total_short_post', 'total_long_post',
        'neg_ratio_post', 'semantic_shift_post',
        'retweet_ratio_post', 'vis_abs_redundancy_post', 'vis_concentration_post',
    ]
    COMMENT_FEATURES = [
        'total_volume_comment', 'gini_comment', 'senti_symbol_comment',
        'comp_ratio_comment', 'total_short_comment', 'total_long_comment',
        'neg_ratio_comment', 'semantic_shift_comment',
    ]
    # 全量（用于 z-score / feature_energy 等兼容旧逻辑）
    FEATURES = POST_FEATURES + COMMENT_FEATURES
    Z_CLIP = 5.0

    def __init__(self, n_components_post=5, n_components_comment=5):
        self.n_components_post = n_components_post
        self.n_components_comment = n_components_comment
        self.z_params = {}
        self.pca_post = None
        self.pca_comment = None
        self.is_fitted = False
        self.quiet_energy_median = None
        self.quiet_energy_mad = None
        self.quiet_energy_p95 = None

    # ------ fit ------
    def fit(self, df_quiet):
        missing = [f for f in self.FEATURES if f not in df_quiet.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # z-score 参数（全局，所有特征共用）
        self.z_params = {}
        for f in self.FEATURES:
            vals = df_quiet[f].dropna()
            median = vals.median()
            if abs(median) < 1e-10:
                p90 = vals.quantile(0.90)
                scale = max(p90, vals.std() * 0.5, 1e-6)
                method = 'P90'
            else:
                mad = np.median(np.abs(vals - median)) * 1.4826
                scale = max(mad, 1e-6)
                method = 'MAD'
            self.z_params[f] = {'median': float(median),
                                'scale': float(scale), 'method': method}

        z_all = self._compute_z_matrix(df_quiet, self.FEATURES)

        # ── 分组PCA ──
        z_post = self._compute_z_matrix(df_quiet, self.POST_FEATURES)
        z_comment = self._compute_z_matrix(df_quiet, self.COMMENT_FEATURES)

        self.pca_post = PCA(n_components=min(self.n_components_post, len(self.POST_FEATURES)))
        self.pca_post.fit(z_post.fillna(0))

        self.pca_comment = PCA(n_components=min(self.n_components_comment, len(self.COMMENT_FEATURES)))
        self.pca_comment.fit(z_comment.fillna(0))

        # 平静期基线统计量（基于全局 z）
        quiet_energy = (z_all ** 2).mean(axis=1)
        self.quiet_energy_median = float(quiet_energy.median())
        self.quiet_energy_mad = float(
            np.median(np.abs(quiet_energy - quiet_energy.median())) * 1.4826)
        self.quiet_energy_p95 = float(quiet_energy.quantile(0.95))

        self.is_fitted = True
        print(f"[TargetBuilder] Fitted on {len(df_quiet)} samples, clip={self.Z_CLIP}")
        print(f"  Zero-inflated: "
              f"{[f for f, p in self.z_params.items() if p['method'] == 'P90']}")
        print(f"  PCA_post var:    {[f'{v:.3f}' for v in self.pca_post.explained_variance_ratio_]}")
        print(f"  PCA_comment var: {[f'{v:.3f}' for v in self.pca_comment.explained_variance_ratio_]}")
        print(f"  Quiet energy: median={self.quiet_energy_median:.3f}, "
              f"MAD_σ={self.quiet_energy_mad:.3f}, P95={self.quiet_energy_p95:.3f}")

    # ------ transform ------
    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")
        df = df.copy()

        z_all = self._compute_z_matrix(df, self.FEATURES)
        # df['feature_energy'] = (z_all ** 2).mean(axis=1)
        # df['max_abs_z'] = z_all.abs().max(axis=1)
        # df['n_features_above_3sigma'] = (z_all.abs() > 3).sum(axis=1).astype(float)

        # ── 分组PCA scores ──
        z_post = self._compute_z_matrix(df, self.POST_FEATURES)
        z_comment = self._compute_z_matrix(df, self.COMMENT_FEATURES)

        pc_post = self.pca_post.transform(z_post.fillna(0))
        pc_comment = self.pca_comment.transform(z_comment.fillna(0))

        for i in range(pc_post.shape[1]):
            df[f'post_pc{i+1}'] = pc_post[:, i]
        for i in range(pc_comment.shape[1]):
            df[f'comment_pc{i+1}'] = pc_comment[:, i]

        # ── 兼容旧代码：保留全局 pc{i}_score（可选，后续可删除） ──
        # 这里不再生成 pc{i}_score，改用 post_pc{i} / comment_pc{i}

        return df

    # ------ internal ------
    def _compute_z_matrix(self, df, feature_list):
        z = pd.DataFrame(index=df.index)
        for f in feature_list:
            p = self.z_params[f]
            z[f] = np.clip((df[f] - p['median']) / p['scale'],
                           -self.Z_CLIP, self.Z_CLIP)
        return z

    # ------ diagnostics ------
    def get_diagnostics(self, df, label_col='label', regime_col='regime'):
        from sklearn.metrics import roc_auc_score
        df_eval = (df[df[regime_col] != 'cp_drift']
                   if regime_col in df.columns else df)
        y = (df_eval[label_col] != 'N').astype(int)
        if y.sum() == 0:
            return pd.DataFrame()

        # targets = ['feature_energy', 'max_abs_z', 'n_features_above_3sigma']
        targets = []
        # 加入分组PC scores
        targets += [f'post_pc{i+1}' for i in range(self.pca_post.n_components_)]
        targets += [f'comment_pc{i+1}' for i in range(self.pca_comment.n_components_)]
        targets = [t for t in targets if t in df_eval.columns]

        rows = []
        for t in targets:
            n_ = df_eval.loc[y == 0, t]
            a_ = df_eval.loc[y == 1, t]
            ps = np.sqrt((n_.std()**2 + a_.std()**2) / 2)
            d = (a_.mean() - n_.mean()) / (ps + 1e-10)
            try:
                auc = roc_auc_score(y, df_eval[t].abs())
            except Exception:
                auc = 0.5
            rows.append({'target': t, 'cohen_d': f'{d:.3f}', 'AUC': f'{auc:.3f}',
                         'normal_p50': f'{n_.median():.3f}',
                         'normal_p95': f'{n_.quantile(0.95):.3f}',
                         'anomaly_p50': f'{a_.median():.3f}',
                         'anomaly_p95': f'{a_.quantile(0.95):.3f}'})
        return pd.DataFrame(rows)

    # ------ PCA loadings 辅助 ------
    def print_loadings(self, top_k=3):
        """打印每个PC主要加载的特征"""
        print("\n📌 Post PCA Loadings:")
        loadings_post = pd.DataFrame(
            self.pca_post.components_,
            columns=self.POST_FEATURES,
            index=[f'post_PC{i+1}' for i in range(self.pca_post.n_components_)]
        )
        for pc in loadings_post.index:
            topk = loadings_post.loc[pc].abs().nlargest(top_k)
            signs = ['+' if loadings_post.loc[pc, f] > 0 else '-' for f in topk.index]
            print(f"  {pc}: {', '.join(f'{s}{f}({v:.2f})' for f, v, s in zip(topk.index, topk.values, signs))}")

        print("\n📌 Comment PCA Loadings:")
        loadings_comment = pd.DataFrame(
            self.pca_comment.components_,
            columns=self.COMMENT_FEATURES,
            index=[f'comment_PC{i+1}' for i in range(self.pca_comment.n_components_)]
        )
        for pc in loadings_comment.index:
            topk = loadings_comment.loc[pc].abs().nlargest(top_k)
            signs = ['+' if loadings_comment.loc[pc, f] > 0 else '-' for f in topk.index]
            print(f"  {pc}: {', '.join(f'{s}{f}({v:.2f})' for f, v, s in zip(topk.index, topk.values, signs))}")

    # ------ save / load ------
    def save(self, path="./checkpoints/target_builder.pkl"):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {
            'z_params': self.z_params,
            'pca_post': self.pca_post,
            'pca_comment': self.pca_comment,
            'n_components_post': self.n_components_post,
            'n_components_comment': self.n_components_comment,
            'post_features': self.POST_FEATURES,
            'comment_features': self.COMMENT_FEATURES,
            'features': self.FEATURES,
            'z_clip': self.Z_CLIP,
            'quiet_energy_median': self.quiet_energy_median,
            'quiet_energy_mad': self.quiet_energy_mad,
            'quiet_energy_p95': self.quiet_energy_p95,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[TargetBuilder] Saved to {path}")

    @classmethod
    def load(cls, path="./checkpoints/target_builder.pkl"):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        b = cls(n_components_post=state['n_components_post'],
                n_components_comment=state['n_components_comment'])
        b.z_params = state['z_params']
        b.pca_post = state['pca_post']
        b.pca_comment = state['pca_comment']
        b.Z_CLIP = state['z_clip']
        b.quiet_energy_median = state.get('quiet_energy_median')
        b.quiet_energy_mad = state.get('quiet_energy_mad')
        b.quiet_energy_p95 = state.get('quiet_energy_p95')
        b.is_fitted = True
        print(f"[TargetBuilder] Loaded from {path}")
        return b