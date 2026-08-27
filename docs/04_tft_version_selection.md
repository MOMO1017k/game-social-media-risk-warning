# 第 4 阶段代码审查：TFT 分层检测版本选择

## 最终选择

论文终稿对应的主版本应以 `TFT_main_v5-9-pc.ipynb` 为基础，但不是整本 Notebook 全量照搬。最贴近论文设计的是以下组合：

| 论文环节 | 采用来源 | 选择理由 |
| --- | --- | --- |
| 19 维特征处理与两组 PCA 目标 | 最新 Notebook 单元 18、22、23 | 使用鲁棒变换处理器和 `TargetBuilder`，目标明确为 `comment_pc1`、`post_pc2` |
| 两个 TFT 子模型训练 | 最新 Notebook 单元 32–35 | 两模型、分位数损失、时间顺序训练/验证、冻结与保存流程与论文一致 |
| TFT 推理与基线 | 最新 Notebook 单元 40、206 中的滚动推理函数 | 输出 Q10/Q50/Q90、残差、注意力和 VSN，能支持论文的四类偏差信号 |
| 四类偏差信号 | 最新 Notebook V7 的 `_extract_tft_batch()` | 包含残差比、注意力 KL、变量选择 JS、Spearman 秩相关距离，并生成 8 窗口滚动最大值 |
| 多维统计分类 | 最新 Notebook 单元 204 的 `StatisticalClassifier7` | 参数与论文表 6.2 完全对应，是所有版本中唯一同时包含 28 天漂移路径的版本 |
| 在线自适应机制 | 最新 Notebook 单元 204、206 的 V7 路径 | 包含集体异常延迟确认、变点立即微调、并行验证、切换、冷却与点异常阈值保护 |

本仓库的 `scripts/04_tft_detection/final/` 已从上述部分提取核心逻辑，并统一了论文术语和接口。

## 五个 Notebook 的定位

### `TFT_main_v5-9-pc存档v1.ipynb`

- 最早的完整骨架，包含数据处理、TFT、异常注入、LightGBM 分类器和正式检测；
- 有一个代码单元存在语法错误；
- 尚未形成论文终稿中的长程漂移和完整自适应状态机；
- 只适合作为早期研究记录。

### `TFT_main_v5-9-pc存档v2保留各项检测.ipynb`

- 保留多条候选检测路线和 `TFTSignalClassifier`；
- 仍以实验性分类器比较为主，未形成最终 V7 双轨自适应逻辑；
- 适合用于解释为什么放弃早期检测器，不应作为 GitHub 主入口。

### `TFT_main_v5-9-pc存档0312v1adp.ipynb`

- 开始加入自适应检测器和滚动微调；
- 文件末尾存在重复定义的 `StatisticalClassifier2` 与 `AdaptiveRollingDetector`；
- 变点、集体异常和状态命名仍混在一起，缺少最终 28 天趋势路径。

### `TFT_main_v5-9-pc0313v2短程完整.ipynb`

- 短程点异常/集体异常检测相对完整；
- 仍以 V2/V3 检测器为主，适合作为短程实验基线；
- 不满足论文终稿同时检测 PA、CA、CP 并驱动双轨自适应的完整要求。

### `TFT_main_v5-9-pc.ipynb`

- 唯一包含检测器 V4–V7 演化记录的版本；
- V7 参数与论文表 6.2 一致：点异常 `k=5.0`、至少 4/8 信号触发；集体异常 `k=3.0`、3 天窗口、密度 0.95、波动率系数 1.5；变点 `k=2.0`、28 天趋势窗口；
- 包含论文描述的双响应轨道、并行验证、切换判定与保护期；
- 因此它是最终版的代码来源，但仍需剔除早期 LightGBM、异常注入、LOSO 和 V2–V6 重复检测器。

## 不进入论文主链路的代码

最新 Notebook 中以下部分与论文终稿不一致或仅用于方法探索，应移动到实验区：

1. 基于危机事件池的人工异常注入；
2. 多世界异常场景生成；
3. LightGBM 三分类器、LOSO 交叉验证和 SHAP 分析；
4. `E_X`、`E_Y` 等候选分层分类器比较；
5. `StatisticalClassifier2` 至 `StatisticalClassifier6`；
6. `AdaptiveRollingDetector`、`AdaptiveRollingDetector2/4/5` 等早期状态机；
7. 为临时调试而动态覆盖 `TFTEngine` 方法的 monkey patch。

这些实验可以证明检测方案的演化过程，但论文终稿明确采用的是“冷启动期校准的统计阈值分类器”，不是依赖合成异常标签训练的 LightGBM 分类器。

## 从 V7 提取时修正的偏差

### 1. 标签语义

V7 代码用 `AP` 表示点异常，用 `CP` 表示集体异常，同时用 `is_baseline_drift` 表示真正的变点。这与论文中的 `PA`、`CA`、`CP` 定义不一致。提取版改为：

- `pred_label = N | PA | CA`；
- `is_changepoint = True | False`；
- 变点只触发自适应机制，不覆盖异常分类标签。

### 2. 模型名前缀

原 Notebook 的模型名已经带有 `model_`，特征构造时再次增加前缀，生成 `model_model_comment_pc1_*`。提取版统一为 `model_comment_pc1_*` 和 `model_post_pc2_*`。

### 3. 微调实现

论文要求“解冻顶层参数进行微调”。V7 在微调路径中先把所有参数设为可训练，随后通过 monkey patch 创建一个新的 TFT 模型再训练，不能证明是从旧权重开始的顶层微调。提取版将状态决策与训练引擎解耦，要求待补充的 `TFT_tft_engine.py` 提供真正的顶层微调接口。

### 4. 回溯参数

V7 构造函数接收 `retrace_windows=288`，但长程漂移分支实际使用 `accumulate_min` 计算回溯起点，导致该参数未生效。提取版的状态机明确使用 `drift_retrace_windows`。

### 5. 超参数一致性

V7 微调 monkey patch 将 Dropout 写死为 `0.1`、早停耐心值写为 `8`，而论文表 6.1 为 Dropout `0.2`、早停耐心值 `3`。正式引擎必须只读取统一配置文件，不在 Notebook 中覆盖。

## 调用文件补齐情况

原 Notebook 外置调用的本地模块现已全部收到，并按论文主链路优先级归档如下。

### P0：已归档的训练与检测模块

1. `TFT_transform.py`
   - 必须提供 `TFTDataProcessor`、`FeatureTransformer`、`TFTDataset`；
   - 负责鲁棒 Z 标准化、零膨胀特征缩放、时间特征、官方日历、全局 `time_idx` 和训练/推理一致性。
2. `target_builder.py`
   - 必须提供 `TargetBuilder`；
   - 负责帖子/评论分组 PCA、生成 `comment_pc1` 与 `post_pc2`、保存和加载 PCA/缩放参数。
3. `TFT_tft_engine.py`
   - 必须提供 `TFTEngine`；
   - 至少实现 `build_and_fit()`、`analyze_rolling()`、`freeze_layers()`、`save()`、`load()` 和基线构造；
   - Notebook V7 的自适应训练过程已原样提取到 `notebook_v7_detector.py`，用于复现已有论文结果；若未来把研究原型升级为生产系统，可再单独实现严格复用旧权重的 `fine_tune_top_layers()`，但这不替换本次论文复现代码。
4. `tft_full_period_utils.py`
   - 必须提供 `create_held_out_split()`、`compute_baseline_from_held_out()`、`verify_baseline_bias()`；
   - 训练集与 held-out 必须时间隔离，不能使用随机切分。
5. `TFT_utils.py`
   - 必须提供 `load_config()` 与 `setup_logger()`；
   - 应统一读取 `TFT_config.yaml`，不再依赖 Notebook 的当前工作目录。

非 Python 配置文件 `TFT_config.yaml` 也已归档。它可以被 PyYAML 加载，但含有重复的顶层 `tft_models` 声明，且引擎把 Dropout 写死为 `0.1`，导致配置值没有实际生效；最终配置仍以 `config/examples/tft_detection.example.yaml` 的论文参数为基准统一。

### P1：已归档的检测后归因模块

6. `TFT_Attribution_engine.py` 与 `TFT_topic_modeling.py`
   - 必须提供 `ContentAttributionEngine` 与 `DashboardGenerator`；
   - 属于下一阶段异常归因，不阻塞 TFT 统计检测本身。

### 已废弃实验代码，不进入最终主流程

7. `build_label_v2.py`
   - 已归档至 `experiments/04_tft_detection/label_injection/`；
   - 当前仍导入未提供的 `build_label.create_injection_pipeline()` 和 `PipelineConfig`，随后又直接引用未定义的 `InjectionPipeline`，因此不能作为独立入口运行；
   - 仅服务于人工异常注入、LightGBM 和 LOSO 实验；
   - 该方案已被最终研究路线完全放弃，只保留为研究过程记录，不加入正式入口或默认依赖。

## 仍需补充的实现

当前不再缺少论文主链路所调用的原始文件。最终 Notebook 的训练、V7 检测、执行与制图单元已提取到 `scripts/04_tft_detection/final/`；便携脚本只增加环境变量路径和必要导入，不改变检测参数与执行顺序。`build_label.py` 仅属于已经废弃的异常注入与 LightGBM 对照实验。

## 本轮生成内容

- 5 个 Notebook 已归档至 `notebooks/04_tft_detection/legacy/`，共清除 2948 个历史输出；
- 新建 `scripts/04_tft_detection/final/signals.py`；
- 新建 `scripts/04_tft_detection/final/statistical_classifier.py`；
- 新建 `scripts/04_tft_detection/final/adaptive_state.py`；
- 新建 `scripts/04_tft_detection/final/pipeline.py`；
- 新建论文参数示例 `config/examples/tft_detection.example.yaml`；
- 新增不依赖 TFT 权重的最小单元测试。
- 原始训练、变换、PCA 目标、全周期基线和配置文件已归档至 `scripts/04_tft_detection/legacy/`；
- 归因引擎、主题建模模块和已清除输出的归因 Notebook 已归档至第 5 阶段；
- `build_label_v2.py` 已隔离到实验目录。

当前仓库同时保留“Notebook 原样复现代码”和“论文命名整理版”。训练与检测脚本已完成语法验证；由于真实微博数据、模型权重和完整中间文件不进入公开仓库，论文数值结果的端到端复跑仍需在作者本地数据环境执行。
