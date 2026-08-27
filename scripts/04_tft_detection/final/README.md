# TFT 分层检测最终代码

本目录以 `TFT_main_v5-9-pc.ipynb` 最后实际执行的 V7 单元为复现来源。Notebook 原始单元和便于作品集阅读的整理版同时保留，两者用途不同。

## 论文结果复现代码

- `notebook_v7_detector.py`：逐字提取 Notebook 第 204 单元，包含 `StatisticalClassifier7` 与 `AdaptiveRollingDetector7`；
- `notebook_v7_run_cell.py`：逐字提取 Notebook 第 206 单元，保留论文结果的原始执行过程；
- `notebook_v7_run.py`：在第 206 单元基础上仅补充数据加载、类导入和环境变量路径，可直接作为运行脚本；
- `notebook_v7_train_cells.py`：按 Notebook 原顺序合并训练所需单元；
- `notebook_v7_train.py`：仅将训练单元中的本机绝对路径改为可覆盖的环境变量；
- `notebook_v7_figure_cells.py`：按 Notebook 原顺序合并最终 PA/CA/CP 结果可视化、危机生命周期分析和论文图单元；
- `notebook_v7_figures.py`：只为上述制图单元补充检测结果和危机事件池的环境变量加载；
- `TFT_transform.py`、`target_builder.py`、`TFT_tft_engine.py`、`tft_full_period_utils.py`、`TFT_utils.py` 与 `TFT_config.yaml`：Notebook 原始调用模块的完整副本。

`notebook_v7_detector.py` 已与 Notebook 第 204 单元逐字进行 SHA-256 核验，源码完全一致。

## 运行顺序

1. 准备 Notebook 使用的模拟正常序列、变换信息、官方运营日历和真实 15 分钟特征文件；
2. 设置下列路径环境变量；
3. 运行 `notebook_v7_train.py` 生成 `pretrained_planPC`；
4. 运行 `notebook_v7_run.py` 生成 `adaptive_detection_resultv7.csv`。

PowerShell 示例：

```powershell
$env:TFT_DATA_PATH = "D:\data\03_TFT"
$env:TFT_RAW_DATA_PATH = "D:\data\01_EDA"
$env:TFT_NORMAL_DATA_PATH = "D:\data\synthetic_physical.csv"
$env:TFT_CRISIS_POOL_PATH = "D:\data\crisis_event_pool.csv"
$env:TFT_CHECKPOINT_DIR = ".\checkpoints\pretrained_planPC"
$env:TFT_OUTPUT_PATH = ".\output\adaptive_detection_resultv7.csv"
$env:TFT_FIGURE_INPUT = ".\output\adaptive_detection_resultv7.csv"

python scripts\04_tft_detection\final\notebook_v7_train.py
python scripts\04_tft_detection\final\notebook_v7_run.py
python scripts\04_tft_detection\final\notebook_v7_figures.py
```

训练轮数默认保持 Notebook 的 `300`，可通过 `TFT_MAX_EPOCHS` 覆盖。检测终点、平静期、V7 阈值和自适应参数均保持 Notebook 原值。原始制图单元读取文件名 `adaptive_detection_resultv5.csv`；便携运行时可用 `TFT_FIGURE_INPUT` 指向 V7 输出。

## 作品集整理版

- `signals.py`：残差比、注意力 KL、变量选择 JS、Spearman 秩距离与滚动信号；
- `statistical_classifier.py`：论文命名下的 PA、CA 与 CP 规则；
- `adaptive_state.py`：自适应状态转换；
- `pipeline.py`：对已有 TFT 推理结果执行统计检测。

整理版用于阅读和轻量测试；复现论文结果时以 `notebook_v7_*` 文件为准。

## 数据边界

原始微博文本、图片、用户信息、训练权重和完整中间数据不进入公开仓库。仓库提供数据文件名、字段契约和运行顺序；论文数值结果的完整复跑需要作者本地数据和 GPU 环境。

本地冒烟验证已使用原论文环境中的 `pretrained_planPC` 成功恢复 `model_comment_pc1`、`model_post_pc2`、两套 `TimeSeriesDataSet`、`TargetBuilder` 与 `TFTDataProcessor`。验证环境为 PyTorch `2.6.0+cu126`、pytorch-forecasting `1.1.0`；未重复执行 300 轮训练和全年滚动检测。
