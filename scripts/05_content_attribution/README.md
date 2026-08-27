# 内容归因阶段

本阶段用于在 TFT 检测到异常窗口后，结合中文文本主题、代表性图片和变量选择结果生成归因材料。

## 当前归档

- `legacy/TFT_topic_modeling.py`：原始主题建模模块，提供 `ClassTFIDF` 和 `TopicModeler`；
- `TopicModeler.run_topic_modeling()`：完成句向量编码、种子主题匹配、UMAP 降维、HDBSCAN 聚类、相似主题合并和可视化输出。
- `legacy/TFT_Attribution_engine.py`：文本、图片与 VSN 特征的归因路由，并生成图文报告；
- `../../notebooks/05_content_attribution/legacy/TFT_reasoning.ipynb`：归因分析和危机生命周期结果展示，历史运行输出已清除。

该模块依赖特征工程阶段的 `version_config.py` 与 `utilis_preprocess.py`。最终重构时将改为包内显式导入，避免依赖运行目录和 `sys.path`。

## 待重构

- 可复现的归因命令入口与最小示例。
- 将 `version_config.py` 与 `utilis_preprocess.py` 改为稳定的包内导入；
- 将 Notebook 中重复的清洗、制图和统计函数提取为可测试模块。
