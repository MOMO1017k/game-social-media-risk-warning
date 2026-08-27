# 第 2 阶段代码审查：多模态特征工程

## 审查结论

论文最终使用的 19 个特征在当前代码中全部存在，没有缺少概念指标。但 `PublicOpinionMonitor.aggregate()` 当前返回 24 列，混入了 5 个研究期候选/诊断特征；部分主流程、依赖和函数存在阻塞性问题，因此本批代码只作为原始实现归档，不能直接作为 GitHub 可运行版本。

## 论文最终 19 个特征

| 类别 | 正式输出列 |
| --- | --- |
| 规模 | `total_volume_post`, `total_volume_comment` |
| 参与集中度 | `gini_post`, `gini_comment` |
| 转发结构 | `retweet_ratio_post` |
| 情绪符号烈度 | `senti_symbol_post`, `senti_symbol_comment` |
| 文本压缩比 | `comp_ratio_post`, `comp_ratio_comment` |
| 短文本条数 | `total_short_post`, `total_short_comment` |
| 长文本条数 | `total_long_post`, `total_long_comment` |
| 负面占比 | `neg_ratio_post`, `neg_ratio_comment` |
| 相邻窗口语义距离 | `semantic_shift_post`, `semantic_shift_comment` |
| 视觉特征 | `vis_abs_redundancy_post`, `vis_concentration_post` |

代码内部仍使用 `origin_ratio`，Notebook 保存结果时才将其改名为 `retweet_ratio_post`。正式版本应直接输出最终列名。

## 当前多出的 5 列

| 候选列 | 建议 |
| --- | --- |
| `unique_users_post` | 作为诊断列或候选实验特征保留，不进入论文最终 19 列 |
| `unique_users_comment` | 同上 |
| `total_medium_post` | 作为文本长度构成校验列保留，不进入最终模型 |
| `total_medium_comment` | 同上 |
| `semantic_shift_cross` | 作为广场—评论跨源差异实验特征保留，不进入最终模型 |

部分 Notebook 单元还会增加 `total_volume`，因此研究期文件可能出现 25 列。正式代码需要提供固定的 `FINAL_FEATURE_COLUMNS` 与 `OPTIONAL_DIAGNOSTIC_COLUMNS`，禁止依赖 Notebook 临时改名或加列。

## 阻塞性问题

1. `PublicOpinionMonitor.process_data()` 完成清洗和情感打分后没有返回 DataFrame，Notebook 中接收其结果时会得到 `None`；
2. `monitor_core.py` 导入了不存在的 `utilis_visual`，但导入的两个函数实际未使用，应删除该依赖或补齐并统一实现；
3. Notebook 导入不存在的 `visualization.ReportVisualizer`，即使后续调用被注释，导入本身仍会阻止运行；
4. `batch_clean_text()` 对过短严格清洗文本只修改 `text_clean`，而后续有效性判断读取 `text_clean_strict`，会让本应过滤的短文本继续进入情感和长度计算；
5. `VisualImpactDetector.analyze()` 调用不存在的 `self.calc_metrics()`；当前正式聚合路径使用另一方法，因此该函数属于失效分支；
6. 情感模型路径、停用词目录和数据路径全部硬编码在原电脑上；SentenceTransformer 还固定使用 `cuda`，没有 CPU 回退；
7. 当前 `official_file` 参数没有进入最终 19 列，官方事件特征函数也没有接入 `aggregate()` 输出；需要明确它属于外生变量而非 19 项监测特征；
8. 视觉空窗口会被直接跳过并在合并后产生缺失值；语义空窗口则被记为 `0`，两者对“缺失”和“真实零值”的定义不一致；
9. Notebook 的历史输出来自较早代码版本，视觉列名与当前模块不同，不能作为当前实现已成功运行的证据。

## 明显冗余或位置不当

1. `main_preprocess_v4.3.ipynb` 中定义了 5 次 `verify_visual_impact`、2 个 `VisualRepetitionAnalyzer` 和 2 个 `VisualImpactDetector`，只应保留最终演示入口，其余版本放入研究记录或删除；
2. `user_dict.txt` 是 `nikki_dict.txt` 的子集，未发现独立用途，因此整理仓库只保留内容更完整的 `nikki_dict.txt`；
3. 两个词典都包含格式错误的 `开学季nz`，且 `nikki_dict.txt` 重复收录“四星”和“五星”；
4. `detection_features.py` 重复导入 `os`、`numpy`、`pandas` 和 `tqdm`，并混入未使用的绘图/调试依赖；
5. `utilis_preprocess.py` 中的 `perform_residual_diagnostics()` 属于后续时序分解与合成环节，不应放在文本预处理模块；
6. `version_config.py` 同时包含预处理词典、主题归因、实体配置和版本映射，职责过多；其中实体与主题配置应移到归因阶段；
7. `monitor_core.py` 中的归因辅助函数因缩进位置错误成为其他方法内部、且位于 `return` 之后，当前不可达；
8. `sent_series` 被计算但没有进入任何输出，可删除或明确为诊断量。

## 需要补充的内容

1. **特征注册表**：为 19 个正式列记录中文名、公式、数据源、单位、平滑参数、空窗口规则和取值范围；
2. **输入数据契约**：明确帖子与评论所需字段、类型、时区、去重主键和允许缺失值；
3. **配置系统**：将模型名称、模型缓存目录、停用词、时间粒度、阈值、设备和路径移出源代码；
4. **模型获取说明**：说明情感模型和句向量模型的来源、版本、许可证、哈希与离线缓存方式，不直接提交大型权重；
5. **停用词资源**：当前外部停用词文件夹缺失，需要提供可公开的词表或下载说明；
6. **测试**：至少为压缩比、符号烈度、Gini、负面占比、长度分层、语义空窗口和视觉聚类建立小型合成测试；
7. **质量字段**：建议输出每个窗口的原始条数、有效文本数、图片匹配率和缺失标记，避免把无数据误解释为正常值；
8. **可重复运行入口**：提供一个从脱敏样例输入到 19 列 CSV 的命令，并固定随机种子、模型版本和输出顺序；
9. **版本日历完整性**：当前实体配置覆盖多个版本，但 `VERSION_SCHEDULE` 只有 4 个时间段；归因阶段需要补齐或删除未使用的映射。

## 本轮归档处理

- 四个 Python 模块和较完整的 `nikki_dict.txt` 原样复制到 `scripts/02_feature_engineering/legacy/`；
- 探索 Notebook 已清除 229 个历史输出和全部执行计数；
- 冗余的 `user_dict.txt` 未复制，原始目录仍保留该文件；
- 本轮没有修改算法公式，等待后续统一重构。
