# 第 1 阶段代码盘点：社交媒体数据采集

## 文件映射

| 整理后文件 | 原始文件 | 作用 | 当前定位 |
| --- | --- | --- | --- |
| `notebooks/01_data_collection/weibo/01_keyword_search_posts.ipynb` | `extract_weibo_posts.ipynb` | 按关键词、日期和小时切片抓取微博搜索结果，解析正文、互动量、作者、转发关系及媒体链接 | 论文主链路 |
| `notebooks/01_data_collection/weibo/02_official_account_posts.ipynb` | `extract_weibo_officialposts.ipynb` | 按账号 UID 与日期区间抓取官方微博，支持原创/转发筛选 | 论文主链路 |
| `notebooks/01_data_collection/weibo/03_official_account_comments.ipynb` | `extract_weibo_comments.ipynb` | 抓取官方微博及评论，记录评论互动、用户字段和相对发帖时间标签 | 论文主链路 |
| `scripts/01_data_collection/auth/get_weibo_cookie.py` | `get_weibo_cookie.py` | 通过浏览器手动登录并在本地保存微博 Cookie | 辅助工具 |
| `experiments/01_data_collection/bilibili/get_bilibili_cookie.py` | `get_bilibili_cookie.py` | Bilibili 登录实验 | 未进入论文主链路 |

真实的 `weibo_cookies.json` 未复制到整理仓库。

## 代码读取结论

### 关键词微博采集

- 使用 `requests` 和 `BeautifulSoup` 解析微博高级搜索页面；
- 将日期范围进一步拆分为逐日、逐小时任务，以降低单次搜索结果的分页上限影响；
- 采集正文、时间、转评赞、作者、原微博关系、图片链接和视频信息；
- 通过微博 ID 去重，并区分原创与转发内容。

### 官方账号与评论采集

- 使用微博页面接口读取指定 UID 的历史微博；
- 根据起止日期过滤内容，并单独处理置顶微博；
- 评论采集使用游标翻页，支持热度或时间排序；
- 将评论时间映射为“当日、1—3 天、4—7 天”等相对时间标签；
- 记录官方账号、微博、评论和用户层面的基础字段。

## 公开前需要重构的问题

1. 三个 Notebook 重复实现了 Session 和请求头构造逻辑，需要提取为公共模块；
2. Cookie 文件名、账号 UID、关键词、日期和输出路径目前写在代码中，需要迁移到配置文件或命令行参数；
3. 原代码包含 Windows 绝对路径，换一台电脑无法直接运行；
4. 网络错误重试缺少统一的次数上限、退避策略和日志记录；
5. 页面结构与非公开接口可能变化，需要增加解析失败检查和小规模离线测试；
6. Notebook 中存在按月份重复执行的单元，应改为单一任务调度入口；
7. 采集结果包含用户标识和原始内容，公开样例需要脱敏和最小化；
8. Bilibili 登录实验仍沿用微博 Cookie 文件名和校验字段，当前不可作为正式采集工具。

## 本轮已完成的安全处理

- 原始目录保持不变；
- 整理后的 Notebook 已清除所有历史运行输出和执行计数；
- 真实 Cookie 文件没有进入整理目录；
- `.gitignore` 已阻止 Cookie、原始数据、日志和本地环境文件被误提交。
