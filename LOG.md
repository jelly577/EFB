# EFB B 同学工作日志

本文记录仓库的实现、实验和交付状态。敏感信息（SSH 密码、令牌、服务器凭据）不写入日志。

## 2026-07-18：仓库初始化

- 创建 Python 项目 `efficient-reasoning-sampling`，配置 `generation`、`evaluation`、`sampling` 和 `tests` 包。
- 实现 Hugging Face 后端、条件 log-likelihood 计算和从 token 位置重采样后缀。
- 实现固定轮次 Power Sampling、基础开销统计、MATH-500 运行入口和 `\boxed{}` 答案提取。
- 增加无需 GPU 的离线测试。
- Git 提交：`9038ea5 Initialize efficient reasoning sampling project`。

## 2026-07-18：自适应停止与数学验证

- 实现自适应停止：达到最小步数后，若连续 `patience` 次 likelihood 增益不超过阈值，则提前结束。
- 增加 `saved_attempts`、`stopped_early`、接受率、拒绝 token 比例和逐步 `trace`。
- 实现 token surprise 难点评分和三状态 proposal ratio 玩具验证。
- 增加结果汇总与对比绘图脚本。
- Git 提交：`0b5f4eb Add adaptive sampling experiments`。

## 2026-07-18：重采样正确性修复

- 发现短候选可能因累计 log-likelihood 数值更高而被错误偏好。
- 将切分位置限制在末尾 `suffix_max_new_tokens` 窗口内。
- 强制候选后缀与原后缀等长，避免重采样预算直接截断答案。
- Git 提交：`da4815f Preserve suffix length during resampling`。
- Git 提交：`55c5ffd Ignore editable install metadata`。

## 2026-07-18：配对可复现比较

- 每道题使用 `seed + 题号` 重置生成和位置选择随机数。
- fixed 与 adaptive 对同一道题共享初始文本，并在 adaptive 停止前共享提议序列。
- 汇总中加入普通初始生成，形成 ordinary / fixed / adaptive 三方比较。
- Git 提交：`a74f804 Make method comparisons reproducible`。
- Git 提交：`b7c276d Compare against ordinary generation`。

## 2026-07-18：GPU 校正试跑

- 设备：RTX 4090 24GB。
- 模型：`Qwen/Qwen2.5-Math-7B`，BF16。
- 数据：MATH-500 第 1 题。
- 固定版：准确率 100%，1095 total token，28.529 秒，执行 8 次。
- 自适应版：准确率 100%，653 total token，16.941 秒，执行 3 次并节省 5 次。
- 产物：`results/fixed_corrected.jsonl`、`results/adaptive_corrected.jsonl`。
- 结论：等长后缀修复后的单题流程可以跑通，但单题结果不具统计意义。

## 2026-07-18：5 题 paired GPU 冒烟实验

### 配置

- 数据：MATH-500 前 5 题。
- 模型：`Qwen/Qwen2.5-Math-7B`，BF16。
- GPU：RTX 4090 24GB。
- 初始最大输出：1024 token；重采样后缀最大窗口：128 token。
- 最大重采样次数：8；随机种子：42，逐题使用 `42 + 题号`。
- 自适应参数：`min_steps=2`、`patience=2`、`gain_threshold=0.01`。
- 配对检查：fixed 与 adaptive 的初始文本 5/5 一致。

### 汇总结果

| 指标 | 普通初始生成 | 固定 8 次 | 自适应停止 |
| --- | ---: | ---: | ---: |
| 题数 | 5 | 5 | 5 |
| 准确率 | 60% | 40% | 40% |
| 平均初始 token | 499.4 | 499.4 | 499.4 |
| 平均总 token | 499.4 | 1082.0 | 704.4 |
| 平均被拒 token | — | 289.2 | 91.6 |
| 平均接受率 | — | 57.5% | 52.7% |
| 平均耗时/题 | 未单独记录 | 27.419 秒 | 17.913 秒 |
| 平均节省尝试 | — | 0 | 5.2/8 |
| 提前停止率 | — | 0% | 100% |

### 当前解释

- 自适应相对固定版平均节省 34.9% total token 和 34.7% 运行时间。
- 自适应平均被拒 token 减少约 68.3%，说明提前停止减少了无效候选生成。
- 5/5 道题都提前停止，当前阈值可能偏激进，需要扩大样本并调参。
- 普通初始生成准确率 60%，重采样后准确率 40%；当前结果没有显示准确率收益。
- 存在初始答案正确但重采样最终答案错误的样例，必须在扩大实验前定位原因。
- 样本量只有 5，所有百分比仅作为流程冒烟信号，不能作为论文结论。

### 产物

- `results/fixed_5_paired.jsonl`
- `results/adaptive_5_paired.jsonl`
- `results/five_paired_summary.json`
- `figures/five_paired_comparison.png`
- `output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf`

## 2026-07-18：验证与交付

- 本地运行 `python -m unittest discover -s tests -v`：11 项测试全部通过，用时 0.020 秒。
- 生成 3 页中文 GPU 冒烟实验 PDF，并完成逐页渲染检查。
- PDF 元数据作者：`jelly577`；文件大小约 14.65 MB。
- 当前 Git 远端同步到 `b7c276d`。
- `results/*` 和 `figures/*` 默认被 `.gitignore` 忽略；PDF 与本次文档更新尚未提交到 GitHub。
- AutoDL 的精确基础镜像名称和环境包版本没有进入结果记录，后续正式实验应保存 `nvidia-smi`、Python、PyTorch、CUDA、Transformers 和数据集版本。

## 当前未完成事项

- [ ] 人工复核 5 道题的最终答案和数学等价判分。
- [ ] 分析初始正确但重采样后错误的样例。
- [ ] 在 5 题上调节自适应停止阈值，避免过早停止。
- [ ] 确认论文中的完整 proposal distribution 与 Metropolis-Hastings 接受概率。
- [ ] 将难点优先选位安全接入真实模型主实验。
- [ ] 运行 100 题 paired 实验并报告不确定性。
- [ ] 100 题结果稳定后再决定是否运行 500 题全量实验。
- [ ] 固化服务器软件版本和可复现环境文件。
- [ ] 决定哪些结果、图表和 PDF 需要纳入 Git，并提交本次更新。

## 状态结论

目前完成的是 **5 题 GPU 冒烟实验，不是 100 题或 500 题完整实验**。代码流水线、配对复现、自适应停止、统计与报告均已跑通；自适应版本在小样本上节省约三分之一计算，但准确率没有改善，因此下一阶段重点是保证答案质量和验证接受公式，而不是立即扩大到全量数据。
