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

## 2026-07-18：准确率下降诊断与 Power 接受规则重写

### 诊断（逐题分析 5 题 paired 结果）

- q1 是唯一"初始对→最终错"的翻转样例。根因：`resample_suffix` 强制新后缀与原后缀严格等长（`min_new_tokens = max_new_tokens`），接受的重写在句中被硬截断（final text 以 "So the" 结尾），`\boxed{}` 丢失，判分变 None。等长约束是上一轮修长度偏置的补丁，属于工程 artifact，不是方法失效。
- q3 初始生成顶满 1024 token 未写完，初始就没有答案，与重采样无关。
- q4 是模型真实错误（Carla vs Evelyn）。
- 另发现自适应停止用 `abs(增益)≤阈值`，被拒步增益恰为 0，`patience=2` 实际等价于"连续 2 次拒绝就停"，解释了 5/5 全部提前停。

### 修复

- **接受规则**：目标分布 p^α、proposal 为模型自身重采样时，接受概率化简为 `(α−1)·(log p′ − log p)`；`accepts_metropolis` 增加 `alpha` 参数（默认 4.0），α=1 时全接受（即从 p 采样，符合理论）。proposal 修正使长度偏置从原理上消除。
- **撤销等长约束**：`resample_suffix` 允许自然 EOS 终止，只限 `max_new_tokens` 上限。
- **答案护栏**：当前文本有 `\boxed{}` 而 proposal 丢失时直接拒绝，计入 `answer_guard_rejections`，trace 增加 `answer_guard_rejected` 字段。
- **自适应停止判据**：patience 窗口只统计**接受步**的增益；拒绝只通过独立的 `rejection_patience`（默认 4，连续拒绝阈值）触发停止。
- **实验入口**：新增 `--alpha`、`--rejection-patience`；`--initial-max-new-tokens` 默认提高到 2048；记录 `initial_truncated` 标志。
- **玩具验证**：新增变长序列玩具自回归模型（长度 2–5，含 EOS 概率），穷举精确分布后验证 `(α−1)Δ` 规则收敛到 p^α（TV≈0.005），而漏掉 proposal 修正的朴素 `αΔ` 规则偏差明显（TV≈0.081）。这是"论文公式核对"的可执行形式。

### 验证

- `python -m unittest discover -s tests -v`：17 项测试全部通过（新增 α 边界、答案护栏、拒绝流停止、power 玩具验证等 6 项）。
- `python -m sampling.toy_validation`：三状态与变长 power 两组验证均通过。
- GPU 5 题 paired 重跑尚未执行，为下一步动作。

## 当前未完成事项

- [x] 分析初始正确但重采样后错误的样例（q1，等长截断丢失 `\boxed{}`，已修复）。
- [x] 修正接受公式：实现 p^α 目标 + proposal 修正的 `(α−1)Δ` 规则，并用变长玩具分布验证。
- [x] 修正自适应停止判据：接受步增益与拒绝流分开统计。
- [ ] 人工复核 5 道题的最终答案和数学等价判分。
- [ ] GPU 重跑 5 题 paired（α=4），确认 q1 不再翻转、无 `\boxed{}` 丢失、提前停止率合理。
- [ ] 与论文原文逐项核对 proposal distribution 与接受概率（玩具验证已通过，等论文原文最终确认）。
- [ ] 在 5–20 题上粗调 `alpha`、`gain_threshold`、`patience`、`rejection_patience`。
- [ ] 将难点优先选位安全接入真实模型主实验（需把选位概率纳入 proposal ratio）。
- [ ] 运行 100 题 paired 实验并报告不确定性（bootstrap 置信区间 / McNemar）。
- [ ] 100 题结果稳定后再决定是否运行 500 题全量实验。
- [ ] 固化服务器软件版本和可复现环境文件。
- [ ] 决定哪些结果、图表和 PDF 需要纳入 Git，并提交本次更新。

## 状态结论

目前完成的是 **5 题 GPU 冒烟实验 + 准确率下降的根因修复，不是 100 题或 500 题完整实验**。逐题诊断表明 60%→40% 的准确率下降来自等长后缀截断这一工程 bug，而非方法失效；现已撤销等长约束、实现带 α 的正确 Metropolis-Hastings 接受规则（玩具分布上收敛到 p^α）、增加 `\boxed{}` 答案护栏并修正自适应停止判据。下一步是在 GPU 上重跑 5 题 paired 验证修复，然后调参并推进 100 题实验。
