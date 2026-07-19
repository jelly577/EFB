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

## 2026-07-18：修复版 α=4 的 5 题 GPU 复跑（服务器时间 07-19 CST）

### 配置与环境

- 同一 AutoDL RTX 4090 24GB 容器；远程执行由本机通过 SSH 驱动，nohup 后台运行。
- Python 3.12.3、PyTorch 2.5.1+cu124、Transformers 5.14.1、Datasets 5.0.0；`nvidia-smi` 与包版本已存入 `results/env_nvidia_smi.txt`、`results/env_versions.txt`（此前待办已完成）。
- 参数：`alpha=4.0`、`steps=8`、初始上限 2048、后缀窗口 128、seed 42 逐题配对；自适应 `min_steps=2`、`patience=2`、`rejection_patience=4`、`gain_threshold=0.01`。
- 代码版本：`465115d`。产物：`results/fixed_5_alpha4.jsonl`、`results/adaptive_5_alpha4.jsonl`、`results/alpha4_summary.json`、`figures/alpha4_comparison.png`。
- 报告：`output/pdf/EFB_B同学_α4修复版复跑报告.pdf`（3 页中文，生成脚本 `tmp/pdfs/build_alpha4_report.py`，逐页渲染检查通过）。
- 人工复核材料：`results/alpha4_answer_review.md`，汇集 5 题题目、官方答案、初始/Fixed/Adaptive 答案与完整解答文本。

### 汇总结果

| 指标 | 固定 8 次 | 自适应停止 |
| --- | ---: | ---: |
| 准确率 | **60%**（=初始准确率） | **60%**（=初始准确率） |
| 平均总 token | 1217.2 | 882.2 |
| 平均被拒 token | 495.8 | 307.6 |
| 平均接受率 | 42.5% | 30.7% |
| 平均耗时/题 | 32.07 秒 | 28.11 秒 |
| 平均执行次数 | 8 | 3.8（节省 4.2/8） |
| 提前停止率 | 0% | 100% |

### 验收结论

- **q1 翻转已消除**：初始与最终答案均为 `p - q`；护栏在 fixed 上拦下 4 个丢失 `\boxed{}` 的 proposal（两方法各拦 6 个/全部 5 题），证明等长截断问题已被根治。
- **无新增 `final_answer=None`**；重采样不再破坏任何初始正确的答案，两方法准确率回到 60%，与普通初始生成持平。
- **修正早前诊断**：q3 的初始生成只有 136 token 且自然终止、从未产生 `\boxed{}`（`initial_truncated=False`），并非此前记录的"顶满 1024 截断"；q3、q4 属于初始生成本身的错误，重采样在当前窗口内未能修复。
- 自适应仍 5/5 提前停止，但平均执行 3.8 次（修复前 2.8），且停止原因符合设计：q1、q4 由连续 4 次拒绝触发，q2、q3 由接受步低增益触发。
- α=4 使接受更挑剔（接受率从 57.5%/52.7% 降至 42.5%/30.7%），相应地自适应节省幅度从 34.9% 收窄到 **27.5% token、12.4% 时间**，被拒 token 减少 38%。
- 5 题样本仍只是冒烟信号；α=4 未在小样本上带来准确率提升（也未损害），是否有增益需 100 题验证。

## 2026-07-18：q4/q5 错误原因人工复核

- **第 4 题（196 的因数个数，官方答案 9）是判分假阴性**：模型初始解答的数学完全正确（196 = 2²×7²，(2+1)(2+1)=9，并明确写出 "196 has 9 positive whole-number divisors"），但全程没有写 `\boxed{}`，提取器返回 None 被判错。**5 题的真实初始正确率是 4/5=80%，自动判分的 60% 低估了**。
- 第 4 题还暴露一个次生问题：无 `\boxed{}` 的文本不受答案护栏保护，fixed 版最终接受了一个顶满 128 token 预算、句子中断的候选（结尾停在公式中间）。候选没有自然 EOS 终止时其概率质量并不完整，后续可考虑"无自然终止即拒绝"的完整性护栏。
- **第 5 题（越野跑最快学生，官方答案 Evelyn）是模型真实错误，判分正确**：模型把图上五个点的数据全部读对（Evelyn 4.5 英里/1.25 小时 = 3.6 mph 应为最大），但走了"写 Python 代码 + 伪造 ```output 块"的路线，伪造的输出宣称 max 是 Carla（1.22 mph，实为距离最高点而非速度最快）。根因是无代码执行器时 Qwen2.5-Math 的 tool-integrated 风格伪造运行结果，叠加"距离最高＝最快"的混淆。
- 两方法重采样都没能修复第 5 题：伪造 output 的延续在上下文中是高似然的，α=4 只会偏好它；错误的源头（读图后的推断方式）需要更靠前的切分点才可能改写。
- 待办新增：判分兜底或答案补写以消除格式假阴性；候选完整性护栏；考虑 Qwen2.5-Math 官方 CoT prompt 抑制伪代码执行行为。

## 2026-07-18：完整性护栏与官方 CoT prompt

- **完整性护栏**：`GeneratedText` 新增 `ended_naturally` 字段；候选后缀顶满 token 预算且末 token 不是停止符时判定为截断，直接拒绝（其概率质量不完整，不得与完整序列比较似然）。计入 `incomplete_rejections`，trace 增加 `incomplete_rejected` 字段。修复 q4 复跑中 fixed 版接受"停在公式中间"候选的问题。
- **官方 CoT prompt**：`run_math500` 弃用自拟指令模板，改用 Qwen2.5-Math 官方 system prompt（"Please reason step by step, and put your final answer within \boxed{}."）经 `apply_chat_template` 构建 ChatML 格式输入；结果记录新增 `prompt_style: qwen-cot-chat`。
- **停止符处理**：base 模型 `eos_token` 是 `<|endoftext|>`(151643)，而 ChatML 对话以 `<|im_end|>`(151645) 结束；后端把两者都加入 `generate` 的 `eos_token_id`，避免模板化后生成无法停止。已在服务器上用缓存 tokenizer 验证模板存在且格式正确。
- 验证：离线测试 18 项全部通过（新增完整性护栏测试）。
- 影响：prompt 变更使旧结果不可直接对比，5 题需要重跑作为新基线。

## 当前未完成事项

- [x] 分析初始正确但重采样后错误的样例（q1，等长截断丢失 `\boxed{}`，已修复）。
- [x] 修正接受公式：实现 p^α 目标 + proposal 修正的 `(α−1)Δ` 规则，并用变长玩具分布验证。
- [x] 修正自适应停止判据：接受步增益与拒绝流分开统计。
- [x] GPU 重跑 5 题 paired（α=4）：q1 不再翻转、无 `\boxed{}` 丢失、准确率回到 60%。
- [x] 固化服务器软件版本（`results/env_versions.txt`、`results/env_nvidia_smi.txt`）。
- [x] 人工复核 q4/q5：q4 为判分假阴性（数学正确、缺 `\boxed{}`），q5 为模型真实错误（伪造代码输出）。
- [x] 官方 CoT prompt：已切换到 Qwen2.5-Math chat template（待 5 题复跑确认格式假阴性消除）。
- [x] 候选完整性护栏：顶满预算且无自然 EOS 的候选直接拒绝。
- [ ] 答案补写兜底（可开关）：无 `\boxed{}` 时贪心补写答案句，触发次数计入报告。
- [ ] 新 prompt 下重跑 5 题 paired，建立新基线并确认 q4 格式假阴性消除。
- [ ] 与论文原文逐项核对 proposal distribution 与接受概率（玩具验证已通过，等论文原文最终确认）。
- [ ] 在 5–20 题上粗调 `alpha`、`gain_threshold`、`patience`、`rejection_patience`。
- [ ] 将难点优先选位安全接入真实模型主实验（需把选位概率纳入 proposal ratio）。
- [ ] 运行 100 题 paired 实验并报告不确定性（bootstrap 置信区间 / McNemar）。
- [ ] 100 题结果稳定后再决定是否运行 500 题全量实验。
- [ ] 决定哪些结果、图表和 PDF 需要纳入 Git，并提交本次更新。

## 状态结论

目前完成的是 **5 题 GPU 冒烟实验 + 根因修复 + α=4 修复版复跑验证，不是 100 题或 500 题完整实验**。复跑证实：准确率下降由等长后缀截断引起，修复后两方法准确率回到 60%（与初始生成持平，不再破坏正确答案）；自适应停止在 α=4 下仍节省 27.5% token 和 12.4% 时间。方法流程已站稳，下一步是在 5–20 题上调参并推进 100 题 paired 实验，验证 α>1 是否带来准确率增益。
