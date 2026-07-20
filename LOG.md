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

## 2026-07-18：新 prompt + 完整性护栏 5 题复跑（v2）与 q3/q4/q5 分析

- 服务器同步到 `8eda4df` 后复跑 5 题 paired（α=4、steps=8、seed=42+题号），输出 `results/fixed_5_v2.jsonl`、`results/adaptive_5_v2.jsonl`、`results/v2_summary.json`。全程仅 6 分钟（旧 prompt 下约 30 分钟）：官方 prompt 使生成大幅变短（平均初始 645 token vs 旧版数千）。
- 汇总：两方法准确率 40%（=初始准确率，重采样仍不破坏正确答案）；自适应节省 15.5% token、18.3% 时间，提前停止率 60%，平均节省 2.2/8 步。
- **新发现的判分失效模式：答完不停，自编新题**。q3、q4 的真实解答都是**对的**（q3 中段写出 `\boxed{\frac{14}{3}}`，q4 第一个框就是 `\boxed{9}`），但模型答完后继续生成一串自编题目并自问自答（q4 尾部连续十几个 "What is the sum of the divisors of ..."），判分取最后一个 `\boxed{}`，取到了自编题的答案（q3→8，q4→8101）。
- **根因：Qwen2.5-Math-7B 是 base 模型，不是 Instruct**。官方零样本 CoT chat prompt 是为 Instruct 版设计的协议；base 模型虽然 tokenizer 带 chat template，但没有被训练成在 `<|im_end|>` 处停止，于是持续续写（还出现 "Shay" 之类退化 token）。base 模型的官方评测协议是 few-shot CoT。上一轮"验证了模板存在"只验证了格式，没验证行为——记为教训。
- **q5 仍是真实模型错误**：数据点读对（Evelyn 4.5/1.25=3.6 应为最大），但重演"伪造 ```output 块"路线，伪造输出宣称 Carla；重采样把它翻成 Angela（伪造输出这次算的是 Angela 的真实速度 0.206，但比较逻辑仍错），错上加错但不更接近正确——与上一轮结论一致：错误源头在切分窗口之前，重采样无法修复。
- **真实数学正确率 4/5=80%**（q1、q2、q3、q4 解答正确，q5 错误），自动判分 40% 严重低估——但这次的假阴性不是"缺 `\boxed{}`"而是"多余 `\boxed{}`"，答案补写兜底解决不了，需要先解决"停不下来"。
- 完整性护栏首战活跃：q3、q4 各拦下 7/8 个未自然终止的候选——文本一旦进入"题目接龙"模式，128 token 的重采样后缀几乎不可能自然收尾，护栏正确地阻止了这些不完整候选进入链，但也使链近乎停滞（q4 接受 0 个）。
- **结论与去向**：判分失效的根因是"模型-协议不匹配"，不是采样方法问题。首选修复：换 `Qwen/Qwen2.5-Math-7B-Instruct`（与官方 zero-shot CoT prompt 匹配、会在 `<|im_end|>` 停止）；备选：留 base 模型改 few-shot 官方协议。待定后重跑建立基线。

## 2026-07-19：换用 Instruct 模型，5 题基线全绿（v3）

- 换用 `Qwen/Qwen2.5-Math-7B-Instruct`（`run_math500` 默认模型同步更新），配置与 v2 完全一致（α=4、steps=8、seed=42+题号、官方 CoT chat prompt），复跑输出 `results/fixed_5_instruct.jsonl`、`results/adaptive_5_instruct.jsonl`、`results/instruct_summary.json`。
- **判分恢复可信**：10 条记录每条初始文本恰好 1 个 `\boxed{}`，"答完不停、自编新题"的失效模式完全消失——确认 v2 的问题就是 base 模型与 chat 协议不匹配。
- **准确率 5/5=100%**（初始 = fixed = adaptive）：q3、q4 判分与真实一致；**q5 首次做对**——Instruct 版走纯 CoT 逐个计算速度（Evelyn 4.5/1.25=3.6 最大），不再伪造 ```output 块，印证了"协议匹配可抑制伪代码执行行为"的预判。
- 链健康度恢复：接受率 70%/74.3%（v2 仅 ~20%），完整性护栏仅偶发触发（fixed 全程 3 次），答案护栏 0 次——上游停止行为正常后两道护栏回到"兜底"角色。
- 自适应：3/5 提前停止（q4 只用 2 步、q3 用 4 步、q2 用 7 步），平均省 2.2/8 步、14.6% token、13.1% 时间；q1、q5 未提前停（接受步增益未收敛），行为符合设计。
- **新的实验局限**：这 5 题对 Instruct 版太简单（初始就 100%），没有余量展示重采样的准确率增益；α 调参与"Fixed > 普通"的验证必须到更大、更难的题集上做（100 题 paired 是下一步）。
- 产出 v3 基线报告：`output/pdf/EFB_B同学_Instruct基线报告.pdf`（3 页：结论与对比图 / 汇总与 v2→v3 对照与逐题表 / 诊断故事与下一步），配图 `figures/instruct_comparison.png`。

## 2026-07-19：难度分层抽题（100 题主实验第一步）

- 新增 `generation/selection.py`：`parse_levels`（解析 `--levels "4,5"`，校验 1–5 范围）与 `select_problems`（按数据集顺序过滤难度并截取前 N 题，保持确定性与可追溯）。
- `run_math500` 新增 `--levels` 参数；结果记录新增 `dataset_index`（原始数据集下标）与 `level`（难度）字段。不加 `--levels` 时行为与之前完全一致（取前 N 题）。
- 配对种子仍按筛选后的位置（`seed + 位置`）设置，fixed/adaptive 共享同一题序即可保证 paired。
- 新增 `tests/test_selection.py` 7 项；测试合计 **25/25 通过**。
- 动机：前 5 题对 Instruct 无区分度（初始即 100%），需要初始正确率 40–70% 的题集来观察重采样增益。计划先跑 MATH-500 level 4–5 的前 20 题 paired（`run_v4.sh` 已备好，输出 `*_20_hard.jsonl`）。
- 遗留：AutoDL 实例当前关机，SSH 连不上；待开机后启动 20 题实验。

## 2026-07-19：20 题难题 paired 实验（v4，level 4–5）

- 服务器同步 `71331b3` 后跑 MATH-500 level 4–5 前 20 题 paired（Instruct、α=4、steps=8、seed=42+位置），全程约 24 分钟。输出 `results/fixed_20_hard.jsonl`、`results/adaptive_20_hard.jsonl`、`results/hard20_summary.json`。
- **题集区分度达标**：初始正确率 50%（10/20），正好落在目标 40–70% 区间——这个题集适合做增益实验。
- **零翻转**：fixed 与 adaptive 均无"对→错"（安全性三连胜）也无"错→对"（尚无增益正例）。两方法准确率 = 初始 = 50%。q15 的答案在 fixed 里从 81/208 变成 81/215（错→另一个错），说明链在动但没到正确答案盆地。
- **自适应收益在难题上更明显**：90% 提前停止，平均省 3.95/8 步、**21.3% token、21.2% 时间**，被拒 token 少 48.5%（191.9 vs 372.8）；接受率 51%/56% 属健康区间。
- **新问题：2/20 题初始生成退化成乱码**（q17 ds=41、q18 ds=43）：生成陷入垃圾 token 循环（"Leone"/表情符号等）直到顶满 2048 截断，无 `\boxed{}`；后续候选同样退化，完整性护栏 8/8 全拦（行为正确）。这两题对当前实验是死重。
- **对"为何没有救回"的判断**：错误通常发生在推理前中段，而当前设计只在**末尾 128 token 窗口**内切分重采样（项目为省算力的设定）；α=4 只会把尾部措辞往高似然方向抛光，够不到上游错误。要看到准确率增益，下一步应优先实验：扩大重采样窗口（如 512）或放开全序列切分，其次对比 α=2/4。

## 2026-07-20：扩大重采样窗口到 512（v5），首个"错→对"出现

- 在同一批 20 道 level 4–5 难题上复跑 paired，唯一改动：`--suffix-max-new-tokens 128 → 512`（其余与 v4 完全一致：Instruct、α=4、steps=8、seed=42+位置）。全程约 46 分钟。输出 `results/fixed_20_w512.jsonl`、`results/adaptive_20_w512.jsonl`、`results/w512_summary.json`。
- **首个救回正例（错→对）**：q14（dataset_index=33，level 4）初始答 `81/208`（错），fixed 链在第 5 步接受了一个 Δlogp=+4.38 的候选，最终答案翻成 `243/625`（正确）。fixed 准确率 **50% → 55%**，方法首次在真实题目上展示准确率增益。"窗口太小够不到上游错误"的假设得到方向性验证。
- **零"对→错"**：两种模式安全底线四连胜（v1修复后/v3/v4/v5）。
- **自适应恰好错过了这次救回**：q14 前 4 步全被拒绝，adaptive 的 `rejection_patience=4` 在第 4 步触发停止——救回发生在第 5 步。adaptive 准确率停在 50%。这是"提前停止牺牲增益"的第一个具体案例：省算力（本轮仍省 19.5% token / 19.5% 时间）与抓住稀有救回之间存在真实张力，`rejection_patience` 成为关键调参对象。
- **窗口扩大的算力代价显著**：fixed 平均总 token 1410→3087（×2.2），时间 34.9s→76.2s（×2.2）；接受率从 51% 降到 38%（长后缀更难被整体接受）；被拒 token 占比 51%。提前停止率从 90% 降到 50%。
- 乱码退化仍是 2/20（q16 ds=41、q17 ds=43 顶满 2048 无 `\boxed{}`；当日曾误记为 1/20，后经 `final_answer is None` 全量复核更正）。退化题每题烧满 6144 token，占全程耗时 16–21%。退化与窗口无关，待复跑判定处理。
- **结论**：窗口 512 用 2.2 倍算力换来 +5pp（1 题）准确率；misfire 在 adaptive 的拒绝耐心上。下一步优先：`rejection_patience` 提高（如 6–8）复跑 adaptive，验证能否在保留大部分省算力的同时抓住救回；其次考虑窗口 128 vs 512 的 α=2 对照。

## 2026-07-20：rejection_patience=8 复跑 adaptive（v6），救回抓住了、省算力归零

- 只复跑 adaptive，唯一改动 `rejection_patience 4 → 8`（steps=8 下等于关闭"连续拒绝即停"通道），其余与 v5 一致；fixed 复用 v5 结果。约 24 分钟。输出 `results/adaptive_20_w512_rp8.jsonl`、`results/w512_rp8_summary.json`。
- **救回抓住了**：q14 走满 8 步，第 5 步照常接受救回候选，最终 243/625 正确。adaptive 准确率 50% → **55%**，与 fixed 对齐；仍零"对→错"。
- **但省算力几乎消失**：token 节省 19.5% → **4.6%**，时间 19.5% → 4.8%；提前停止率 50% → 10%（仅 2/20 由接受增益收敛触发）。说明在 512 窗口、38% 接受率的环境里，v5 adaptive 的节省几乎全部来自"连续拒绝即停"——正是砍掉救回的那个通道。
- **调参张力被完全量化**：rp=4 省 19.5% 但丢救回；rp=8 保救回但只省 4.6%。q14 的救回恰好在第 5 步，rp=5–6 理论上两全，但按单例调参就是过拟合——这个权衡曲线需要 100 题样本才能定形。
- **结论**："提前停止牺牲增益"从假设变成了带数字的权衡曲线（两个端点已测）。当前 20 题上 adaptive 的定位修正为：**大部分节省来自放弃低接受率的链，而这些链里偶尔藏着救回**。下一步不再继续在 20 题上调 rp，直接推进 100 题 paired（fixed + rp4 + rp8 三臂），用足够样本画出权衡曲线并做统计检验。

## 2026-07-20：退化探针 + 复跑判定实现（服务器迁移到新实例）

- **服务器迁移**：原实例（西北B区 007机）关机后 GPU 被占，无法开机；克隆到新实例（westc 区，RTX 4080 SUPER 32GB）。克隆未拷数据盘：代码经本地 `git archive` 直传重建，模型经 hf-mirror 重新下载（注意：新环境 `huggingface-cli` 已改名 `hf`）。旧实例结果文件本地均有备份。
- **退化探针实验**（`probe_degen.py`，只做初始生成）：对 v5 两道乱码退化题（ds=41、ds=43）各换 5 个新种子重新生成 + 各 2 次 `repetition_penalty=1.05` 生成，检验退化是"种子倒霉"还是"题目内在"。定时关机截断了完整日志回收，监控截到的部分结果已可定性：**退化是种子依赖的随机事件**——ds=41 换种子后有的正常收尾（`\boxed{17}`）有的仍退化，ds=43 观察到的 2 个新种子全部正常。完整计数待服务器再开机后从 `setup_probe.log` 回收。
- **据此选定复跑判定方案并实现**（弃用重复惩罚：它会改变所有题的 proposal 分布，破坏 MH 接受公式的 proposal 抵消；复跑只重掷链的起点，不参与 MH 核，数学干净）：
  - 新增 `generation/degeneration.py`：`is_degenerate`（顶满预算且无 `\boxed{}`，复用 `ended_naturally` 标志）与 `generate_initial_with_reroll`（换种子重生成，`problem_seed + 100003×retry`，默认最多 3 次）。
  - `run_math500` 新增 `--degeneration-retries`（默认 3）；初始生成移到 sampler 外做复跑判定后经 `initial_text` 传入（`sampler.run` 原生支持）。首发即正常的题随机流与旧版完全一致，历史 paired 结果可比。
  - 记录新增 `degeneration_retries` 与 `initial_degenerate`（重试用尽仍退化）字段；`initial_truncated` 改用生成时 token 数判定。
  - 新增 `tests/test_degeneration.py` 8 项；测试合计 **33/33 通过**。
- 下一步：服务器开机后回收探针完整日志，同步新代码，启动 100 题三臂实验（fixed + adaptive-rp4 + adaptive-rp8，512 窗口，约 6–7 小时）。

## 当前未完成事项

- [x] 分析初始正确但重采样后错误的样例（q1，等长截断丢失 `\boxed{}`，已修复）。
- [x] 修正接受公式：实现 p^α 目标 + proposal 修正的 `(α−1)Δ` 规则，并用变长玩具分布验证。
- [x] 修正自适应停止判据：接受步增益与拒绝流分开统计。
- [x] GPU 重跑 5 题 paired（α=4）：q1 不再翻转、无 `\boxed{}` 丢失、准确率回到 60%。
- [x] 固化服务器软件版本（`results/env_versions.txt`、`results/env_nvidia_smi.txt`）。
- [x] 人工复核 q4/q5：q4 为判分假阴性（数学正确、缺 `\boxed{}`），q5 为模型真实错误（伪造代码输出）。
- [x] 官方 CoT prompt：已切换到 Qwen2.5-Math chat template。
- [x] 候选完整性护栏：顶满预算且无自然 EOS 的候选直接拒绝（v2 复跑中正确拦截 q3/q4 各 7/8 个不完整候选）。
- [x] 新 prompt 下重跑 5 题 paired（v2）：暴露 base 模型对 chat prompt "答完不停、自编新题"的新失效模式，真实正确率 80% 但自动判分 40%。
- [x] 换用 `Qwen/Qwen2.5-Math-7B-Instruct` 重跑 5 题（v3）：判分恢复可信，5/5 全对，q5 伪造输出行为消失，新基线成立。
- [ ] 答案补写兜底（可开关）：无 `\boxed{}` 时贪心补写答案句，触发次数计入报告（注意：解决不了"多余 boxed"型假阴性）。
- [ ] 与论文原文逐项核对 proposal distribution 与接受概率（玩具验证已通过，等论文原文最终确认）。
- [x] 跑 MATH-500 level 4–5 前 20 题 paired（v4）：初始正确率 50% 达标；零翻转；自适应省 21.3% token。
- [x] 扩大重采样窗口到 512 并复跑难题集（v5）：首个"错→对"出现（q14，fixed 55% > 初始 50%），代价 2.2× token。
- [x] 提高 `rejection_patience` 到 8 复跑 adaptive（v6）：救回抓住（55% 对齐 fixed），但省算力 19.5%→4.6%——权衡两端点已量化。
- [ ] 在难题集上粗调 `alpha`（2 vs 4）、`gain_threshold`、`patience`。
- [x] 处理乱码退化生成：探针证实退化是种子依赖，实现复跑判定（`--degeneration-retries`，默认 3 次换种子重生成），弃用会污染 proposal 分布的重复惩罚。
- [ ] 服务器开机后回收 `setup_probe.log` 完整探针计数，验证复跑判定的实际成功率。
- [ ] 将难点优先选位安全接入真实模型主实验（需把选位概率纳入 proposal ratio）。
- [ ] 运行 100 题 paired 实验并报告不确定性（bootstrap 置信区间 / McNemar）。
- [ ] 100 题结果稳定后再决定是否运行 500 题全量实验。
- [ ] 决定哪些结果、图表和 PDF 需要纳入 Git，并提交本次更新。

## 状态结论

工程链路全部打通且判分可信（Instruct + 官方 CoT prompt），安全底线稳固（v3–v6 均零"对→错"）。**方法的两个核心主张各有数据支点，且它们的冲突已被量化**：① 重采样提准确率——512 窗口下首个"错→对"（fixed 与 adaptive-rp8 均 55% > 初始 50%）；② 自适应省算力——但 v5/v6 显示节省与救回来自同一处：rp=4 省 19.5% 丢救回，rp=8 保救回只省 4.6%。20 题样本无法定形这条权衡曲线。下一步：100 题 paired（fixed + rp4 + rp8 三臂）+ bootstrap / McNemar，同时处理乱码退化死重。
