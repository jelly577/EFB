# 推理效率研究 - B 同学工作区

本仓库实现《推理效率研究·零基础实操版》中 B 同学负责的 **Power Sampling 算力分配**。当前已经完成固定轮次基线、自适应停止、配对复现实验、离线测试和 5 题 GPU 冒烟实验；**尚未运行 100 题或 MATH-500 全量实验**。

完整实现与实验时间线见 [`LOG.md`](LOG.md)。

## 当前状态（2026-07-18）

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 项目初始化 | ✅ 完成 | Python 包、命令行入口、测试和分析脚本已建立 |
| 固定轮次 Power Sampling | ✅ 完成 | 每题最多进行 8 次后缀重采样 |
| 自适应停止 | ✅ 完成 | 连续低增益时提前停止，保留逐步 trace |
| 难点位置评分 | ✅ 完成基础模块 | 已按 token surprise 排序，尚未接入真实模型主实验 |
| 非均匀提议数学验证 | ✅ 完成玩具验证 | proposal ratio 修正已通过三状态分布测试 |
| Power (p^α) 接受规则 | ✅ 完成 | `(α−1)Δ` 规则 + 答案护栏，变长玩具分布上收敛到 p^α |
| 准确率下降根因修复 | ✅ 已 GPU 复跑验证 | q1 翻转由等长后缀截断引起；修复后准确率回到 60% |
| 5 题 α=4 修复版复跑 | ✅ 完成 | 两方法准确率 60%=初始准确率，自适应仍省 27.5% token |
| 完整性护栏 + 官方 CoT prompt | ✅ 完成 | 未自然终止的候选直接拒绝；prompt 走 chat template |
| 5 题新 prompt 复跑（v2） | ✅ 完成 | 暴露 base 模型"答完不停"失效模式，真实正确率 80%，判分 40% |
| Instruct 模型基线（v3） | ✅ 完成 | 换 `Qwen2.5-Math-7B-Instruct`：判分可信、5/5 全对、q5 伪造输出消失 |
| 固定版/自适应版配对复现 | ✅ 完成 | 同题共享初始生成，停止前共享提议序列 |
| 离线单元测试 | ✅ 17/17 通过 | 2026-07-18 本地重新验证（含新增 6 项） |
| 1 题 GPU 校正试跑 | ✅ 完成 | 用于确认等长后缀重采样修复 |
| 5 题 GPU 冒烟实验 | ✅ 完成 | RTX 4090 24GB，fixed 与 adaptive 各 5 题 |
| 100 题实验 | ⏳ 未运行 | 正确率下降已修复；待 5–20 题调参后运行 |
| 500 题完整实验 | ⏳ 未运行 | 当前没有全量结果，不能作最终统计结论 |
| 难点优先真实模型实验 | ⏳ 未运行 | 需先确认完整 Metropolis-Hastings 接受概率 |

GitHub 主分支目前同步到提交 `8eda4df`（`Add completeness guard and official CoT prompt`）。GPU 结果、图表和最新 PDF 均为本地产物，受 `.gitignore` 或尚未提交状态影响；本次文档更新也需要后续提交后才会同步到远端。

## 5 题 GPU 冒烟结果

实验使用 MATH-500 前 5 题、`Qwen/Qwen2.5-Math-7B`（BF16）、RTX 4090 24GB 和逐题配对随机种子。固定版与自适应版均设置最大 8 次重采样；自适应参数为 `min_steps=2`、`patience=2`、`gain_threshold=0.01`。

| 方法 | 准确率 | 平均总 token | 平均被拒 token | 接受率 | 平均耗时/题 | 提前停止率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通初始生成 | 60% | 499.4 | — | — | — | — |
| 固定 8 次 | 40% | 1082.0 | 289.2 | 57.5% | 27.42 秒 | 0% |
| 自适应停止 | 40% | 704.4 | 91.6 | 52.7% | 17.91 秒 | 100% |

相对固定 8 次，自适应版本：

- 平均节省 `377.6 token/题`，即 **34.9%**。
- 平均节省 `9.51 秒/题`，即 **34.7%**。
- 平均少做 `5.2/8` 次重采样，实际平均执行 2.8 次。
- 平均被拒 token 从 289.2 降至 91.6，减少约 **68.3%**。
- 5/5 道题均触发提前停止，且 paired 初始文本 5/5 一致。

这批结果只证明自适应停止在当前设置下明显减少计算，**不证明方法提升准确率**。样本仅 5 题；普通初始生成准确率为 60%，两种重采样方法均为 40%。

**逐题诊断（2026-07-18）已定位准确率下降的根因**：唯一的"初始对→最终错"翻转（q1）来自等长后缀约束——重写在句中被硬截断导致 `\boxed{}` 丢失，属工程 bug 而非方法失效（q3 是初始生成过早自然终止、从未产生 `\boxed{}`；q4 是模型真实错误）。

## 5 题 α=4 修复版复跑结果

修复代码（`465115d`）在同一 4090 容器复跑，配置同上但 `alpha=4.0`、初始上限 2048、自适应新增 `rejection_patience=4`。环境：Python 3.12.3、PyTorch 2.5.1+cu124、Transformers 5.14.1（详见 `results/env_versions.txt`）。

| 方法 | 准确率 | 平均总 token | 平均被拒 token | 接受率 | 平均耗时/题 | 提前停止率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通初始生成 | 60% | 499.4 | — | — | — | — |
| 固定 8 次 | **60%** | 1217.2 | 495.8 | 42.5% | 32.07 秒 | 0% |
| 自适应停止 | **60%** | 882.2 | 307.6 | 30.7% | 28.11 秒 | 100% |

复跑验收结论：

- **q1 不再翻转**：答案护栏拦下丢失 `\boxed{}` 的候选（每方法各 6 次），重采样不再破坏任何初始正确的答案，两方法准确率回到 60%。
- α=4 使接受更挑剔（接受率从 ~55% 降至 30–43%），自适应节省幅度相应收窄：**省 27.5% token、12.4% 时间**，被拒 token 减少 38%。
- 自适应平均执行 3.8/8 次，停止原因符合设计（低增益或连续 4 次拒绝），不再是"2 次拒绝即停"。
- 5 题样本仍只是冒烟信号：α=4 未损害也未提升准确率，是否有增益需 100 题验证。
- 人工复核发现：第 4 题（因数个数）是判分假阴性——模型数学全对但没写 `\boxed{}`；**真实初始正确率为 4/5=80%**。第 5 题（Evelyn/Carla）为模型真实错误，判分无误。详见 LOG 复核条目。

详细报告：[`output/pdf/EFB_B同学_α4修复版复跑报告.pdf`](output/pdf/EFB_B同学_α4修复版复跑报告.pdf)

## 5 题新 prompt + 完整性护栏复跑结果（v2）

代码 `8eda4df` 在同一 4090 复跑（α=4、steps=8、seed 配对同上），改动：官方 Qwen CoT system prompt 经 chat template、完整性护栏。全程仅 6 分钟——官方 prompt 使生成大幅变短（平均初始 645 token）。

| 方法 | 判分准确率 | 平均总 token | 平均被拒 token | 接受率 | 平均耗时/题 | 提前停止率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通初始生成 | 40% | 645.4 | — | — | — | — |
| 固定 8 次 | 40% | 1446.2 | 713.6 | 22.5% | 38.06 秒 | 0% |
| 自适应停止 | 40% | 1221.6 | 503.0 | 19.0% | 31.08 秒 | 60% |

v2 复跑主要结论：

- **暴露新的判分失效模式**：Qwen2.5-Math-7B 是 **base 模型**，官方零样本 chat prompt 是 Instruct 版协议；base 模型不会在 `<|im_end|>` 停止，答完正确答案后继续自编新题自问自答，"取最后一个 `\boxed{}`"取到自编题的答案。q3（`\boxed{14/3}` 在中段）、q4（第一个框即 `\boxed{9}`）实际都做对了，**真实数学正确率 4/5=80%，判分 40% 严重低估**。
- q5 仍是模型真实错误（伪造 ```output 块宣称 Carla；重采样翻成 Angela，仍错），与 α=4 复跑结论一致：错误源头在重采样窗口之前，方法无法修复。
- 完整性护栏正确工作：q3/q4 各拦下 7/8 个未自然终止的候选（"题目接龙"文本让 128 token 后缀几乎无法自然收尾），防止不完整候选进入链，但也使接受率降至 ~20%。
- 重采样依旧不破坏任何初始正确答案；自适应省 15.5% token、18.3% 时间。
- **下一步**：换 `Qwen/Qwen2.5-Math-7B-Instruct`（首选）或 base + few-shot 官方协议，解决"停不下来"后重建 5 题基线。

## 5 题 Instruct 模型基线（v3，2026-07-19）

换用 `Qwen/Qwen2.5-Math-7B-Instruct`（与官方零样本 CoT chat 协议匹配；`run_math500` 默认模型已同步更新），其余配置与 v2 完全一致。

| 方法 | 准确率 | 平均总 token | 平均被拒 token | 接受率 | 平均耗时/题 | 提前停止率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通初始生成 | **100%** | 493.0 | — | — | — | — |
| 固定 8 次 | **100%** | 1079.4 | 215.6 | 70.0% | 27.03 秒 | 0% |
| 自适应停止 | **100%** | 921.4 | 164.4 | 74.3% | 23.49 秒 | 60% |

v3 基线结论：

- **判分恢复可信**：每题初始文本恰好 1 个 `\boxed{}`，v2 的"答完不停、自编新题"失效模式完全消失，确认根因就是 base 模型与 chat 协议不匹配。
- **q5 首次做对**：Instruct 版走纯 CoT 逐个计算速度（Evelyn 3.6 最大），不再伪造 ```output 块。
- 链健康度恢复：接受率回到 70%+（v2 仅 ~20%），完整性护栏仅偶发触发（3 次）、答案护栏 0 次，回归兜底角色。
- 自适应 3/5 提前停止（q4 仅 2 步、q3 4 步），省 14.6% token、13.1% 时间。
- **局限**：这 5 题对 Instruct 太简单（初始即 100%），展示准确率增益需要更难/更多题目——下一步 5–20 题调参后推进 100 题 paired。

详细报告：[`output/pdf/EFB_B同学_Instruct基线报告.pdf`](output/pdf/EFB_B同学_Instruct基线报告.pdf)

详细报告：[`output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf`](output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf)

## 实验产物

```text
results/fixed_corrected.jsonl        1 题固定版校正试跑
results/adaptive_corrected.jsonl     1 题自适应版校正试跑
results/fixed_5_paired.jsonl         5 题固定版配对结果（修复前）
results/adaptive_5_paired.jsonl      5 题自适应版配对结果（修复前）
results/five_paired_summary.json     5 题汇总指标（修复前）
results/fixed_5_alpha4.jsonl         5 题固定版 α=4 修复版复跑
results/adaptive_5_alpha4.jsonl      5 题自适应版 α=4 修复版复跑
results/alpha4_summary.json          α=4 复跑汇总指标
results/alpha4_answer_review.md      5 题人工复核清单（题目/官方答案/模型答案/全文）
results/fixed_5_v2.jsonl             5 题固定版新 prompt+护栏复跑（v2）
results/adaptive_5_v2.jsonl          5 题自适应版新 prompt+护栏复跑（v2）
results/v2_summary.json              v2 复跑汇总指标
results/fixed_5_instruct.jsonl       5 题固定版 Instruct 基线（v3）
results/adaptive_5_instruct.jsonl    5 题自适应版 Instruct 基线（v3）
results/instruct_summary.json        v3 基线汇总指标
results/env_versions.txt             服务器 Python/torch/transformers 版本
results/env_nvidia_smi.txt           服务器 GPU 状态记录
figures/five_paired_comparison.png   5 题正式对比图（修复前）
figures/alpha4_comparison.png        α=4 修复版复跑对比图
figures/instruct_comparison.png      v3 Instruct 基线对比图
figures/smoke_comparison.png         早期冒烟对比图
output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf
output/pdf/EFB_B同学_α4修复版复跑报告.pdf
output/pdf/EFB_B同学_Instruct基线报告.pdf
```

`results/*`、`figures/*` 和 `tmp/` 默认不提交 Git；PDF 当前位于 `output/pdf/`，但尚未提交。若要公开实验结果，需要显式调整 `.gitignore` 或使用 `git add -f`，并先确认文件体积和内容。

## 已实现模块

- `sampling/hf_backend.py`：计算条件 log-likelihood、构建 ChatML prompt，并从指定 token 位置重采样后缀（自然 EOS 终止，兼容 `<|im_end|>`/`<|endoftext|>` 双停止符）。
- `sampling/power.py`：p^α 目标的 `(α−1)Δ` Metropolis-Hastings 接受规则、`\boxed{}` 答案护栏、截断候选完整性护栏、固定轮次基线和自适应提前停止（接受步增益 + 连续拒绝双判据）。
- `sampling/criticality.py`：用 token surprise 找出模型最不确定的位置。
- `sampling/toy_validation.py`：三状态玩具分布验证 proposal ratio 修正；变长序列玩具模型验证 `(α−1)Δ` 收敛到 p^α。
- `sampling/metrics.py`：记录生成、拒绝、接受、提前停止和节省尝试等开销。
- `generation/run_math500.py`：读取 MATH-500、运行实验并写入 JSONL。
- `evaluation/answers.py`：提取 `\boxed{}` 答案并做基础格式归一化。
- `evaluation/summarize_results.py`：汇总准确率、token、接受率和运行时间。
- `evaluation/plot_results.py`：生成普通生成、fixed、adaptive 的比较图。
- `tests/`：无需 GPU、无需下载模型的离线回归测试。

## 目录结构

```text
generation/   MATH-500 生成流水线
evaluation/   判分、汇总与画图工具
sampling/     B 同学的 Power Sampling 实现
tests/        离线单元测试
results/      本地 JSONL 与汇总结果，默认忽略
figures/      本地图表，默认忽略
output/pdf/   交付用 PDF 报告
tmp/          本地临时文件，默认忽略
```

## 环境

建议使用 NVIDIA 24GB 或更大显存的 Linux GPU 环境。当前 GPU 冒烟实验已在 RTX 4090 24GB 上跑通；项目要求 Python 3.10+，手册推荐 PyTorch CUDA 12.1 构建。AutoDL 的精确基础镜像名称和已安装包版本尚未写入结果文件，因此本仓库只记录可确认的信息，不猜测具体镜像版本。

```bash
conda create -n reason python=3.10 -y
conda activate reason
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[dev,analysis]'
```

国内访问 Hugging Face 较慢时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

令牌和服务器登录信息只能放在环境变量或本机配置中，禁止写入 README、LOG、源码或 Git 历史。

## 验证与运行

离线测试：

```bash
python -m unittest discover -s tests -v
python -m sampling.toy_validation
```

固定版 5 题：

```bash
python -m generation.run_math500 \
  --model Qwen/Qwen2.5-Math-7B \
  --mode fixed \
  --limit 5 \
  --steps 8 \
  --alpha 4.0 \
  --seed 42 \
  --output results/fixed_5_paired.jsonl
```

自适应版 5 题：

```bash
python -m generation.run_math500 \
  --model Qwen/Qwen2.5-Math-7B \
  --mode adaptive \
  --limit 5 \
  --steps 8 \
  --alpha 4.0 \
  --min-steps 2 \
  --patience 2 \
  --rejection-patience 4 \
  --gain-threshold 0.01 \
  --seed 42 \
  --output results/adaptive_5_paired.jsonl
```

汇总和画图：

```bash
python -m evaluation.summarize_results \
  results/fixed_5_paired.jsonl \
  results/adaptive_5_paired.jsonl

python -m evaluation.plot_results \
  results/fixed_5_paired.jsonl \
  results/adaptive_5_paired.jsonl \
  --output figures/five_paired_comparison.png
```

每道题使用 `seed + 题号` 重新设置模型生成和选位随机数，因此 fixed 与 adaptive 会共享同一道题的初始生成，并在停止前使用一致的提议序列。Prompt 采用 Qwen2.5-Math 官方 CoT system prompt 经 chat template 构建（结果记录 `prompt_style: qwen-cot-chat`）。切分点只在最后 `suffix_max_new_tokens` 个位置内选择；重采样后缀允许自然 EOS 终止（不再强制等长），长度偏置由接受规则中的 proposal 修正从原理上处理：目标分布 p^α、proposal 为模型自身时，接受概率化简为 `min(1, exp((α−1)·(log p′ − log p)))`。两道护栏会直接拒绝不合格候选：丢失当前已有 `\boxed{}` 答案的候选（答案护栏），以及顶满预算而无自然终止的候选（完整性护栏，其概率质量不完整）。

## 下一步

1. 人工复核 5 题的预测与标准答案，确认基础判分没有把数学等价答案判错。
2. 在 5–20 题上粗调 `alpha`、`gain_threshold`、`patience`、`rejection_patience`，观察准确率与计算量。
3. 通过上述检查后运行 100 题 paired 实验，并报告均值、方差或置信区间。
4. 与论文原文逐项核对 proposal distribution 和接受概率（变长玩具验证已通过），再接入难点优先选位。
5. 只有 100 题结果稳定后，才考虑运行 MATH-500 全量实验。

> 接受规则已从 likelihood-ratio 入门基线升级为 p^α 目标的 Metropolis-Hastings（含 proposal 修正），并在变长玩具分布上验证收敛。改变选点分布（如难点优先）时，仍必须把新的选位概率纳入 proposal ratio。
