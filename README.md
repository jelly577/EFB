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
| 固定版/自适应版配对复现 | ✅ 完成 | 同题共享初始生成，停止前共享提议序列 |
| 离线单元测试 | ✅ 11/11 通过 | 2026-07-18 本地重新验证 |
| 1 题 GPU 校正试跑 | ✅ 完成 | 用于确认等长后缀重采样修复 |
| 5 题 GPU 冒烟实验 | ✅ 完成 | RTX 4090 24GB，fixed 与 adaptive 各 5 题 |
| 100 题实验 | ⏳ 未运行 | 应先解决正确率下降和判分可靠性问题 |
| 500 题完整实验 | ⏳ 未运行 | 当前没有全量结果，不能作最终统计结论 |
| 难点优先真实模型实验 | ⏳ 未运行 | 需先确认完整 Metropolis-Hastings 接受概率 |

GitHub 主分支目前同步到提交 `b7c276d`（`Compare against ordinary generation`）。GPU 结果、图表和最新 PDF 均为本地产物，受 `.gitignore` 或尚未提交状态影响；本次文档更新也需要后续提交后才会同步到远端。

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

这批结果只证明自适应停止在当前设置下明显减少计算，**不证明方法提升准确率**。样本仅 5 题；普通初始生成准确率为 60%，两种重采样方法均为 40%，其中存在原本答对但重采样后答错的负面样例。扩大实验前应优先检查接受规则、答案保持和数学等价判分。

详细报告：[`output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf`](output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf)

## 实验产物

```text
results/fixed_corrected.jsonl        1 题固定版校正试跑
results/adaptive_corrected.jsonl     1 题自适应版校正试跑
results/fixed_5_paired.jsonl         5 题固定版配对结果
results/adaptive_5_paired.jsonl      5 题自适应版配对结果
results/five_paired_summary.json     5 题汇总指标
figures/five_paired_comparison.png   5 题正式对比图
figures/smoke_comparison.png         早期冒烟对比图
output/pdf/EFB_B同学_5题GPU冒烟实验报告.pdf
```

`results/*`、`figures/*` 和 `tmp/` 默认不提交 Git；PDF 当前位于 `output/pdf/`，但尚未提交。若要公开实验结果，需要显式调整 `.gitignore` 或使用 `git add -f`，并先确认文件体积和内容。

## 已实现模块

- `sampling/hf_backend.py`：计算条件 log-likelihood，并从指定 token 位置重采样后缀。
- `sampling/power.py`：固定轮次基线和按 likelihood 连续低增益提前结束的自适应版本。
- `sampling/criticality.py`：用 token surprise 找出模型最不确定的位置。
- `sampling/toy_validation.py`：在三状态玩具分布中验证 proposal ratio 修正。
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
  --min-steps 2 \
  --patience 2 \
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

每道题使用 `seed + 题号` 重新设置模型生成和选位随机数，因此 fixed 与 adaptive 会共享同一道题的初始生成，并在停止前使用一致的提议序列。当前基线只在最后 `suffix_max_new_tokens` 个位置内选择切分点，并强制生成与原后缀等长的替代文本，避免短序列仅因累计 log-likelihood 较高而被偏好。

## 下一步

1. 人工复核 5 题的预测与标准答案，确认基础判分没有把数学等价答案判错。
2. 定位“初始正确、重采样后错误”的样例，检查接受规则与答案保持约束。
3. 在 5 题上调节 `gain_threshold`、`patience` 和最小执行步数，观察准确率与计算量。
4. 通过上述检查后运行 100 题 paired 实验，并报告均值、方差或置信区间。
5. 与论文逐项核对 proposal distribution 和接受概率，再接入难点优先选位。
6. 只有 100 题结果稳定后，才考虑运行 MATH-500 全量实验。

> 当前接受规则是手册中的 likelihood-ratio 入门基线，不等同于论文公式的完整复现。改变选点或提议分布时，必须把 proposal ratio 纳入 Metropolis-Hastings 接受概率。
