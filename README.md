# 推理效率研究 - B 同学工作区

本仓库按《推理效率研究·零基础实操版》初始化，聚焦 B 同学负责的 **Power Sampling 算力分配**：先跑通固定轮次版本，再研究自适应停止与难点优先重采样。

## 当前已完成

- `sampling/hf_backend.py`：计算条件 log-likelihood、从指定 token 位置重采样后缀。
- `sampling/power.py`：固定轮次基线和按 likelihood 连续低增益提前结束的自适应版。
- `sampling/criticality.py`：用 token surprise 找出模型最不确定的位置。
- `sampling/toy_validation.py`：在三状态玩具分布验证 proposal ratio 修正。
- `sampling/metrics.py`：记录总生成 token、被拒绝 token、接受率等开销。
- `generation/run_math500.py`：从 MATH-500 取少量题目运行并写入 JSONL。
- `evaluation/answers.py`：提取 `\boxed{}` 答案并做基础格式归一化。
- `tests/`：无需 GPU、无需下载模型的离线单元测试。

## 目录约定

```text
generation/   共用生成流水线
evaluation/   共用判分工具
sampling/     B 同学的 Power Sampling 代码
results/      实验结果，不提交 Git
figures/      生成的图，不提交 Git
tests/        离线测试
```

## 环境

建议在带 NVIDIA GPU 的 Linux 服务器上使用 Python 3.10：

```bash
conda create -n reason python=3.10 -y
conda activate reason
pip install -e '.[dev,analysis]'
```

若服务器需要 Hugging Face 镜像，可复制 `.env.example` 中的变量到 shell 环境。密钥只放环境变量，禁止写进源码。

## 先做离线验证

```bash
python -m unittest discover -s tests -v
```

## 跑 5 道 MATH-500 冒烟实验

```bash
python -m generation.run_math500 \
  --model Qwen/Qwen2.5-Math-7B \
  --mode fixed \
  --limit 5 \
  --steps 8 \
  --output results/fixed_power_sampling.jsonl
```

首次运行会下载模型和数据集。程序默认使用固定 8 次重采样，并为每道题保存：

- 初始/最终文本与 log-likelihood
- 初始/最终答案与各自正确率
- 重采样次数、接受次数和接受率
- 初始 token、候选 token、总生成 token
- 被拒绝候选的 token 数与浪费比例
- 最终答案、基础判分结果和运行时间

确认 5 道题流程无误后再把 `--limit` 改成 `100`。不要一开始直接跑 500 道。

为避免短序列因累计 log-likelihood 较高而被错误偏好，当前基线只在最后 `suffix_max_new_tokens` 个位置内选择切分点，并强制生成与原后缀等长的替代文本，不会因为重采样预算而直接截断答案。

每道题使用 `seed + 题号` 重新设置模型生成和位置选择随机数，因此 fixed 与 adaptive 会共享同一道题的初始生成，并在停止前使用一致的提议序列，可做配对比较。

## 跑自适应停止版

```bash
python -m generation.run_math500 \
  --model Qwen/Qwen2.5-Math-7B \
  --mode adaptive \
  --limit 5 \
  --steps 8 \
  --min-steps 2 \
  --patience 2 \
  --gain-threshold 0.01
```

`steps` 是最大预算；连续 `patience` 次当前链的 likelihood 变化不超过阈值时会提前结束。结果中的 `saved_attempts` 表示相对固定预算少做了多少次重采样，`trace` 保存每一步的选位、提议分数和接受结果。

## 验证非均匀提议公式

```bash
python -m sampling.toy_validation
```

输出会同时展示加入和遗漏 proposal ratio 时的经验分布。只有修正后的链应接近理论目标分布。真实模型的“难点优先重写”暂未接入主实验，必须先根据论文/导师确认完整接受概率，避免得到数学上错误的结果。

## 汇总与画图

固定版和自适应版各跑出一个 JSONL 后：

```bash
python -m evaluation.summarize_results \
  results/fixed_power_sampling.jsonl \
  results/adaptive_power_sampling.jsonl

python -m evaluation.plot_results \
  results/fixed_power_sampling.jsonl \
  results/adaptive_power_sampling.jsonl
```

第二条命令生成 `figures/power_sampling_report.png`：左图同时展示普通初始生成、fixed 和 adaptive 的 token-正确率对比，右图是归一化重写位置分布。

## 后续里程碑

1. 固定 8 次基线：先确认开销统计可信。
2. 自适应停止：已实现，下一步在 5 道题上调阈值。
3. 难点优先：评分和排序已实现，确认接受公式后再接入真实模型。
4. 数学验证：三状态玩具验证已实现，需要保留为回归测试。
5. 完整实验：对比普通生成、固定版、自适应版、难点版和完整版。

> 当前接受规则只是手册中的 likelihood-ratio 入门基线，不等同于对论文公式的完整复现。正式实验前必须与原论文逐项核对；改变选点或提议分布时，还要把 proposal ratio 纳入 Metropolis-Hastings 接受概率。
