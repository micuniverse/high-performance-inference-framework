# Qwen3-0.6B：FP16 / INT8 KV Cache 实测

## 汇总结果

主表使用全部 Decode token / 全部 Decode 耗时，非最佳单轮；单位 token/s。

| 输入 token | FP16 Prefill | INT8 Prefill | FP16 Decode | INT8 Decode | FP16 TTFT（ms） | INT8 TTFT（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8152.1 | 8143.5 | 93.75 | 19.28 | 64.43 | 63.34 |
| 1024 | 7891.5 | 7765.0 | 92.22 | 19.02 | 131.30 | 132.07 |
| 2048 | 7697.3 | 7607.9 | 85.31 | 18.54 | 265.46 | 270.37 |

FP16 单请求 Decode 为 **85.3–93.8 token/s**，Prefill 约 **7.70k–8.15k token/s**。INT8 Decode 为 **18.5–19.3 token/s**，当前实现没有获得吞吐提升。

实测每 token、全 28 层的 K/V 缓存存储：FP16 **114,688 B（112 KiB）**；INT8 加 FP32 scale **59,136 B（57.75 KiB）**，减少 **48.4375%**。例如，同样存储 2048 个 token，缓存本体及 scale 对应 **224 MiB → 115.5 MiB**；这是按实测单位存储计算的同容量值，不是本轮自动预分配总量。

本轮 FP16 自动预分配容量分别为 12,032 / 14,336 token，INT8 两轮均为 27,648 token。框架会随启动时内存状态分配不同容量，因此不能从这些数字认定总显存降低或并发容量提升固定倍数。

各长度和轮次 PyTorch peak allocated 范围：FP16 **2503.5–2797.7 MiB**；INT8 **2747.5–2789.7 MiB**。本次未证明运行时峰值显存下降。当前 INT8 Attention 会对整份预分配 Cache 反量化，存在额外读写和临时缓冲开销；未做 profiler 归因，不能将全部吞吐差异归因于单一环节。

| 小样本检查 | FP16 | INT8 |
| --- | ---: | ---: |
| 最高分 token 与参考一致 | 26/26 | 24/26 |
| 平均 logits 绝对误差（逐步均值再平均） | 0.0106 | 0.7382 |
| 最大 logits 绝对误差 | 0.1074 | 12.7031 |

INT8 有可观的数值变化，不能描述为精度无损。3 个短提示、26 步的 token 一致率不是整体任务准确率。四轮自由采样的自然语言冒烟输出均保留在各 JSON 的 `output_smoke_test` 字段。

## 简历表述建议

> 基于 nano-vLLM 扩展 INT8 KV Cache，采用按 token/head 量化及 FP32 scale，使单位 token KV 缓存存储较 FP16 减少 48.4%；构建可复现的 Prefill/Decode 评测流程，在 RTX 3050 Laptop 4GB 上运行 Qwen3-0.6B，FP16 单请求 Decode 吞吐为 85–94 token/s（输入 512–2048 token，输出 128 token）。

不要将 FP16 吞吐写成 INT8 吞吐；不要写“INT8 推理加速”“整体显存降低 48.4%”或“相对上游提升 30%”。实验比较的是缓存模式，未运行上游/vLLM 性能基线。

## 数据文件

- `fp16_round1.json`、`fp16_round2.json`：FP16 原始数据。
- `int8_round1.json`、`int8_round2.json`：INT8 原始数据。
- `summary.json`：所有轮次聚合数据、每轮吞吐与峰值 allocated。
- `reference.json`、`check_fp16.json`、`check_int8.json`：短提示及逐步数值检查。
- `environment.json`：版本、模型校验和与预定参数。

## 实验设计

2026-09-07，RTX 3050 Laptop GPU（4 GB），单 GPU、单请求、CUDA Graph 开启。模型为本地 Qwen3-0.6B，计算 dtype 为 FP16；两组仅切换 KV Cache 存储格式。模型校验和、依赖版本与 GPU 信息见 `environment.json`。未锁定 GPU 频率，测试在日常 WSL2/桌面环境中执行，结果按两轮实测汇总。

输入长度预先选择为 512、1024、2048 token，输出固定 128 token，忽略 EOS。输入生成器沿用 `20260824 + length × 1009 + case_id`，PyTorch 采样种子为 `20260907`；性能输入是随机 token，不是自然语言质量评测数据。每个长度先运行一次完整请求预热，然后测量 5 次 Prefill-only 请求和 3 次完整生成请求。

进程顺序为 FP16 → INT8 → INT8 → FP16，各进程独立加载模型和捕获 CUDA Graph。每种模式、每个长度最终汇总 10 次 Prefill 和 6 次完整请求（共 762 个 Decode 步骤）。所有轮次都纳入统计，没有取最好的一轮。

## 指标口径

- **Decode token/s**：所有测得的 Decode token 数除以 Decode 总耗时；每次请求的首 token 在 Prefill 阶段产生，因此每次请求有 127 个 Decode 步骤。该指标是单请求 Decode 阶段吞吐，不能代替多请求服务吞吐。
- **TPOT P50/P90**：汇总 Decode 步骤延迟的分位数。原脚本 `decode_tok_s` 是 P50 延迟的倒数；新增的 `decode_tok_s_aggregate` 才是这里报告的总 token / 总耗时。
- **TTFT**：完整请求首次 step 的延迟，汇总 6 次请求取中位数。Prefill-only 的耗时另取 10 次中位数。
- **端到端输出吞吐**：生成的全部 128 token 除以 Prefill + Decode 测量耗时，与纯 Decode 吞吐区分。
- 计时覆盖 scheduler、H2D 输入准备、模型、采样，并在每个 step 前后同步 GPU；不包括 tokenizer、模型加载、JIT 编译或 CUDA Graph 捕获。
- **显存**：记录 PyTorch allocator 的峰值 allocated/reserved，包括预分配 Cache 与 Graph。它不是进程完整显存或 nvidia-smi 的显存使用值。

两组均保留框架 `gpu_memory_utilization=0.9` 的自动缓存分配策略，因此实际预分配 token 容量可能不同；每轮均记录 blocks 和 capacity。INT8 会在 Attention 前反量化缓存，容量变化也会影响反量化开销。这个实验比较当前框架在相同预算策略下的行为，未隔离固定缓存容量的 kernel 性能。

缓存压缩率采用运行时实际 Tensor 的 `numel × element_size`，包含 FP32 scale，再按已分配 token 容量归一化。单位 token KV 存储减少不能等同于总峰值显存减少。

## 小样本数值检查

`check_reference.py` 使用同一模型的 Transformers FP16 eager Attention 作为参考。先在独立进程生成参考 token 和 logits，再让本框架在同一 token 序列上执行 Prefill/Decode（teacher forcing），逐步比较输出分数。测试不进入性能计时。

3 个英文提示共 26 步。这里只检验少量短上下文，不能代替困惑度、长上下文验证或自然语言准确率。量化路径即使 top-1 相同，也可能有其他 logits 误差；完整逐步结果保留在 `check_fp16.json` 和 `check_int8.json`。

## 复现

在仓库根目录、安装好依赖的环境中执行。参考 Tensor 文件约十几 MB，放在临时目录，不提交模型权重或该二进制文件。

```bash
python check_reference.py --model /path/to/Qwen3-0.6B --backend transformers \
  --reference /tmp/nano-reference.pt --output /tmp/reference.json
python check_reference.py --model /path/to/Qwen3-0.6B --backend nano --kv-quant off \
  --reference /tmp/nano-reference.pt --output /tmp/check_fp16.json
python check_reference.py --model /path/to/Qwen3-0.6B --backend nano --kv-quant on \
  --reference /tmp/nano-reference.pt --output /tmp/check_int8.json

MAX_JOBS=2 python run_resume_benchmarks.py --model /path/to/Qwen3-0.6B \
  --output-dir benchmark_results/new_run
python summarize_benchmarks.py benchmark_results/new_run
```

汇总脚本会校验模型配置、脚本版本、参数以及每次请求输入的 SHA256 一致，防止把不同工作负载混为一次对照。结果适用于该硬件和实验条件，不是相对上游 nano-vLLM 或 vLLM 的加速比例。
