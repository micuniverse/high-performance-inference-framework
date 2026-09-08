# CUDA Graph 与 metadata staging 性能复现（2026-09-08）

## 结果与“10%”的含义

**相对关闭 CUDA Graph，本轮明显超过 10%；但 CPU metadata staging 在已有 Graph 上的增量没有复现稳定的 10%。** 两者使用不同的基线，不能混用。

| 输入 token | A：关闭 Graph | B：Graph + 临时 metadata | C：Graph + staging | 完整 Graph 路径 C/A 提升 | staging 增量 C/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.11 | 84.19 | 88.50 | +449.28% | +5.12% |
| 1024 | 15.87 | 81.86 | 84.72 | +433.67% | +3.49% |
| 2048 | 16.24 | 76.02 | 75.08 | +362.42% | -1.24% |

主表单位为 **Decode token/s**。当前完整 Graph 路径的吞吐达到关闭 Graph 时的 **4.62–5.49 倍**，即提高 **362%–449%**。这些值适用于当前自写 RMSNorm + eager 残差归一化设置下的单请求测试，不能推广为所有算子配置、批量或设备的收益。

仅开启 Graph、保留临时 metadata 的 B/A 也有约 **368%–423%** 的提升。因此本次主要可确认的是 Graph 开关带来的大幅差异，不能把它归因于 CPU staging。

两轮存在波动。例如 2048 token 输入时，C 的逐轮 Decode 为 82.50 / 68.88 token/s，B 为 77.88 / 74.25 token/s，增量方向并不一致。因此 C/B 的小幅变化不支持“buffer 复用稳定提升 10%”的结论；本次没有做频率锁定或 profiler 归因。

## 简历表述建议

> 基于 nano-vLLM 适配并验证 CUDA Graph Decode 路径，实现 CPU pinned metadata buffer 复用；在 RTX 3050 Laptop 4GB、Qwen3-0.6B、FP16 KV Cache、batch=1、输入 512–2048 token / 输出 128 token 条件下，相对关闭 CUDA Graph，Decode 吞吐由约 16 token/s 提升至 75–89 token/s。

如果保留百分比，必须写明相对关闭 Graph 及测试条件。不能将本次结果写成“在原有 Graph 基础上，metadata 优化提升 10%”，也不能据此宣称已验证多请求稀疏 Decode 场景。历史那次测试的完整配置未知，本次不能证明历史的 10% 数字。

本轮 Graph 收益不能与先前算子 eager/compile 消融的提升比例直接相加，它们的对照基线不同，优化也可能相互影响。

## 面试说明

> 我把模型、算子后端和 KV Cache 固定，只切换 Graph 及 CPU staging。每个 step 的计时包含调度、输入准备、模型和采样，排除加载、预热及捕获。提升比例按 `(开启后的总 token/总耗时) / (关闭后的总 token/总耗时) - 1` 计算，并使用两个独立进程的所有样本。

> 需要分清原版已经有的 Graph 和固定 GPU buffer，以及我增加的 CPU metadata buffer 复用。Graph 捕获模型主体以避免每步重复执行主体的 Python/CUDA launch 流程；metadata 每步仍更新，LM Head 和采样也仍在图外。本次没有用 profiler 对收益逐项归因，CPU staging 本身也没有测出稳定的 10%。

## 逐轮吞吐（token/s）

| 输入 token | A 两轮 | B 两轮 | C 两轮 |
| --- | --- | --- | --- |
| 512 | 16.37 / 15.87 | 88.53 / 80.26 | 91.14 / 86.01 |
| 1024 | 17.44 / 14.56 | 85.12 / 78.84 | 88.79 / 81.00 |
| 2048 | 16.32 / 16.15 | 77.88 / 74.25 | 82.50 / 68.88 |

## 对照设计

同一份 Qwen3-0.6B 权重，RTX 3050 Laptop 4GB，FP16 KV Cache，普通 RMSNorm 使用当前自写 CUDA kernel，残差 Add+RMSNorm 仍为 PyTorch eager。其他算子、torch.compile 设置、模型与参数保持一致，仅切换 Graph 和 CPU metadata staging。

| 配置 | CUDA Graph | CPU pinned metadata buffer 复用 | 路径 |
| --- | --- | --- | --- |
| A：`graph_off` | 关闭 | 无 | 每步普通模型前向，临时创建 metadata |
| B：`graph_temp_metadata` | 开启 | 无 | 临时创建 metadata，复制进 Graph 固定 GPU buffer 后 replay |
| C：`graph_staging` | 开启 | 有 | 复用 CPU pinned buffer，更新内容并 H2D 复制进固定 GPU buffer 后 replay |

A→B 测试当前框架中开启 Graph 的效果；A→C 测试当前完整 Graph 路径的效果；B→C 单独测试已有 Graph 路径上 CPU metadata staging 的增量效果。这三个百分比不能互换。

B 使用当前代码已有的 fallback：在捕获完成后禁用 `decode_staging` 属性，`prepare_decode` 创建临时 CPU/GPU metadata，`run_model` 将其复制到 Graph 输入缓冲。没有删除 Graph 必须使用的固定 GPU buffer。C 是框架当前默认实现。实验开关不改变生产默认行为。

## 测量条件与统计

- batch=1，输入 512/1024/2048 token，输出固定 128 token，忽略 EOS。
- 三种配置各两轮独立进程，顺序 A→B→C→C→B→A。
- 每个输入长度先预热一个完整请求，再测 5 次 Prefill-only 和 3 次完整请求。
- 每组、每个长度合计 10 次 Prefill、6 次完整生成、762 个 Decode 步骤。全实验共 54 次测量用完整生成、6858 个 Decode 步骤；不挑选最好轮次。
- 主指标为总 Decode token 数 / 总 Decode 耗时。首 token 在 Prefill 阶段生成，因此每次 128 token 输出对应 127 个 Decode 步骤。
- 每步前后同步 CUDA，覆盖 scheduler、metadata 准备/H2D、模型、LM Head 和采样。不计 tokenizer、权重加载、JIT 编译、Graph 捕获及预热时间。
- 另保留 TTFT、TPOT P50/P90、Prefill、含 Prefill 的端到端输出吞吐和 PyTorch allocator 峰值 allocated。
- 固定随机输入生成方式和采样种子；汇总时核对输入 SHA256、代码版本与参数一致。

`enforce_eager=True` 在这个项目中表示禁用 CUDA Graph，不会关闭 RoPE/SwiGLU 等层已有的 torch.compile。不能把 A 描述成“所有层均未编译”。

三组都采用 `gpu_memory_utilization=0.9` 自动分配 FP16 KV Cache，实际容量保存在每轮元数据中；并未固定相同总缓存容量。未锁定 GPU 频率，运行于日常 WSL2/桌面环境。结果只适用于该硬件、单请求、输入输出长度和固定算子配置。

## 捕获范围与实现细节

Graph 捕获的是 `self.model(input_ids, positions)` 模型主体。`compute_logits`（LM Head）、采样、调度和 metadata 更新在图外，主指标仍包含这些时间。因此单次 Graph replay 的耗时不等于这里的完整 Decode step 耗时。

固定 GPU buffer 包括 input IDs、positions、slot mapping、context lengths、block tables 和模型输出。C 额外复用 CPU pinned metadata staging buffer，每步填入新的 token、位置及页表，再异步复制到捕获时的固定 GPU 地址。温度等采样输入并未全部改成 persistent buffer，不能说所有分配都已消除。

原版 nano-vLLM 已有 CUDA Graph 和固定 GPU buffer 支持；学习分支新增了 CPU metadata staging 路径。本轮只验证 batch=1，没有测试多请求、不同有效 batch bucket 或特定“稀疏 Decode”负载，不能用这些数据证明该类场景的收益。

## 数值检查

在独立进程重新生成同一权重的 Transformers FP16 eager Attention 参考，用 3 个英文提示、26 个步骤做 teacher-forcing logits 比较。A/B/C 三组 top-1 均为 26/26 一致，原始误差见 `check_*.json`。这是小样本短上下文检查，不是准确率或困惑度评估。每轮性能测试另保留一次自然语言自由采样输出。

## 复现命令

在仓库根目录、安装依赖的 Python 环境中执行：

```bash
MAX_JOBS=2 python run_graph_benchmarks.py --model /path/to/Qwen3-0.6B \
  --output-dir benchmark_results/graph_new_run
python summarize_graph_benchmarks.py benchmark_results/graph_new_run
```

单次实验仍使用同一个 benchmark，固定其余参数：

```bash
python benchmark_fa2_e2e.py --model /path/to/Qwen3-0.6B \
  --kv-quant off --rmsnorm-backend cuda --cuda-graph off --decode-staging off \
  --lengths 512 1024 2048 --decode-tokens 128 \
  --prefill-repeats 5 --decode-repeats 3 --warmup-runs 1 --output /tmp/graph_off.json
```

分别将 Graph/staging 参数设置为 `off/off`、`on/off`、`on/on`。数值检查也支持同名选项，见 `check_reference.py`。完整参数、模型校验和、软件版本及预定顺序在 `environment.json`；聚合数据在 `summary.json`，逐步时延在六份 `*_round*.json`。
