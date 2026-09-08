# 高性能推理框架

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的个人学习分支，整理本地 `nano-vllm1` 的改动。探索 Qwen3 推理、INT8 KV Cache、自定义 CUDA 算子接入，以及 Prefill/Decode 性能测量。

**当前是实验项目。** 发布整理后，FP16 与 INT8 均完成短序列推理冒烟测试，不再出现历史报告中的重复感叹号现象；完整生成质量、困惑度和长上下文正确性仍待验证。历史 INT8 Decode 吞吐低于 FP16，原始数据保留用于分析，不能作为加速结论。详见 [验证记录](docs/VALIDATION.md)。

## CUDA Graph 与 metadata staging 对照（2026-09-08）

固定当前 CUDA RMSNorm、FP16 KV Cache，在单请求、输入 512/1024/2048 token、输出 128 token 条件下，完成 Graph 关闭、Graph + 临时 metadata、Graph + CPU staging 三组共六个独立进程测试。

- 当前完整 Graph 路径的 Decode 吞吐由约 **16 token/s 提高至 75–89 token/s**，达到关闭 Graph 时的约 **4.62–5.49 倍**。
- 在已有 Graph 上单独启用 CPU staging，汇总增量分别为 **+5.12%、+3.49%、−1.24%**，没有复现稳定的 10% 提升。
- 三组均完成 26 步小样本数值检查；这里只验证 batch=1，不能推广为多请求稀疏 Decode 结果。

完整基线定义、逐轮波动、复现命令及简历表述见 [Graph 性能复现报告](benchmark_results/graph_20260908/README.md)。

## 自写算子接入前后对照（2026-09-07）

当前模型主路径实际启用的自写算子是 RMSNorm。固定 Qwen3-0.6B、FP16 KV Cache、单请求、CUDA Graph 后，完成四种 norm 配置、两轮反向顺序共八个独立进程的对照：

- 相对 **PyTorch eager RMSNorm**，当前 CUDA 版本的 Decode 吞吐提升 **6.1%–6.9%**。
- 相对恢复原版 norm 的 **torch.compile 设置**，当前保存版本的 Decode 吞吐低 **8.2%–8.7%**；残差 Add+RMSNorm 的编译设置也发生了变化。
- 保持相同的编译残差，仅替换普通 RMSNorm 后，Decode 差异约 **−0.06% 至 −0.54%**，未观察到明确优势。

本次没有复现“相对原版提升 30%”。完整对照表、可用简历表述、面试回答示例和原始数据见 [自写 RMSNorm 复现报告](benchmark_results/operators_20260907/README.md)。输入长度 512/1024/2048 token、输出 128 token；不要混淆 eager、编译基线和整套推理框架。

## FP16 / INT8 KV Cache 实测（2026-09-07）

RTX 3050 Laptop 4GB、Qwen3-0.6B、单请求、CUDA Graph，输入 512/1024/2048 token，输出 128 token。每种模式执行两个独立进程，每个长度预热后测量 5 次 Prefill 和 3 次完整请求；汇总全部轮次：

| 输入 token | FP16 Decode（token/s） | INT8 Decode（token/s） |
| --- | ---: | ---: |
| 512 | 93.75 | 19.28 |
| 1024 | 92.22 | 19.02 |
| 2048 | 85.31 | 18.54 |

单位 token、全层 KV 缓存存储（含 scale）由 **112 KiB 减至 57.75 KiB，减少 48.4%**。当前 INT8 模式吞吐更低，本次未证明整体峰值显存下降。3 个短提示的 teacher-forcing 数值检查中，FP16/INT8 与 Transformers 的 top-1 token 分别有 26/26、24/26 步一致；该小样本结果不代表整体模型准确率。

完整指标、峰值显存、逐步误差、复现命令与简历表述见 [实测报告及原始数据](benchmark_results/resume_20260907/README.md)。报告未进行与上游 nano-vLLM/vLLM 的吞吐对比。

## 相对上游的学习内容

| 模块 | 位置 | 内容 |
| --- | --- | --- |
| INT8 KV Cache | `nanovllm/layers/kv_quant.py` | Triton 实现按 token/head 存储 INT8 K/V 和 FP32 scale，Attention 前反量化 |
| 缓存接入 | `nanovllm/engine/model_runner.py`、`layers/attention.py` | `kv_quant` 开关、缓存及 scale 分配、Prefill/Decode 接入 |
| CUDA 算子 | `nanovllm/ops/` | PyTorch JIT + PyBind11，提供 RMSNorm、Embedding、Softmax、Linear |
| RMSNorm 集成 | `nanovllm/layers/layernorm.py` | 满足条件时调用 FP16 CUDA 算子，其他输入走 PyTorch |
| 调度与分析 | `nanovllm/engine/` | Block/Sequence/Scheduler 学习改动与 NVTX 标记 |
| 端到端基准 | `benchmark_fa2_e2e.py` | Prefill、TTFT、TPOT、Decode token/s，分进程比较缓存模式 |

量化对象是 **KV Cache，不是模型权重**。INT8 缓存本体加 scale 的字节数约为 FP16 的一半，但当前 Attention 会产生 FP16 反量化临时缓冲，不能直接推断总显存减半或吞吐提高。`Config.kv_quant` 当前默认 `True`，比较时请显式设置。

扩展中提供了四种算子，不代表四种算子都已在推理主路径启用；当前多个 Linear、Embedding 和采样调用仍使用 PyTorch。继承的多卡结构未完成本学习分支的验证，建议先使用单卡。

## 安装

需要 Linux / WSL2、NVIDIA GPU、CUDA Toolkit、C++17 编译器和 Python 3.10–3.12。历史实验环境为 RTX 3050 Laptop GPU、PyTorch 2.5.1+cu121；完整依赖版本见 [验证记录](docs/VALIDATION.md)。

```bash
# 先准备与本机 CUDA 匹配的 PyTorch；FlashAttention 需要兼容的构建环境
python -m pip install ninja packaging
python -m pip install flash-attn --no-build-isolation
python -m pip install -e .
```

首次导入会 JIT 编译 CUDA 扩展，编译需要一定时间和内存，可设置 `MAX_JOBS=2`。架构由 PyTorch 自动检测，也可用 `TORCH_CUDA_ARCH_LIST=8.6` 指定 RTX 3050。仓库不包含模型权重。

## 最小运行示例

将 Qwen3-0.6B 的模型权重、配置和 tokenizer 放在本地目录，然后运行：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-0.6B",
    kv_quant=False,             # 改为 True 开启 INT8 KV Cache
    enforce_eager=True,         # 先验证 eager，再比较 CUDA Graph
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_model_len=512,
    max_num_batched_tokens=512,
)
outputs = llm.generate(["Hello"], SamplingParams(temperature=0.6, max_tokens=32))
print(outputs[0]["text"])
```

输出需要与参考实现做质量/数值对比，不能仅以程序不报错判断正确性。

## 复现历史性能测量

```bash
python benchmark_fa2_e2e.py --model /path/to/Qwen3-0.6B --kv-quant off --output benchmark_results/local_fp16.json
python benchmark_fa2_e2e.py --model /path/to/Qwen3-0.6B --kv-quant on --output benchmark_results/local_int8.json
```

在独立进程中运行两种模式，避免显存分配和 CUDA Graph 捕获互相影响。小显存设备可先用 `--lengths 512 --decode-tokens 32`。

已有原始结果：[FP16](benchmark_results/fa2_fp16.json)、[INT8](benchmark_results/fa2_kv_int8.json)。测量包含调度、H2D 准备、模型与采样，不包含 tokenizer；Decode 吞吐为每步延迟中位数的倒数。两份报告的 `output_smoke_test` 均异常，且这些是发布整理前的历史结果，并非当前代码的质量验证。

| 输入 token 数 | FP16 Decode token/s | INT8 Decode token/s |
| --- | ---: | ---: |
| 512 | 100.41 | 20.68 |
| 2048 | 78.37 | 19.95 |
| 8192 | 64.47 | 18.80 |

没有已验证的困惑度（perplexity）结论，也没有支持“端到端提升 30%”的对照结果。

## 代码来源与发布整理

上游基础提交为 `2f21442`，保留上游 Git 历史、[原始 README](README.upstream.md) 与 [MIT LICENSE](LICENSE)。推理框架来自上游；上述模块记录本地学习与修改，不能将整个框架描述为独立从零实现。

发布整理恢复了包元数据、许可证和使用文档，补上 CUDA 源码打包；将扩展 kernel 改为使用当前 CUDA stream（与 PyTorch/CUDA Graph 调度一致），恢复 RMSNorm 的 PyTorch fallback，并明确 Linear 的偶数 K 约束。这些改动发生在独立发布副本中。

配套算子库：[MiniLLM-OP](https://github.com/micuniverse/MiniLLM-OP)。
