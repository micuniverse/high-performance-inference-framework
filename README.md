# 高性能推理框架

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的 Qwen3 推理项目，围绕 Decode 阶段优化 CUDA Graph 执行、metadata 准备和 RMSNorm，并实现 INT8 KV Cache 存储。

## 主要实现

- **CUDA Graph**：复用捕获的模型前向计算，结合 CPU pinned staging 与 GPU persistent buffer 准备 Decode metadata，减少逐步提交开销。
- **CUDA RMSNorm**：FP16 输入、128-bit 向量化访存、FP32 累加，结合 Warp Shuffle 和 Shared Memory 完成规约，接入模型归一化路径。
- **INT8 KV Cache**：使用 Triton 按 token/head 量化 K/V，保存 FP32 scale，在 FlashAttention 计算前反量化为 FP16。
- **性能测量**：分别统计 Prefill、TTFT、TPOT 和 Decode 吞吐，支持切换 KV Cache、RMSNorm 后端及 Graph 执行方式。

## 性能

测试设备为 **RTX 3050 Laptop 4GB**，模型为 **Qwen3-0.6B**，PyTorch 2.5.1+cu121。吞吐测试使用单请求、FP16 KV Cache，输出长度为 128 token。

### CUDA Graph

固定普通 RMSNorm 与残差 Add+RMSNorm 均使用 `torch.compile`，保留 RoPE、SwiGLU 的编译设置，对比关闭 Graph 与开启 Graph + metadata staging 的 Decode 吞吐。

| 输入长度（token） | 关闭 Graph（token/s） | Graph + staging（token/s） | 吞吐倍数 |
| --- | ---: | ---: | ---: |
| 512 | 24.59 | 97.84 | 3.98× |
| 1024 | 24.05 | 100.06 | 4.16× |
| 2048 | 23.84 | 93.25 | 3.91× |

两种配置各运行两个独立进程，按相反顺序执行，预热后每个输入长度各测量 6 个完整请求。Decode 吞吐按总生成 token 数除以总 Decode 时间计算；计时包含调度、metadata 准备、模型、LM Head 和采样，排除模型加载、编译、Graph 捕获与预热。

这里的编译范围是上述算子；Graph 捕获模型前向，LM Head 和采样在 Graph 外执行。表中倍数对应完整 Graph + staging 路径相对关闭 Graph 的收益。

[测试配置与完整数据](benchmark_results/compiled_graph_20260908/README.md)

### CUDA RMSNorm

开启 CUDA Graph，保持残差 Add+RMSNorm 为 PyTorch eager，仅将普通 RMSNorm 从 **PyTorch eager 实现**替换为自定义 CUDA 实现。在输入长度 512、1024、2048 token 下，Decode 吞吐提升 **6.1%–6.9%**。

该对照与上面的编译归一化配置分别测量，性能收益不叠加。

[RMSNorm 后端对照与原始数据](benchmark_results/operators_20260907/README.md)

### KV Cache 存储

Qwen3-0.6B 全层 KV Cache 的单位 token 存储量由 FP16 的 **112 KiB** 降至 INT8 的 **57.75 KiB**，包含 FP32 scale，减少 **48.4%**。按 2048 token 计算，对应 **224 MiB → 115.5 MiB**。

该指标统计 K/V 与 scale 的存储字节数；运行时还包含模型权重、工作区及 Attention 的 FP16 反量化缓冲。

[缓存测量与原始数据](benchmark_results/resume_20260907/README.md)

## 代码结构

| 模块 | 入口 |
| --- | --- |
| KV Cache 量化与反量化 | `nanovllm/layers/kv_quant.py` |
| 缓存分配、Graph 捕获与 metadata 准备 | `nanovllm/engine/model_runner.py` |
| FlashAttention 与缓存接入 | `nanovllm/layers/attention.py` |
| CUDA RMSNorm 集成 | `nanovllm/layers/layernorm.py` |
| RMSNorm、Embedding、Softmax、Linear 扩展 | `nanovllm/ops/` |
| Prefill / Decode 基准 | `benchmark_fa2_e2e.py` |

模型主路径接入的自定义算子为 RMSNorm；Embedding、Linear 和采样使用 PyTorch。独立算子实现见 [MiniLLM-OP](https://github.com/micuniverse/MiniLLM-OP)。

## 安装与运行

环境要求：Linux / WSL2、NVIDIA GPU、CUDA Toolkit、C++17 编译器、Python 3.10–3.12。先安装与 CUDA 环境匹配的 PyTorch，再安装依赖：

```bash
python -m pip install ninja packaging
python -m pip install flash-attn --no-build-isolation
python -m pip install -e .
```

CUDA 扩展首次加载时进行 JIT 编译，可通过 `MAX_JOBS=2` 控制编译并行度。准备本地 Qwen3-0.6B 权重、配置和 tokenizer 后运行：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-0.6B",
    kv_quant=False,          # True：INT8 KV Cache；False：FP16 KV Cache
    enforce_eager=False,     # 开启 CUDA Graph
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_model_len=512,
    max_num_batched_tokens=512,
)
outputs = llm.generate(["Hello"], SamplingParams(temperature=0.6, max_tokens=32))
print(outputs[0]["text"])
```

## 运行基准

编译归一化后的 Graph 对照：

```bash
MAX_JOBS=2 python run_graph_benchmarks.py \
    --model /path/to/Qwen3-0.6B \
    --rmsnorm-backend torch-compile \
    --modes graph_off graph_staging \
    --output-dir benchmark_results/compiled_graph_run

python summarize_graph_benchmarks.py benchmark_results/compiled_graph_run \
    --modes graph_off graph_staging
```

RMSNorm 后端对照：

```bash
MAX_JOBS=2 python run_operator_benchmarks.py \
    --model /path/to/Qwen3-0.6B \
    --output-dir benchmark_results/operator_run

python summarize_operator_benchmarks.py benchmark_results/operator_run
```

更多环境信息与数值检查见 [验证记录](docs/VALIDATION.md)。

## 来源与许可

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 `2f21442` 提交开发，保留上游 Git 历史、[原始 README](README.upstream.md) 和 [MIT 许可证](LICENSE)。
