# 自写 RMSNorm 接入前后的性能复现

## 实测结论

**没有复现“相对原版 nano-vLLM 提升 30%”。** 相对未编译的 eager 对照，当前自写 RMSNorm 版本有 6.1%–6.9% 的 Decode 吞吐收益；相对原版 RMSNorm 的编译设置，当前保存版本反而更慢。

| 输入 token | A：eager | B：当前 CUDA | C：原版 norm 编译设置 | D：CUDA + 编译残差 | A→B 吞吐变化 | C→B 吞吐变化 | C→D 吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 89.07 | 95.05 | 104.10 | 104.04 | +6.71% | -8.70% | -0.06% |
| 1024 | 88.21 | 94.28 | 102.92 | 102.37 | +6.88% | -8.40% | -0.54% |
| 2048 | 81.61 | 86.55 | 94.29 | 94.04 | +6.06% | -8.21% | -0.27% |

表中为单请求 **Decode token/s**，包含各轮全部 Decode 计时。不是整批服务吞吐，也不是独立 RMSNorm kernel 的速度。

相对 eager：Prefill 吞吐提升 **7.9%–9.9%**，包含 Prefill 的端到端输出吞吐提升 **6.3%–7.6%**。不能把这些指标混用，也没有哪个已测指标支持“30%”。

相同编译残差条件下，C→D 的 Decode 变化为 **−0.06% 至 −0.54%**，没有观察到明确优势；Prefill 反而下降 **6.5%–6.9%**。因此不得将 A→B 的 eager 收益写成相对 `torch.compile` 的收益。

B→D 仅保留残差方法的 `torch.compile`，本轮 Decode 从当前保存版本提高约 **8.6%–9.5%**。这说明残差编译设置是重要的实验变量，但该收益不是“自写普通 RMSNorm 比原版编译 RMSNorm 更快”。D 是本次实验配置，框架默认仍为 B。

## 可用于简历的准确表述

> 实现 FP16 CUDA RMSNorm，采用向量化访存、Warp/Block 规约及 FP32 累加，并接入 Qwen3-0.6B 推理；在 RTX 3050 Laptop 4GB、FP16 KV Cache、batch=1、CUDA Graph 开启、输入 512–2048 token / 输出 128 token 条件下，相对 PyTorch eager RMSNorm 基线，Decode 吞吐提升 6.1%–6.9%。

如果简历要强调相对原版 nano-vLLM 的性能提升，就不能使用上述比例，因为原版 norm 开启了 torch.compile。也可以只写实现、集成和可复现评测，不写加速百分比。

## 面试回答示例

> 我实现的是普通 RMSNorm CUDA kernel：FP16 向量化加载，FP32 计算平方和，通过 Warp Shuffle 和共享内存做规约，再乘权重写回。模型主路径实际启用的是 RMSNorm，其余几个学习算子的调用目前没有打开。

> 我用同一份 Qwen3-0.6B 权重和 FP16 KV Cache，固定单请求输入长度及 128 token 输出，排除加载、编译和预热后，在独立进程中做两轮反向顺序对照。Decode 吞吐按生成的 Decode token 总数除以总耗时计算，提升比例是 `(替换后吞吐 / 替换前吞吐 - 1) × 100%`。例如 512 token 输入时，eager 对照为 89.07 token/s，自写版本为 95.05 token/s，提升约 6.71%。

> 我后来补做了编译基线的消融测试。原版 RMSNorm 有 torch.compile，不能把它和 eager 混在一起比较；相同残差编译设置下，自写 kernel 的 Decode 与编译版本基本持平，Prefill 更慢。原先记录的 30% 没有找到足够证据复现，因此当前简历改用明确 eager 基线的实测结果。

> 数值检查使用相同输入和参考生成 token，比对 Transformers 的 logits。四种配置在 3 个短提示、26 步上的 top-1 一致，但这只是小样本检查，不是完整准确率评估。

## 原始记录

共 **8 个独立性能测试进程、72 次测量用完整生成请求、9,144 个 Decode 步骤**，另有未计入吞吐的预热和数值检查。所有轮次、逐步延迟、TTFT、Prefill、allocator 峰值显存及自由采样文本均在 `*_round*.json`，聚合表在 `summary.json`，小样本数值对比在 `check_*.json`。

## 这次比较的对象

本次使用相同的学习框架、Qwen3-0.6B 权重、FP16 KV Cache 和其他层实现，仅在模型构建、预热及 CUDA Graph 捕获前切换 RMSNorm 方法。这是对当前代码的受控复现，不是找回历史实验记录，也不是完整上游 nano-vLLM 与学习分支的全工程对比。

本地原版 `nano-vllm/nanovllm/layers/layernorm.py` 与上游提交 `2f21442` 的普通 RMSNorm、残差 Add+RMSNorm 方法均有 `@torch.compile`。当前学习版本将普通 RMSNorm 换成自写 CUDA kernel，但残差方法已不带该装饰器。因此需区分下面四种配置：

| 标记 / 参数 | 普通 RMSNorm | 残差 Add+RMSNorm | 用途 |
| --- | --- | --- | --- |
| A / `torch-eager` | 原版公式，eager | eager | 未编译对照，不是原版默认设置 |
| B / `cuda` | 自写 CUDA | eager | 当前保存的学习实现 |
| C / `torch-compile` | 原版公式，torch.compile | torch.compile | 恢复原版 RMSNorm 的编译设置 |
| D / `cuda-compiled-residual` | 自写 CUDA | torch.compile | 本次新增的受控配置，保留残差编译 |

**A→B** 和 **C→D** 分别只改变普通 RMSNorm 后端。**C→B** 更接近原版 RMSNorm 类替换为当前学习类的实际差异，但同时包含残差融合编译设置变化，不能把差异全归因于单个 CUDA kernel。

## 实验方法

RTX 3050 Laptop 4GB，Qwen3-0.6B，batch=1，CUDA Graph 开启，输入 512 / 1024 / 2048 token，输出 128 token。顺序为 A、B、C、D，再 D、C、B、A，每项运行独立进程。每个长度先预热一次完整请求，再测 5 次 Prefill-only 和 3 次完整生成。每种配置、每个长度共 10 次 Prefill、6 次完整请求、762 个 Decode 步骤。

全量统计、不选最好的一轮。Decode 吞吐 = 总 Decode token 数 / 总 Decode 耗时；首 token 属于 Prefill，128 token 输出对应 127 个 Decode 步骤。另记录 TTFT、TPOT 和包含 Prefill 的端到端输出吞吐。计时包含调度、H2D 准备、模型和采样，排除 tokenizer、模型加载、编译和 Graph 捕获。源码版本、工作负载 SHA256 和参数均记录在 JSON；汇总时检查一致。

四组均使用 `gpu_memory_utilization=0.9` 自动预分配 FP16 KV Cache，容量可能随启动时内存状态变化，实际容量保存在元数据中。没有锁定 GPU 频率，采用顺序反转和重复进程降低顺序偏差；结果限于本次桌面/WSL2 环境。

## 实际接入情况与 kernel 实现

当前推理主路径只有 RMSNorm 启用了本地 CUDA 扩展。`embed_head.py` 的 Embedding、`linear.py` 的 Linear、`sampler.py` 的 Softmax 扩展调用均处于注释状态；不能声称这些 kernel 共同带来了本次端到端变化。

按 Qwen3-0.6B 的 28 层结构和当前 forward 分支，每次前向有 57 次普通 RMSNorm：56 次 Q/K head norm（hidden dimension 为 128）及第一层输入 norm（hidden dimension 为 1024）；另有 56 次残差 Add+RMSNorm。普通 norm 才使用自写 kernel，残差融合方法由 PyTorch 执行。

自写 kernel 位于 `nanovllm/ops/rmsnorm_kernel.cu`：每个 block 处理一行，256 个线程；每次向量化读取 8 个 FP16，FP32 累加平方和；Warp Shuffle + Shared Memory 完成跨 Warp 规约，再用共享的逆 RMS 和权重写回 FP16。Python 入口包含 reshape/contiguous，这些开销也计入端到端结果。

实际 Q/K 的维度为 128，不能拿 `[32,4096]` 等独立算子基准的加速倍数直接代替模型吞吐提升。`torch.compile` 已能融合原始公式；自写 kernel 的优势需相对具体编译基线实测。不同形状、内存布局、残差融合和其余层耗时都可能影响收益，本次未做 profiler 因果归因。

## 数值核对

复用同一权重生成的 Transformers FP16 eager Attention 参考，3 个短英文提示共 26 个步骤；通过 teacher forcing 保持各实现的输入及历史 token 完全一致，分别核对 Prefill/Decode logits。四种配置的 top-1 均为 26/26 一致，逐步误差保存在 `check_*.json`。该小样本检查不是整体准确率或长上下文质量评估；性能测量另用固定随机 token 输入。

## 复现命令

在仓库根目录、已安装依赖的 Python 环境执行：

```bash
MAX_JOBS=2 python run_operator_benchmarks.py --model /path/to/Qwen3-0.6B \
  --output-dir benchmark_results/operators_new_run
python summarize_operator_benchmarks.py benchmark_results/operators_new_run
```

单独切换配置仍使用用户原来的第二个 benchmark：

```bash
python benchmark_fa2_e2e.py --model /path/to/Qwen3-0.6B --kv-quant off \
  --rmsnorm-backend torch-eager --lengths 512 1024 2048 --decode-tokens 128 \
  --prefill-repeats 5 --decode-repeats 3 --warmup-runs 1 --output /tmp/eager.json
```

将 `--rmsnorm-backend` 依次改为 `cuda`、`torch-compile`、`cuda-compiled-residual`。这些选项仅用于 benchmark，未修改框架默认 RMSNorm 后端。

数值核对使用 `check_reference.py` 的 `--rmsnorm-backend` 参数，参考文件生成方法见 [前一份量化实验报告](../resume_20260907/README.md)。`environment.json` 保留模型校验和及本轮预定顺序。
