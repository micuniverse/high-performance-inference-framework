# RMSNorm 开启 torch.compile 后的 CUDA Graph 对照

## 汇总结果

**普通与残差 RMSNorm 均开启 torch.compile 后，当前完整 CUDA Graph 路径仍然有明显收益。** 两组均保留相同的逐层编译设置，Graph 组还使用框架现有的 CPU metadata staging；这不是全模型编译或单独 staging 的比较。

| 输入 token | 编译 + 无 Graph（token/s） | 编译 + Graph（token/s） | Decode 吞吐提升 | 达到原来的倍数 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 24.59 | 97.84 | +297.94% | 3.98× |
| 1024 | 24.05 | 100.06 | +316.13% | 4.16× |
| 2048 | 23.84 | 93.25 | +291.20% | 3.91× |

关闭 Graph 时为 **23.84–24.59 token/s**，开启时为 **93.25–100.06 token/s**；达到原来的 **3.91–4.16 倍**，即吞吐增加约 **291%–316%**。这里统计 Decode 阶段所有 token / 总耗时，不能与包含 Prefill 的端到端吞吐混用。

| 输入 token | 无 Graph 两轮 | Graph 两轮 |
| --- | --- | --- |
| 512 | 24.63 / 24.54 | 98.21 / 97.46 |
| 1024 | 24.51 / 23.60 | 99.27 / 100.87 |
| 2048 | 23.53 / 24.16 | 94.71 / 91.85 |

所有轮次均纳入汇总，两组各 3 个长度、每长度 6 次测量用完整请求。四次自然语言冒烟输出一致；两组与 Transformers 参考的 26 步 top-1 全部一致，平均 logits 绝对误差均约 0.01082。未在日志中发现编译失败或回退告警，但这不等于做了编译图或 GPU profiler 的完整审计。

Prefill 仍通过普通模型前向执行，不在该 CUDA Graph 中。JSON 中的 Prefill/TTFT 差异只是本次实测，不能据此声称 Graph 捕获并加速了 Prefill；本轮没有定位这些差异的原因，也没有分解 Decode 时间差的全部来源。

## 表述边界

> 在 Qwen3-0.6B 的普通 RMSNorm、残差 Add+RMSNorm、RoPE 和 SwiGLU 保留逐层 torch.compile 的条件下，比较当前完整 CUDA Graph 路径与关闭 Graph。RTX 3050 Laptop 4GB、FP16 KV Cache、batch=1、输入 512–2048 token / 输出 128 token 的 Decode 吞吐由约 24 token/s 提升至 93–100 token/s。

不能将结果描述为“全模型 torch.compile 后仍快四倍”，因为本轮没有编译整个模型；也不能称为 CPU metadata buffer 的独立收益，或所有 GPU、模型及 batch 都适用的倍数。

## 固定的编译范围

两组都使用 `--rmsnorm-backend torch-compile`：普通 RMSNorm 和残差 Add+RMSNorm 按原版公式开启 torch.compile。RoPE、SwiGLU 原有的编译设置也保留。Linear/Embedding 保持 PyTorch 实现，采样函数仍未加 torch.compile。

**这不是整模型 torch.compile，也不是所有函数都开启编译。** 没有使用 `mode="reduce-overhead"`。关闭 Graph 只设置 `enforce_eager=True`，不会关闭上述逐层编译。

| 组别 | 归一化编译 | CUDA Graph | metadata 路径 |
| --- | --- | --- | --- |
| `graph_off` | 普通和残差 norm 均编译 | 关闭 | 普通前向，每步临时 metadata |
| `graph_staging` | 普通和残差 norm 均编译 | 开启 | Graph 固定 GPU buffer + CPU pinned staging 复用 |

对比的是当前完整 Graph 路径与关闭 Graph，不隔离 staging 的单独贡献。两组中的普通 RMSNorm 都使用编译的 PyTorch 公式，不调用自写 RMSNorm CUDA kernel。

## 实验方法

RTX 3050 Laptop 4GB、Qwen3-0.6B、FP16 KV Cache、batch=1。输入长度为 512/1024/2048 token，输出固定 128 token，忽略 EOS。模型权重校验和已重新核对。

顺序为关闭→开启→开启→关闭，各为独立进程。每个长度先预热一次完整请求，再测 5 次 Prefill-only 和 3 次完整请求；两轮合计每组每长度 10 次 Prefill、6 次完整请求、762 个 Decode 步骤。实验共 36 次测量用完整生成、4572 个 Decode 步骤，不选最好轮次。

主指标为所有 Decode token 数 / 所有 Decode 耗时。首 token 在 Prefill 生成，每次请求计 127 个 Decode 步骤。计时包含调度、metadata 准备/H2D、模型主体、LM Head 和采样，每步前后同步 CUDA；排除模型加载、编译、Graph 捕获、预热和 tokenizer。所有统计与请求输入 SHA256 保存在 JSON。

两组使用相同的 0.9 显存利用率自动分配 KV Cache，实际容量可能不同，见元数据。运行于日常 WSL2/桌面环境，没有锁定 GPU 频率。结果仅适用于这些模型、算子配置、batch 和长度，不是多请求吞吐或稀疏批次的验证。

普通归一化、残差归一化和 Graph 的优化作用不同：逐层编译可融合公式中的小操作；Graph 可重放捕获的模型主体，减少逐步重新执行主体 Python/CUDA launch 流程的开销。LM Head、采样、调度仍在图外。本轮没有 profiler 因果分解，不能把全部时间差都称为 CPU launch 时间。

## 数值检查

使用前一轮重新生成且模型权重校验一致的 Transformers FP16 eager Attention 参考，进行相同输入及参考历史 token 的 teacher-forcing logits 比较。两组在 3 个短提示、26 步上的 top-1 均为 26/26，逐步误差见 `check_graph_off.json`、`check_graph_staging.json`。这不是全面准确率评估；各性能轮次另保留自然语言自由采样输出。

## 复现

在仓库根目录、安装依赖的 Python 环境中运行：

```bash
MAX_JOBS=2 python run_graph_benchmarks.py --model /path/to/Qwen3-0.6B \
  --rmsnorm-backend torch-compile --modes graph_off graph_staging \
  --output-dir benchmark_results/compiled_graph_new_run
python summarize_graph_benchmarks.py benchmark_results/compiled_graph_new_run \
  --modes graph_off graph_staging
```

单次运行可使用 `benchmark_fa2_e2e.py --rmsnorm-backend torch-compile --cuda-graph off/on`，固定 `--kv-quant off`，Graph 开启时保留 `--decode-staging on`。

环境及校验和见 `environment.json`，四份 `*_round*.json` 包含逐步原始延迟，`summary.json` 汇总全部轮次。前一轮未编译残差 norm 的结果见 [Graph/staging 报告](../graph_20260908/README.md)；它使用自写普通 RMSNorm，与本轮并非仅一个编译开关的差别，不能把跨批次差异直接称为编译的独立收益。
