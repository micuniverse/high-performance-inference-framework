# 发布验证记录

自写算子接入前后的四种 RMSNorm 配置对照已完成，见 [算子复现报告](../benchmark_results/operators_20260907/README.md)。四种配置均完成相同的 26 步数值检查，性能测试包含八个独立进程。结果区分 eager 和原版编译设置，没有复现 30% 提升。

后续完整 Prefill/Decode 对照测试及 26 步 teacher-forcing logits 检查已完成，见 [2026-09-07 实测报告](../benchmark_results/resume_20260907/README.md)。下文记录首次发布时的验证范围；后续小样本检查仍不能替代困惑度和长上下文质量评估。

日期：2026-09-07。GPU：NVIDIA GeForce RTX 3050 Laptop GPU（4 GB）。环境：WSL2/Linux、CUDA Toolkit 12.1、Python 3.10、PyTorch 2.5.1+cu121、FlashAttention 2.8.3、Transformers 5.2.0。

## 已执行

- Python 源文件语法检查通过。
- Wheel 构建通过，包含 `nanovllm/ops` 的两个 CUDA 源文件。
- 本地 CUDA 扩展 JIT 编译、加载通过。
- FP16、INT8 两种缓存模式分别在独立进程中执行短序列 Prefill/Decode 及自然语言生成冒烟测试，退出码均为 0。

```bash
MAX_JOBS=2 python benchmark_fa2_e2e.py \
  --model /path/to/Qwen3-0.6B --kv-quant off \
  --lengths 32 --decode-tokens 8 --prefill-repeats 1 \
  --output benchmark_results/publication_smoke_fp16.json
MAX_JOBS=2 python benchmark_fa2_e2e.py \
  --model /path/to/Qwen3-0.6B --kv-quant on \
  --lengths 32 --decode-tokens 8 --prefill-repeats 1 \
  --output benchmark_results/publication_smoke_int8.json
```

原始输出：[FP16 短测试](../benchmark_results/publication_smoke_fp16.json)、[INT8 短测试](../benchmark_results/publication_smoke_int8.json)。测试脚本启用 CUDA Graph。自然语言部分另外生成 32 个 token，只用于检查输出现象。

发布副本将 CUDA 扩展中的默认 stream 改为当前 stream 后，本次短测试不再出现历史报告中的重复感叹号现象。这里没有通过逐项回滚实验确认单一根因。生成样本很短，并且可能截断 Qwen3 的 thinking 内容，不能认定完整回答质量通过。

一次 Prefill 重复会包含首次编译/运行开销；本次测试的计时不用于稳定性能比较。没有完成困惑度评估、与 Transformers 的 logits 对比、长上下文正确性或多卡验证。

## 历史数据与限制

`fa2_fp16.json` 和 `fa2_kv_int8.json` 是发布修改前的记录，两份输出均为异常重复感叹号。INT8 Decode 吞吐低于 FP16；数据保留供分析，不能证明 INT8 加速或模型质量达标。

INT8 Cache 前向会反量化到 FP16 临时缓冲。缓存本体的理论存储减少不能等同于总峰值显存降低相同比例。没有证据支持原代码注释中的困惑度提升小于 0.5 的说法，发布文档已移除该无验证结论。

## 发布整理改动

恢复上游许可证、包元数据，新增中文说明与 CUDA 源码打包；CUDA kernel launcher 使用当前 stream；恢复 RMSNorm 非 FP16/不适用向量化时的 PyTorch fallback；Linear 检查偶数 K。原学习目录未覆盖，发布代码位于独立副本。
