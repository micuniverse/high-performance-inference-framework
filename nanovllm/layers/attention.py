import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

#这个文件做两件事：
# 1.把新算出来的 K/V 写进 KV cache（用 Triton kernel store_kvcache_kernel 加速）
# 2.执行 attention
#   prefill（提示词阶段，批量算注意力）
#   decode（逐 token 生成阶段，使用 KV cache 做增量注意力


#作用
# 对每个 token（或每条 KV 向量）：
#   读取 key_ptr/value_ptr 中的 K/V 向量
#   根据 slot_mapping 找到它应该写入 cache 的“槽位 slot”
#   把 K/V 向量写入 k_cache_ptr/v_cache_ptr 的对应位置
# 这相当于：把本次算出来的 KV 追加/散写到全局 cache 里，slot 可能不是连续的（由 block manager 决定）。
@triton.jit #让 Triton JIT 编译这个 kernel，在 GPU 上跑
def store_kvcache_kernel(
    key_ptr,
    key_stride, # 输入的“第一维步长”（每个 idx 对应一条向量，idx 变动时内存跳多少）
    value_ptr,
    value_stride,
    k_cache_ptr,# 输出 cache 的指针（通常是预分配的大张量的某个 view）
    v_cache_ptr,
    slot_mapping_ptr, # 长度为 N 的映射表，slot_mapping[idx] 告诉你第 idx 条 KV 应该写到 cache 的哪个 slot
    D: tl.constexpr, # 编译期常量（向量长度 = num_heads*head_dim），用 constexpr 能让 kernel 更好优化
):
    #Triton 的执行模型：你 launch 的 grid 是 (N,)
    # #tl.program_id(0) 返回当前程序实例的 id，相当于 “当前处理第 idx 条数据”
    idx = tl.program_id(0)
    #读出这条 KV 要写入 cache 的 slot（整数）
    slot = tl.load(slot_mapping_ptr + idx)
    #表示这条数据不需要写入 cache（无效/跳过）
    if slot == -1: return
    # tl.arange(0, D) 得到 [0,1,2,...,D-1]
    # idx * key_stride 表示第 idx 条向量的起始位置
    # 加上 arange 得到这条向量的所有元素地址偏移
    # 这相当于把输入 key/value 当成展平的 2D：[N, D]
    # 注意：这要求输入在内存上“最后一维连续”，否则地址会错或性能差，所以后面 Python wrapper 会断言 stride。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    #从输入加载这条 KV 向量（长度 D）
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # cache 也当成 [num_slots, D] 的展平视图
    # slot * D 是目标 slot 的起始位置
    cache_offsets = slot * D + tl.arange(0, D)
    #把 key/value 写入 cache 的对应 slot
    # 这样 decode 阶段就能按 slot 找到历史 KV
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)



# 作用
# 这是 Python 包装函数，负责：
# 从 key.shape 推出 N、head 相关参数
# 做一堆 stride 断言，确保内存布局符合 kernel 的假设
# 以 (N,) grid launch Triton kernel
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    #把 [num_heads, head_dim] 合并成一条向量长度 D。kernel 以 D 为单位读写。
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    #作用
    #保存 attention 配置，并给 k_cache/v_cache 做占位。真正 cache 之后会被 allocate_kv_cache() 注入成大张量的切片。
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_quant: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        #softmax scale（通常是 1/sqrt(head_dim)）
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_quant = kv_quant
        #先给 k_cache 和 v_cache 创建一个占位符（placeholder），真正的 KV cache 会在后面由 model_runner.allocate_kv_cache() 分配并替换。3
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = torch.tensor([])

    #作用概览
    # 从全局 context 取运行时信息（slot_mapping、是否 prefill、block_table 等）
    # 如果已经分配了 cache：把本步的 k/v 写进 cache（store_kvcache）
    # 根据 prefill/decode 走两条注意力路径：
    # prefill：对完整输入序列计算 attention
    # decode：用 kv cache 做增量 attention
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # get_context() 返回一次推理/一次 batch 的运行时上下文（vLLM 风格），里面包括：
        # slot_mapping：当前 token 写入 cache 的位置
        # is_prefill：是否处于 prefill
        # block_tables：prefix cache / paged attention 相关的块表
        # cu_seqlens_*、max_seqlen_*：varlen attention 所需的序列边界信息
        # context_lens：decode 时每条序列当前长度
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        #numel() 非 0 表示 cache 已经被注入（不是空 tensor）
        if k_cache.numel() and v_cache.numel():
            #使用 context.slot_mapping 决定每条写入到哪个 slot
            # store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if self.kv_quant:
                from nanovllm.layers.kv_quant import store_kvcache_int8
                store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, 
                                   self.v_scale, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # print("k_cache dtype:", k_cache.dtype)
            # print("v_cache dtype:", v_cache.dtype)
            # print("k_scale shape:", None if self.k_scale is None else self.k_scale.shape)
            # print("v_scale shape:", None if self.v_scale is None else self.v_scale.shape)
            if context.block_tables is not None:    # prefix cache
                # k, v = k_cache, v_cache
                if self.kv_quant:
                    from nanovllm.layers.kv_quant import dequant_kvcache
                    k = dequant_kvcache(k_cache, self.k_scale, self.num_kv_heads, self.head_dim)
                    v = dequant_kvcache(v_cache, self.v_scale, self.num_kv_heads, self.head_dim)
                else:
                    k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                      max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                      max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            #                             cache_seqlens=context.context_lens, block_table=context.block_tables, 
            #                             softmax_scale=self.scale, causal=True)
            if self.kv_quant:
                from nanovllm.layers.kv_quant import dequant_kvcache
                k_fp = dequant_kvcache(k_cache, self.k_scale, self.num_kv_heads, self.head_dim)
                v_fp = dequant_kvcache(v_cache, self.v_scale, self.num_kv_heads, self.head_dim)
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_fp, v_fp,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)       
        return o
