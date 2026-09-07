import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    #创建一个 runner 实例。多卡时会被多个进程分别创建（每个进程一个 rank）
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        #初始化 torch.distributed 进程组，后端用 NCCL（GPU 通信）。
        #tcp://localhost:2333 是 rendezvous 地址（本机端口），用于让各进程彼此发现并建立通信。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        #把当前进程绑定到对应 GPU（rank=0→cuda:0，rank=1→cuda:1 …）
        torch.cuda.set_device(rank)
        #默认 dtype 改成模型 dtype（比如 fp16/bf16）
        # 默认 device 改成 cuda
        # 这样后面创建模块/张量默认就在 GPU 上、用模型 dtype。
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        #创建模型结构并加载权重（load_model 一般会从 safetensors/pt 文件 load）。
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        #创建采样器（top-k/top-p/temperature 等）。
        self.sampler = Sampler()
        #warmup_model()：通常做一次空 forward 或固定 shape forward，让 kernel 编译、缓存、初始化通信等
        # allocate_kv_cache()：按显存预算分配 KV cache，并注入到各层 attention
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        #恢复默认设置，避免影响外部代码。
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多卡才启用共享内存控制。
        # rank0：创建共享内存段 nanovllm，大小 1MB。
        # dist.barrier()：所有 rank 同步，确保共享内存创建好后其它 rank 再去打开。
        # rank>0：打开已存在的共享内存，然后直接进入 loop()，开始等待 rank0 指令。
        # 注意：这意味着 rank>0 的进程会在 __init__ 里阻塞在 loop，不会返回给上层调用者；上层只和 rank0 交互。
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                #所有进程必须都执行到这一行，程序才会继续往下运行
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    #作用：rank>0 的命令循环：一直等 rank0 发来的 method call。
    def loop(self):
        while True:
            # read_shm() 阻塞等待事件
            # 收到方法名+参数后调用 self.call
            # 如果方法是 exit，退出循环（rank>0 进程结束）
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        #等待 rank0 event.set()，表示有新命令。
        self.event.wait()
        #共享内存前 4 个字节存数据长度 n（小端）。
        n = int.from_bytes(self.shm.buf[0:4], "little")
        #读取后面 n 字节的数据并反序列化，得到：method_name,args
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        #清除事件，返回命令
        self.event.clear()
        return method_name, args

    #rank0 把方法名+参数写入共享内存，并通知所有 worker（event set）。
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        #序列化命令并得到长度。
        data = pickle.dumps([method_name, *args])
        n = len(data)
        #写入共享内存：前 4 bytes 写长度后面写 payload
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        #rank0 通知所有 worker：有新命令可以读了。
        # 这里 self.event 是 list[Event]（每个 worker 一个 event），所以能逐个唤醒。
        for event in self.event:
            event.set()

    #作用统一方法调用入口：多卡时 rank0 先广播命令给其它 ranks,然后本 rank 调用自身方法
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

#根据当前 GPU 显存情况，动态计算最多能分配多少个 KV cache block，
# 然后一次性在 GPU 上申请一块大的 KV cache 张量，并把每一层 attention 模块的 k_cache / v_cache 指向这块张量对应的切片。
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        #本进程历史“峰值已分配显存”
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        #当前“已分配显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        #num_key_value_heads：模型的 KV heads（MHA/MQA/GQA 都会有）
        #如果做 tensor parallel（多 GPU 切分），把 KV heads 按 GPU 均分
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        #根据显存预算算最多能分配多少 block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        #真正申请一块连续的 KV cache 张量
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        #把每层模块的 k_cache / v_cache 指向对应切片
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            #把 block 级别的映射 → 展开成 token 级别的 slot mapping
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    #end = start + seq.last_block_num_tokens 
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
        # st = getattr(self, "decode_staging", None)
        # if (not self.enforce_eager) and hasattr(self, "graphs") and len(seqs) <= 512 and (st is not None):
        #     bs = len(seqs)
        #     if bs == 0:
        #         set_context(False)
        #         return self.decode_staging["input_ids"][:0], self.decode_staging["positions"][:0]
        #     bucket = next(x for x in self.graph_bs if x >= bs)
        #     st = self.decode_staging
        #     st["input_ids"][:bucket].zero_()
        #     st["positions"][:bucket].zero_()
        #     st["slot_mapping"][:bucket].fill_(-1)
        #     st["context_lens"][:bucket].fill_(1)
        #     st["block_tables"][:bucket].fill_(-1)

        #     for i, seq in enumerate(seqs):
        #         st["input_ids"][i] = seq.last_token
        #         st["positions"][i] = len(seq) - 1
        #         st["context_lens"][i] = len(seq)
        #         st["slot_mapping"][i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        #         row = st["block_tables"][i]
        #         for j, b in enumerate(seq.block_table):
        #             row[j] = b

        #     set_context(False)
        #     return st["input_ids"][:bs], st["positions"][:bs]

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
        # if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        #     return self.model.compute_logits(self.model(input_ids, positions))
        # else:
        #     bs = input_ids.size(0)
        #     bucket = next(x for x in self.graph_bs if x >= bs)
        #     graph = self.graphs[bucket]
        #     graph_vars = self.graph_vars

        #     if input_ids.device.type == "cpu" and hasattr(self, "decode_staging"):
        #         st = self.decode_staging
        #         graph_vars["input_ids"][:bucket].copy_(st["input_ids"][:bucket], non_blocking=True)
        #         graph_vars["positions"][:bucket].copy_(st["positions"][:bucket], non_blocking=True)
        #         graph_vars["slot_mapping"][:bucket].copy_(st["slot_mapping"][:bucket], non_blocking=True)
        #         graph_vars["context_lens"][:bucket].copy_(st["context_lens"][:bucket], non_blocking=True)
        #         graph_vars["block_tables"][:bucket].copy_(st["block_tables"][:bucket], non_blocking=True)
        #     else:
        #         context = get_context()
        #         graph_vars["input_ids"][:bucket].zero_()
        #         graph_vars["positions"][:bucket].zero_()
        #         graph_vars["input_ids"][:bs].copy_(input_ids, non_blocking=True)
        #         graph_vars["positions"][:bs].copy_(positions, non_blocking=True)

        #         graph_vars["slot_mapping"][:bucket].fill_(-1)
        #         graph_vars["slot_mapping"][:bs].copy_(context.slot_mapping, non_blocking=True)

        #         graph_vars["context_lens"][:bucket].fill_(1)
        #         graph_vars["context_lens"][:bs].copy_(context.context_lens, non_blocking=True)

        #         graph_vars["block_tables"][:bucket].fill_(-1)
        #         graph_vars["block_tables"][:bs, :context.block_tables.size(1)].copy_(context.block_tables, non_blocking=True)

        #     graph.replay()
        #     return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    #作用：提前为一组常见 batch size（bs）把“decode 前向推理”用 CUDA Graph 录制下来，
    # 之后运行时只要把输入张量内容填好、选择对应 bs 的 graph 回放，
    # 就能减少 Python 调度/Kernel launch 开销，显著提升 decode 阶段吞吐和稳定性（尤其是小 batch、多步生成时）
    @torch.inference_mode() #推理模式，关闭 autograd，减少开销、避免图捕获过程中引入梯度状态。
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        #max_bs：用于录 graph 的最大 batch size，取 max_num_seqs 和 512 的较小值（避免录太大、占太多显存/时间）。
        # max_num_blocks：最大上下文长度按 block_size 切块后需要的 block 数。
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # input_ids：decode 阶段通常是一 token / 每序列一个 token，所以 shape 是 [bs]
        # positions：对应每条序列当前 token 的 position（同样 [bs]）
        # slot_mapping：本步 token 的 KV 写入 cache 的 slot（[bs]）
        # context_lens：每条序列已有上下文长度（[bs]），decode attention 需要它
        # block_tables：每条序列的 block 映射表（[bs, max_num_blocks]）
        # outputs：模型输出的 hidden（这里是 [bs, hidden_size]），用于接住输出，避免捕获中出现临时分配
        # CUDA Graph 要求捕获的内存分配/形状尽量固定，所以这里把所有可能用到的输入都预先分配好，后面只切片 [:bs] 使用。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup 的意义：
            # 触发 kernel 选择/编译（有些是首次运行才会编译的 triton kernel）
            # 让 PyTorch/allocator 做一次必要的内存准备
            # 避免把“首次运行开销/动态分配”录进图里（图捕获对动态分配很敏感）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            #torch.cuda.graph(graph, pool)：进入捕获上下文
            # 捕获期间发生的 GPU kernel launch 序列会被录制
            # 之后 replay 时会按同样顺序执行，省掉 Python 调度和 launch 开销
            # 传 self.graph_pool（可能为 None 或某个 pool）：
            # 第一个 graph 捕获后会创建 pool
            # 后续 graph 使用同一 pool
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            #记录 memory pool 并保存 graph
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        # self.decode_staging = dict(
        #     input_ids=torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True),
        #     positions=torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True),
        #     slot_mapping=torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True),
        #     context_lens=torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True),
        #     block_tables=torch.empty(max_bs, max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True),
        # )
