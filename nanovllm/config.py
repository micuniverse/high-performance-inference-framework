import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kv_quant: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    #新增
    force_fp16_for_ampere_gaming: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        # 3050/3050Ti 这类本地单卡，第一版强制 fp16 更稳
        if self.force_fp16_for_ampere_gaming and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # RTX 3050 是 sm_86
            if major == 8 and minor == 6:
                self.hf_config.torch_dtype = torch.float16
