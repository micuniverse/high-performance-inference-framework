import torch
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# model_path = "~/huggingface/Qwen3-0.6B/"
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["介绍一下CUDA Graph的作用"] * 32

# 先来一次预热，不计入分析
_ = llm.generate(prompts, sampling_params)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("decode_test")
outputs = llm.generate(prompts, sampling_params)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print(outputs[0]["text"][:200])