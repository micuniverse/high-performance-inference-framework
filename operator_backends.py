"""Explicit RMSNorm ablations for benchmarks; production defaults stay intact.

The PyTorch formulas reproduce nano-vLLM upstream 2f21442 layernorm.py (MIT).
Only RMSNorm methods are changed, before constructing the model/graphs.
"""
import torch

BACKENDS = ("torch-eager", "cuda", "torch-compile", "cuda-compiled-residual")


def rms_torch(self, x):
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x


def add_rms_torch(self, x, residual):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual


def configure_backend(name):
    from nanovllm.layers.layernorm import RMSNorm
    if name not in BACKENDS:
        raise ValueError(name)
    if name == "torch-eager":
        RMSNorm.rms_forward = rms_torch
    elif name == "torch-compile":
        RMSNorm.rms_forward = torch.compile(rms_torch)
        RMSNorm.add_rms_forward = torch.compile(add_rms_torch)
    elif name == "cuda-compiled-residual":
        RMSNorm.add_rms_forward = torch.compile(add_rms_torch)
    # cuda preserves the saved implementation, including eager add+RMSNorm.
    return {
        "name": name,
        "standalone_rmsnorm": "CUDA" if name.startswith("cuda") else name,
        "residual_add_rmsnorm": "torch.compile" if name in ("torch-compile", "cuda-compiled-residual") else "torch eager",
        "other_operators": "unchanged: PyTorch linear/embedding/sampling and existing FlashAttention/RoPE/SwiGLU",
        "scope": "only RMSNorm backend ablation inside this learning fork; not a full upstream-vs-fork benchmark",
    }
