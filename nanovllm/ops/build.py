import os
from pathlib import Path
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent

def load_ops():
    return load(
        name="nanovllm_rmsnorm_ops",
        sources=[str(_THIS_DIR / "rmsnorm_kernel.cu"),
                 str(_THIS_DIR/ "linear_kernel.cu"),],   
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        verbose=True,
    )