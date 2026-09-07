import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


# import torch
# from torch import nn
# import torch.nn.functional as F

# try:
#     from nanovllm.ops1 import get_ops
#     _ops = get_ops()
# except Exception:
#     _ops = None

# class SiluAndMul(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if (
#             _ops is not None
#             and x.is_cuda
#             and x.is_contiguous()
#             and x.dtype in (torch.float16, torch.float32)
#             and x.dim() >= 2
#             and x.shape[-1] % 2 == 0
#         ):
#             x2d = x.view(-1, x.shape[-1]).contiguous()
#             y2d = _ops.silu_and_mul_forward(x2d)
#             return y2d.view(*x.shape[:-1], x.shape[-1] // 2)

#         x, y = x.chunk(2, -1)
#         return F.silu(x) * y