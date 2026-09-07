# import torch
# from torch import nn


# class RMSNorm(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         eps: float = 1e-6,
#     ) -> None:
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(hidden_size))

#     @torch.compile
#     def rms_forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         orig_dtype = x.dtype
#         x = x.float()
#         var = x.pow(2).mean(dim=-1, keepdim=True)
#         x.mul_(torch.rsqrt(var + self.eps))
#         x = x.to(orig_dtype).mul_(self.weight)
#         return x

#     @torch.compile
#     def add_rms_forward(
#         self,
#         x: torch.Tensor,
#         residual: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         orig_dtype = x.dtype
#         x = x.float().add_(residual.float())
#         residual = x.to(orig_dtype)
#         var = x.pow(2).mean(dim=-1, keepdim=True)
#         x.mul_(torch.rsqrt(var + self.eps))
#         x = x.to(orig_dtype).mul_(self.weight)
#         return x, residual

#     def forward(
#         self,
#         x: torch.Tensor,
#         residual: torch.Tensor | None = None,
#     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
#         if residual is None:
#             return self.rms_forward(x)
#         else:
#             return self.add_rms_forward(x, residual)
import torch
from torch import nn
import os

from nanovllm.ops import get_ops
_ops = get_ops()
print("loaded _ops:", _ops)

# so_path = os.path.join(os.path.dirname(__file__), "../../minillm_ops.so")
# torch.ops.load_library(so_path)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
             _ops is not None
            and x.is_cuda
            and x.dtype == torch.float16
            and self.weight.dtype == torch.float16
            #and x.is_contiguous()
            #and self.weight.is_contiguous()
            and x.shape[-1] % 8 == 0
        ):
            x2d = x.reshape(-1, x.shape[-1]).contiguous()
            w = self.weight.contiguous()
            #x2d = x.view(-1, x.shape[-1]).contiguous()
            y2d = _ops.rmsnorm_forward(x2d, self.weight.contiguous(), self.eps)
            return y2d.view_as(x)

        x_fp32 = x.float()
        var = x_fp32.square().mean(dim=-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight

    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor):
        # 先不替换，保留原版
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)