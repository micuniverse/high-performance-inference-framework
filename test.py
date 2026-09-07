import torch
from nanovllm.layers.layernorm import RMSNorm

torch.manual_seed(0)
device = "cuda"

B, L, D = 2, 8, 1024
x = torch.randn(B, L, D, device=device, dtype=torch.float16)

mod = RMSNorm(D).to(device=device, dtype=torch.float16)
w = mod.weight.detach().clone()

# 你的 CUDA 路线
x2d = x.reshape(-1, x.shape[-1]).contiguous()
y_cuda = mod.rms_forward(x)

# PyTorch 参考路线
x_fp32 = x.float()
var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
y_ref = (x_fp32 * torch.rsqrt(var + mod.eps)).to(torch.float16) * w

diff = (y_cuda - y_ref).abs()
print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())
print("allclose:", torch.allclose(y_cuda, y_ref, atol=1e-2, rtol=1e-2))