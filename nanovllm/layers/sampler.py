import torch
from torch import nn

from nanovllm.ops import load_ops
_ops = load_ops()


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) #.float()将半精度改为单精度
        probs = torch.softmax(logits, dim=-1)
        # probs = _ops.softmax_forward(logits)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
