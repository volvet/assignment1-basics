import numpy as np
import torch

from .attention import MultiHeadSelfAttention
from .rms_norm import RMSNorm
from .position_wise_feed_forward import PositionWiseFeedForward


class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 theta: int = 10000,
                 max_seq_len: int = 2048,
                 use_rope: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.attn = MultiHeadSelfAttention(d_model=d_model,
                                           n_heads=n_heads,
                                           use_rope=use_rope,
                                           theta=theta,
                                           max_seq_len=max_seq_len,
                                           device=device,
                                           dtype=dtype)
        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(d_model=d_model,
                                             d_ff=d_ff,
                                             device=device,
                                             dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms1(x))
        x = x + self.ffn(self.rms2(x))
        return x