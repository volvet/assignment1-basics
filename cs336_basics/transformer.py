import numpy as np
import torch

from .attention import MultiHeadSelfAttention
from .rms_norm import RMSNorm
from .position_wise_feed_forward import PositionWiseFeedForward
from .embedding import Embedding


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


class TransformerLM(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 content_length: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 num_layers: int,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.vocab_size = vocab_size
        self.content_length = content_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.num_layers = num_layers
        self.token_embedding = Embedding(vocab_size, d_model, **self.factory_kwargs)
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             n_heads=num_heads,
                             d_ff=d_ff,
                             use_rope=True,
                             theta=rope_theta,
                             device=device,
                             dtype=dtype)
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = torch.nn.Linear(d_model, vocab_size, bias=False, **self.factory_kwargs)

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.output_linear(x)
        return x


