import numpy as np
import torch

class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, 
                 theta : float,
                 d_k : int,
                 max_seq_len : int,
                 device : torch.device = None,
                 dtype : torch.dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, **factory_kwargs).float() / d_k))
        position = torch.arange(0, max_seq_len, **factory_kwargs).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq) # (max_seq_len, d_k/2)
        self.register_buffer("cos_cached", sinusoid_inp.cos(), persistent=False)
        self.register_buffer("sin_cached", sinusoid_inp.sin(), persistent=False)

    def forward(self, x : torch.Tensor, token_position : int) -> torch.Tensor:
        if token_position is None:
            token_position = torch.arange(self.max_seq_len, device=x.device)
        cos = self.cos_cached[token_position]
        sin = self.sin_cached[token_position]

        if cos.shape[-2] > x.shape[-2]:
            cos = cos[:x.shape[-2], :]
            sin = sin[:x.shape[-2], :]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin # (1, seq_len, d_k/2)
        out2 = x1 * sin + x2 * cos # (1, seq_len, d_k/2)
        out = torch.stack((out1, out2), dim=-1).flatten(-2) # (1, seq_len, d_k)
        return out
