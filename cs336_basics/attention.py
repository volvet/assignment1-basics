import numpy as np
import torch


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)  # (batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.softmax(scores)  # (batch_size, n_heads, seq_len, seq_len)
        output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, head_dim)
        return output

    def softmax(self, x : torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=-1)