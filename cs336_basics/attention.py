import numpy as np
import torch
import torch.nn as nn
from .rotary_position_embedding import RotaryPositionEmbedding

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)  # (batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = self.softmax(scores)  # (batch_size, n_heads, seq_len, seq_len)
        output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, head_dim)
        return output

    def softmax(self, x : torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=-1)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 use_rope: bool = True,
                 theta: int = 10000,
                 token_positions: torch.Tensor = None,
                 max_seq_len: int = 2048,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        super().__init__()
        assert d_model % n_heads == 0 # d_model must be divisible by n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.attention = ScaledDotProductAttention(device=device, dtype=dtype)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionEmbedding(theta=theta, d_k=d_model//n_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)
            self.token_positions = token_positions
        self.wq = torch.nn.Linear(d_model, d_model, bias=False, **self.factory_kwargs)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False, **self.factory_kwargs)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False, **self.factory_kwargs)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False, **self.factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.wq.weight)
        torch.nn.init.xavier_uniform_(self.wk.weight)
        torch.nn.init.xavier_uniform_(self.wv.weight)
        torch.nn.init.xavier_uniform_(self.wo.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.wq(x) # (batch_size, seq_len, d_model)
        k = self.wk(x) # (batch_size, seq_len, d_model)
        v = self.wv(x) # (batch_size, seq_len, d_model)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)
        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(torch.bool)
        out = self.attention(q, k, v, mask=~casual_mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, self.d_model)
        return self.wo(out)

