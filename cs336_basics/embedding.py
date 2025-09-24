
import numpy as np
import torch

class Embedding(torch.nn.Module):
    def __init__(self, 
                 vocab_size : int, 
                 d_model : int,
                 device : torch.device = None,
                 dtype : torch.dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.empty((vocab_size, d_model), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
