import numpy as np
import torch

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, 
                 d_model : int, 
                 d_ff : int,
                 device : torch.device = None,
                 dtype : torch.dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.linear1 = torch.nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.linear2 = torch.nn.Linear(d_ff, d_model, bias=False, **factory_kwargs)
        self.linear3 = torch.nn.Linear(d_model, d_ff, bias=False, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def silu(self, x : torch.Tensor) -> torch.Tensor:
        # return x / (1 - torch.exp(-x))
        return x * torch.sigmoid(x)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))