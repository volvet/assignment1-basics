import numpy as np
import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, 
                 d_model : int, 
                 eps : float = 1e-5,
                 device : torch.device = None,
                 dtype : torch.dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def rms(self, x) -> torch.Tensor:
        return (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        rms_x = self.rms(x)
        return x / rms_x * self.weight
