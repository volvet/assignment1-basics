import numpy as np
import torch

class Linear(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 bias : bool = True,
                 device : torch.device = None,
                 dtype : torch.dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)

