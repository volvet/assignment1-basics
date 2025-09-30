import numpy as np
import torch

class ADAMW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 betas = (0.9, 0.999),
                 eps = 1e-8,
                 weight_decay = 0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                g = p.grad.data
                beta1, beta2 = group["betas"]
                m = m.mul_(beta1).add_(g, alpha=1-beta1)
                v = v.mul_(beta2).addcmul_(g, g, value=1-beta2)
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1
                denom = (v / bias_correction2).sqrt().add_(group["eps"])
                #p.data.addcdiv_(-step_size, m, denom)
                torch.addcdiv(p.data, m, denom, value=-step_size, out=p.data)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss