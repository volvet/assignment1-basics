import numpy as np
import torch

def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between input logits and target labels.

    Args:
        input (torch.Tensor): A tensor of shape (batch_size, num_classes) representing the input logits.
        target (torch.Tensor): A tensor of shape (batch_size,) representing the target class indices.

    Returns:
        torch.Tensor: A scalar tensor representing the average cross-entropy loss over the batch.
    """
    log_probs = torch.nn.functional.log_softmax(input, dim=-1)
    loss = torch.nn.functional.nll_loss(log_probs, target, reduction='mean')
    return loss

def clip_gradients(parameters : torch.nn.Parameter, max_l2_norm : float) -> None:
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(torch.sum(p.grad.data.pow(2)) for p in parameters_with_grad))
    clip_coef = max_l2_norm / (total_norm + 1e-9)
    if clip_coef < 1:
        for p in parameters_with_grad:
            p.grad.data.mul_(clip_coef)