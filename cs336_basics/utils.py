import os
from typing import BinaryIO, IO
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


def lr_cosine_schedule(current_iter: int,
                       max_learning_rate: float,
                       min_learning_rate: float,
                       warmup_iters: int,
                       cosine_cycle_iters: int) -> float:
    if current_iter < warmup_iters:
        lr = max_learning_rate * current_iter / warmup_iters
        # lr = max(lr, min_learning_rate)
        return lr

    if current_iter > cosine_cycle_iters:
        return min_learning_rate

    cosine_decay = (1 + np.cos(np.pi *(current_iter - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2.0
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    return lr

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int,
                    path: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    if isinstance(path, (str, os.PathLike)):
        with open(path, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, path)

def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    path: str | os.PathLike | BinaryIO | IO[bytes])-> int:
    if isinstance(path, (str, os.PathLike)):
        with open(path, 'rb' ) as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
