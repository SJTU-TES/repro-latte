import torch


def modulate(x, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)