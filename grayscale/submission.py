import torch


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    """RGB to grayscale: Y = 0.2989 R + 0.5870 G + 0.1140 B"""
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device, dtype=data.dtype)
    return torch.sum(data * weights, dim=-1)
