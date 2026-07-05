import torch

def transpose_for_conv1d(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2)

def transpose_for_transformer(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2)
