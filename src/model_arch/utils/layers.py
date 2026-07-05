import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, activation: nn.Module = nn.ReLU(), normalization: nn.Module = None):
        super().__init__()
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalization is not None:
            layers.append(normalization)
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
