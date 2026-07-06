import torch
import torch.nn as nn
from ..utils.layers import Conv1DBlock
from ..utils.tensor_ops import transpose_for_conv1d, transpose_for_transformer

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list, dropout: float = 0.1, downsample_factor: int = 16):
        super().__init__()
        self.branches = nn.ModuleList([
            Conv1DBlock(in_channels, out_channels, k, padding=k//2) for k in kernel_sizes
        ])
        self.proj = nn.Linear(out_channels * len(kernel_sizes), out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.downsample_factor = downsample_factor
        if self.downsample_factor > 1:
            self.pool = nn.MaxPool1d(kernel_size=downsample_factor, stride=downsample_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = transpose_for_conv1d(x)
        features = [branch(x) for branch in self.branches]
        out = torch.cat(features, dim=1)
        
        if self.downsample_factor > 1:
            out = self.pool(out)
            
        out = transpose_for_transformer(out)
        out = self.dropout(self.proj(out))
        return out
