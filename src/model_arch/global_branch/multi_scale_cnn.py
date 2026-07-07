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
        
        merged_channels = out_channels * len(kernel_sizes)
        
        self.downsample_factor = downsample_factor
        if self.downsample_factor > 1:
            # We keep self.pool so architecture.py can still easily downsample the boolean mask
            self.pool = nn.MaxPool1d(kernel_size=downsample_factor, stride=downsample_factor)
            
            # Instead of a single massive max pool for features, we use progressive convolutions
            layers = []
            current_factor = downsample_factor
            in_c = merged_channels
            
            while current_factor > 1:
                # Find a gentle factor to downsample
                if current_factor % 2 == 0:
                    pool_stride = 2
                elif current_factor % 3 == 0:
                    pool_stride = 3
                elif current_factor % 5 == 0:
                    pool_stride = 5
                else:
                    pool_stride = current_factor
                
                layers.append(Conv1DBlock(in_c, in_c, kernel_size=3, padding=1))
                layers.append(nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride))
                current_factor //= pool_stride
                
            self.progressive_features = nn.Sequential(*layers)
            
        self.proj = nn.Linear(merged_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = transpose_for_conv1d(x)
        features = [branch(x) for branch in self.branches]
        out = torch.cat(features, dim=1)
        
        if self.downsample_factor > 1:
            out = self.progressive_features(out)
            
        out = transpose_for_transformer(out)
        out = self.dropout(self.proj(out))
        return out
