import torch
import torch.nn as nn
from ..utils.layers import Conv1DBlock
from ..utils.tensor_ops import transpose_for_conv1d, transpose_for_transformer

class TransitProposalNetwork(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 5, nms_dist: int = 64):
        super().__init__()
        self.conv = Conv1DBlock(in_channels, hidden_dim, kernel_size, padding=kernel_size//2)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.nms_dist = nms_dist

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = transpose_for_conv1d(x)
        features = self.conv(x_conv)
        features = transpose_for_transformer(features)
        scores = torch.sigmoid(self.score_proj(features)).squeeze(-1)
        
        if self.nms_dist > 0:
            import torch.nn.functional as F
            # Apply 1D Non-Maximum Suppression using MaxPool1d
            pad = self.nms_dist
            padded_scores = F.pad(scores.unsqueeze(1), (pad, pad), mode='constant', value=-1.0)
            local_max = F.max_pool1d(padded_scores, kernel_size=self.nms_dist * 2 + 1, stride=1)
            keep_mask = (scores.unsqueeze(1) == local_max).float()
            scores = scores * keep_mask.squeeze(1)
            
        return scores
