import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.attention(x)
        if mask is not None:
            # mask is True for padding tokens
            scores = scores.masked_fill(mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)
