import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        return torch.sum(x * weights, dim=1)
