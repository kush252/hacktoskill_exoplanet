import torch
import torch.nn as nn
from ..global_branch.attention_pooling import AttentionPooling as GlobalAttentionPooling

class LocalAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pooler = GlobalAttentionPooling(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, top_k, window_size, d_model = x.size()
        x = x.view(-1, window_size, d_model)
        pooled = self.pooler(x)
        return pooled.view(batch_size, top_k, d_model)
