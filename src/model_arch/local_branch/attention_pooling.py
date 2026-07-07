import torch
import torch.nn as nn
from ..global_branch.attention_pooling import AttentionPooling as GlobalAttentionPooling

class LocalAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pooler = GlobalAttentionPooling(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, top_k, window_size, d_model = x.size()
        x = x.view(-1, window_size, d_model)
        
        if mask is not None:
            # mask shape is [batch_size, top_k, window_size]
            mask = mask.view(-1, window_size)
            
        pooled = self.pooler(x, mask=mask)
        return pooled.view(batch_size, top_k, d_model)
