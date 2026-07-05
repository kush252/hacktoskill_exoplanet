import torch
import torch.nn as nn
from ..global_branch.transformer import TransformerEncoderBlock
from ..utils.positional_encoding import PositionalEncoding

class LocalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1, max_window_size: int = 500):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_window_size)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        batch_size, top_k, window_size, d_model = x.size()
        x = x.view(-1, window_size, d_model)
        x = self.pos_encoder(x)
        
        # If masks are provided, they might need reshaping if they are [batch_size, top_k, window_size]
        # But we assume they are passed correctly reshaped or broadcastable for the batched windows
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x.view(batch_size, top_k, window_size, d_model)
