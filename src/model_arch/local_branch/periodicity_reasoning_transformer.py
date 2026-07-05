import torch
import torch.nn as nn
from ..global_branch.transformer import TransformerEncoderBlock
from ..utils.positional_encoding import PositionalEncoding

class PeriodicityReasoningTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1, max_top_k: int = 100):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_top_k)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
