import torch
import torch.nn as nn
from ..utils.positional_encoding import PositionalEncoding
from ..utils.layers import ResidualBlock, FeedForward

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        
        norm_x2 = self.norm2(x)
        x = x + self.dropout(self.ff(norm_x2))
        return x

class GlobalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
