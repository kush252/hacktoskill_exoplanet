import torch
import torch.nn as nn

class WindowExtractor(nn.Module):
    def __init__(self, top_k: int, window_size: int):
        super().__init__()
        self.top_k = top_k
        self.window_size = window_size

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=1)
        
        # Calculate start indices, clamping to avoid going out of bounds
        start_idx = torch.clamp(topk_indices - self.window_size // 2, min=0)
        end_idx = torch.clamp(start_idx + self.window_size, max=seq_len)
        start_idx = torch.clamp(end_idx - self.window_size, min=0)
        
        # Vectorized generation of indices
        offsets = torch.arange(self.window_size, device=x.device)
        # start_idx is [batch_size, top_k], offsets is [window_size]
        # window_indices will be [batch_size, top_k, window_size]
        window_indices = start_idx.unsqueeze(-1) + offsets
        
        # Gather the windows using advanced indexing
        b_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        windows = x[b_idx, window_indices] # shape: [batch_size, top_k, window_size, d_model]
        
        # Soft-Attention: multiply the windows by their corresponding TPN score
        # This provides a differentiable path so gradients can flow into the TPN
        windows = windows * topk_scores.unsqueeze(-1).unsqueeze(-1)
        
        return windows
