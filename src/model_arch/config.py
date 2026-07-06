from dataclasses import dataclass, field
from typing import List

@dataclass
class GlobalBranchConfig:
    downsample_factor: int = 16
    in_channels: int = 1
    d_model: int = 64
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    num_heads: int = 4
    d_ff: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    max_seq_len: int = 10000

@dataclass
class LocalBranchConfig:
    in_channels: int = 1
    tpn_hidden_dim: int = 32
    tpn_kernel_size: int = 5
    top_k: int = 10
    window_size: int = 128
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 256
    num_layers: int = 2
    prt_num_layers: int = 2
    dropout: float = 0.1

@dataclass
class ModelConfig:
    global_cfg: GlobalBranchConfig = field(default_factory=GlobalBranchConfig)
    local_cfg: LocalBranchConfig = field(default_factory=LocalBranchConfig)
    num_classes: int = 3
    fusion_dropout: float = 0.1
