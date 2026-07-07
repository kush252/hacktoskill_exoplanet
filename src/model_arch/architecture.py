import torch
import torch.nn as nn
from .config import ModelConfig
from .global_branch import MultiScaleCNN, GlobalTransformerEncoder, AttentionPooling as GlobalAttentionPooling
from .local_branch import TransitProposalNetwork, WindowExtractor, LocalTransformerEncoder, LocalAttentionPooling, PeriodicityReasoningTransformer
from .utils import initialize_weights

class ExoplanetModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.global_cnn = MultiScaleCNN(
            in_channels=config.global_cfg.in_channels,
            out_channels=config.global_cfg.d_model,
            kernel_sizes=config.global_cfg.cnn_kernel_sizes,
            dropout=config.global_cfg.dropout
        )
        self.global_encoder = GlobalTransformerEncoder(
            d_model=config.global_cfg.d_model,
            num_heads=config.global_cfg.num_heads,
            d_ff=config.global_cfg.d_ff,
            num_layers=config.global_cfg.num_layers,
            dropout=config.global_cfg.dropout,
            max_seq_len=config.global_cfg.max_seq_len
        )
        self.global_pool = GlobalAttentionPooling(config.global_cfg.d_model)

        self.tpn = TransitProposalNetwork(
            in_channels=config.local_cfg.in_channels,
            hidden_dim=config.local_cfg.tpn_hidden_dim,
            kernel_size=config.local_cfg.tpn_kernel_size
        )
        self.window_extractor = WindowExtractor(
            top_k=config.local_cfg.top_k,
            window_size=config.local_cfg.window_size
        )
        self.local_proj = nn.Linear(config.local_cfg.in_channels, config.local_cfg.d_model)
        self.local_encoder = LocalTransformerEncoder(
            d_model=config.local_cfg.d_model,
            num_heads=config.local_cfg.num_heads,
            d_ff=config.local_cfg.d_ff,
            num_layers=config.local_cfg.num_layers,
            dropout=config.local_cfg.dropout,
            max_window_size=config.local_cfg.window_size
        )
        self.local_pool = LocalAttentionPooling(config.local_cfg.d_model)
        self.prt = PeriodicityReasoningTransformer(
            d_model=config.local_cfg.d_model,
            num_heads=config.local_cfg.num_heads,
            d_ff=config.local_cfg.d_ff,
            num_layers=config.local_cfg.prt_num_layers,
            dropout=config.local_cfg.dropout,
            max_top_k=config.local_cfg.top_k
        )
        self.prt_pool = GlobalAttentionPooling(config.local_cfg.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(config.global_cfg.d_model + config.local_cfg.d_model, (config.global_cfg.d_model + config.local_cfg.d_model) // 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear((config.global_cfg.d_model + config.local_cfg.d_model) // 2, config.num_classes)
        )
        
        self.apply(initialize_weights)

    def forward(
        self, 
        x_global: torch.Tensor, 
        x_local: torch.Tensor = None,
        global_mask: torch.Tensor = None,
        global_key_padding_mask: torch.Tensor = None,
        local_mask: torch.Tensor = None,
        local_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if x_local is None:
            x_local = x_global
            
        g_feat = self.global_cnn(x_global)
        
        if global_key_padding_mask is not None and getattr(self.global_cnn, 'downsample_factor', 1) > 1:
            # Downsample the boolean mask to match the downsampled sequence length
            # Invert mask: 1.0 for valid (False), 0.0 for padding (True)
            float_mask = global_key_padding_mask.unsqueeze(1).float()
            inverted_mask = 1.0 - float_mask
            # Pool it the same way the CNN pools features
            pooled_inverted = self.global_cnn.pool(inverted_mask)
            # Revert mask: True if all elements in the window were padding (True)
            global_key_padding_mask = (1.0 - pooled_inverted).bool().squeeze(1)
            
        g_feat = self.global_encoder(g_feat, src_mask=global_mask, src_key_padding_mask=global_key_padding_mask)
        g_rep = self.global_pool(g_feat)

        tpn_scores = self.tpn(x_local)
        l_windows = self.window_extractor(x_local, tpn_scores)
        
        batch_size, top_k, window_size, in_channels = l_windows.size()
        l_windows = self.local_proj(l_windows.view(-1, in_channels)).view(batch_size, top_k, window_size, -1)
        
        # If a local key padding mask is provided, we would need to extract windows from it similarly to x_local,
        # but for simplicity we assume local_mask and local_key_padding_mask are already shaped for the windows if provided.
        # Often, local windows don't have padding since they are fixed size.
        l_feat = self.local_encoder(l_windows, src_mask=local_mask, src_key_padding_mask=local_key_padding_mask)
        l_feat = self.local_pool(l_feat)
        
        # We can also pass masks to PRT if necessary, but typically PRT reasons over top_k without padding
        # unless top_k varies per batch, which is not the case here.
        l_feat = self.prt(l_feat)
        l_rep = self.prt_pool(l_feat)

        fused = torch.cat([g_rep, l_rep], dim=-1)
        out = self.classifier(fused)
        
        return out
