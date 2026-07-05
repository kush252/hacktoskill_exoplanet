import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'leaky_relu': nn.LeakyReLU()
    }
    return activations.get(name.lower(), nn.ReLU())
