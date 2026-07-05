import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model_arch.architecture import ExoplanetModel
from model_arch.config import ModelConfig
from pipelines.utils.dataset_loader import get_dataloaders

def verify():
    dataset_root = os.path.join(os.path.dirname(__file__), 'dataset')
    
    print("Loading dataloader...")
    train_loader, val_loader, dataset, stats = get_dataloaders(
        root_dir=dataset_root, batch_size=2, num_workers=0
    )
    print("Stats:", stats)
    
    print("Fetching one batch...")
    batch = next(iter(train_loader))
    x, g_mask, l_mask, labels = batch
    print("x shape:", x.shape)
    print("g_mask shape:", g_mask.shape)
    print("l_mask shape:", l_mask.shape)
    print("labels:", labels)
    
    print("Building model...")
    config = ModelConfig()
    model = ExoplanetModel(config)
    
    print("Forward pass...")
    outputs = model(
        x_global=x,
        x_local=x,
        global_key_padding_mask=g_mask,
        local_key_padding_mask=l_mask
    )
    print("outputs shape:", outputs.shape)
    
    print("Backward pass...")
    loss = outputs.sum()
    loss.backward()
    print("Backward pass successful!")

if __name__ == "__main__":
    verify()
