import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional

class ExoplanetDataset(Dataset):
    """
    Production-quality PyTorch Dataset for Exoplanet NPZ files.
    """
    def __init__(self, root_dir: str, transform=None, max_seq_len: int = 80000):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        self.classes = sorted(os.listdir(root_dir))
        # Ignore metadata.csv if it's in the root
        self.classes = [c for c in self.classes if os.path.isdir(os.path.join(root_dir, c))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        self.samples = []
        self._scan_dataset()

    def _scan_dataset(self):
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            label = self.class_to_idx[cls_name]
            npz_files = glob.glob(os.path.join(cls_dir, "*.npz"))
            
            for file_path in npz_files:
                self.samples.append((file_path, label))
                
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        
        try:
            # Lazy loading
            data = np.load(file_path)
            
            # Verify required keys
            if 'flux' not in data.files:
                raise ValueError(f"Missing 'flux' key in {file_path}")
            
            flux = data['flux']
            
            # Verify no NaN or Inf
            if np.isnan(flux).any() or np.isinf(flux).any():
                raise ValueError(f"NaN or Inf values found in {file_path}")
            
            # Convert to float32
            flux = flux.astype(np.float32)
            
            # Shape into [seq_len, 1] as expected by in_channels=1
            flux = np.expand_dims(flux, axis=-1)
            
            # Truncate if exceeds max_seq_len
            if flux.shape[0] > self.max_seq_len:
                flux = flux[:self.max_seq_len]
                
            tensor_x = torch.from_numpy(flux)
            
            if self.transform:
                tensor_x = self.transform(tensor_x)
                
            return tensor_x, label
            
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")


def pad_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate_fn to pad sequences to the maximum length in the batch.
    It returns:
    - padded_x: [batch_size, max_seq_len_in_batch, 1]
    - global_padding_mask: [batch_size, max_seq_len_in_batch // 8]
    - local_padding_mask: [batch_size, max_seq_len_in_batch]
    - labels: [batch_size]
    """
    xs, labels = zip(*batch)
    
    # Calculate lengths
    lengths = [x.size(0) for x in xs]
    max_len = max(lengths)
    
    # No need to make max_len divisible by downsample_factor here; handled by the model.
    
    # Pad tensors to max_len
    padded_xs = []
    global_masks = []
    local_masks = []
    
    for x in xs:
        seq_len = x.size(0)
        pad_size = max_len - seq_len
        
        # Pad with zeros [seq_len, 1] -> [max_len, 1]
        if pad_size > 0:
            padded_x = torch.nn.functional.pad(x, (0, 0, 0, pad_size), value=0.0)
        else:
            padded_x = x
            
        padded_xs.append(padded_x)
        
        # Create masks (True means padding token, ignored in attention)
        local_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_size > 0:
            local_mask[seq_len:] = True
        local_masks.append(local_mask)
        
        # Global mask requires downsampling which is now handled automatically in the model.
        # We supply the full-resolution padding mask.
        global_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_size > 0:
            global_mask[seq_len:] = True
        global_masks.append(global_mask)
        
    padded_xs = torch.stack(padded_xs)
    global_masks = torch.stack(global_masks)
    local_masks = torch.stack(local_masks)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_xs, global_masks, local_masks, labels


def get_dataloaders(
    root_dir: str, 
    batch_size: int = 16, 
    val_split: float = 0.2, 
    num_workers: int = 4, 
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, ExoplanetDataset, Dict[str, int]]:
    """
    Create training and validation dataloaders.
    """
    dataset = ExoplanetDataset(root_dir=root_dir)
    
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
        pin_memory=True
    )
    
    stats = {
        'num_classes': len(dataset.classes),
        'train_size': train_size,
        'val_size': val_size,
        'total_size': dataset_size
    }
    
    return train_loader, val_loader, dataset, stats
