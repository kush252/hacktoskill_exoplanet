import os
import time
import json
import csv
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_arch.architecture import ExoplanetModel
from model_arch.config import ModelConfig
from pipelines.utils.dataset_loader import get_dataloaders


class TrainingPipeline:
    def __init__(
        self,
        dataset_root: str,
        checkpoint_dir: str = 'checkpoints',
        metrics_dir: str = 'metrics',
        batch_size: int = 16,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        resume: bool = True
    ):
        self.dataset_root = dataset_root
        self.checkpoint_dir = checkpoint_dir
        self.metrics_dir = metrics_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.resume = resume
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            
        self.start_epoch = 0
        self.best_acc = 0.0
        self.history = []

    def print_summary(self, stats, model):
        print("="*50)
        print("🚀 TRAINING PIPELINE INITIALIZED")
        print("="*50)
        print(f"Device: {self.device}")
        print(f"Dataset Stats:")
        print(f"  - Classes: {stats['num_classes']}")
        print(f"  - Train Size: {stats['train_size']}")
        print(f"  - Val Size: {stats['val_size']}")
        print(f"  - Total: {stats['total_size']}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Stats:")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Trainable Params: {trainable_params:,}")
        print("="*50)

    def save_metrics(self):
        json_path = os.path.join(self.metrics_dir, 'training_history.json')
        csv_path = os.path.join(self.metrics_dir, 'training_history.csv')
        
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=4)
            
        if len(self.history) > 0:
            keys = self.history[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)

    def load_checkpoint(self, model, optimizer, scheduler, scaler):
        if not self.resume:
            return
            
        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if not checkpoints:
            print("No checkpoint found. Starting from scratch.")
            return
            
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_ckpt = checkpoints[-1]
        ckpt_path = os.path.join(self.checkpoint_dir, latest_ckpt)
        
        print(f"Resuming from checkpoint: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_accuracy']
            self.history = checkpoint['metrics']
            
            # Restore RNG state if saved
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
                
            print(f"Successfully resumed from Epoch {self.start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_accuracy': self.best_acc,
            'metrics': self.history,
            'torch_rng_state': torch.get_rng_state(),
            'config': model.config
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
            
        epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"--> Saved new best model (Acc: {self.best_acc:.4f})")

    def train_epoch(self, model, loader, optimizer, criterion, scaler):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training")
        for x, g_mask, l_mask, labels in pbar:
            x, g_mask, l_mask, labels = x.to(self.device), g_mask.to(self.device), l_mask.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=True):
                outputs = model(
                    x_global=x,
                    x_local=x,
                    global_key_padding_mask=g_mask,
                    local_key_padding_mask=None
                )
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{correct/total:.4f}"})
            
        return total_loss / total, correct / total

    @torch.no_grad()
    def validate_epoch(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Validation")
        for x, g_mask, l_mask, labels in pbar:
            x, g_mask, l_mask, labels = x.to(self.device), g_mask.to(self.device), l_mask.to(self.device), labels.to(self.device)
            
            with autocast(device_type=self.device.type, enabled=True):
                outputs = model(
                    x_global=x,
                    x_local=x,
                    global_key_padding_mask=g_mask,
                    local_key_padding_mask=None
                )
                loss = criterion(outputs, labels)
                
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{correct/total:.4f}"})
            
        return total_loss / total, correct / total

    def run(self):
        train_loader, val_loader, dataset, stats = get_dataloaders(
            root_dir=self.dataset_root,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        config = ModelConfig()
        model = ExoplanetModel(config).to(self.device)
        
        self.print_summary(stats, model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Use 'cuda' if available, else 'cpu' for GradScaler
        scaler = GradScaler(self.device.type, enabled=True)
        
        self.load_checkpoint(model, optimizer, scheduler, scaler)
        
        total_start_time = time.time()
        
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6e}")
            
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0],
                'epoch_time': epoch_time,
                'best_acc': self.best_acc
            })
            
            self.save_metrics()
            self.save_checkpoint(model, optimizer, scheduler, scaler, epoch, is_best)
            
        total_time = time.time() - total_start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours. Best Val Acc: {self.best_acc:.4f}")
        
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    pipeline = TrainingPipeline(
        dataset_root=os.path.join(os.path.dirname(__file__), '..', '..', 'dataset'),
        batch_size=2, # Small batch size for large sequence length
        num_epochs=10
    )
    pipeline.run()
