# Exoplanet Model Training Pipeline

This repository contains the model architecture, dataset loading utilities, and the training pipeline for the Exoplanet detection project. The code is cross-platform and will automatically utilize a GPU if available.

## Getting Started

Follow these steps to set up the environment, download the data, and start training.

### 1. Set up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Requirements
Install the required packages (PyTorch, NumPy, tqdm, etc.):
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Run the dataset installer script to download and extract the NPZ files. 
*(Note: Replace `dataset_installer.py` with the actual name of your installer script if it is different)*
```bash
python dataset_installer.py
```
Ensure that the downloaded dataset is placed in a folder named `dataset/` at the root of the project. The folder structure should look like this:
```
dataset/
    CONFIRMED/
    FALSE_POSITIVE/
    CANDIDATE/
```

### 4. Run the Training Pipeline
Once the dataset is ready, you can start the training process:
```bash
python src/pipelines/training_pipeline.py
```

## GPU Optimization (Maximizing Potential)

The pipeline already uses Automatic Mixed Precision (AMP) and gradient scaling to run efficiently on GPUs. If you have a dedicated GPU, you can further maximize its potential:

1. **Install PyTorch with CUDA Support:** Instead of the standard `requirements.txt`, install PyTorch manually with the correct CUDA index for your system, for example:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. **Increase `batch_size`:** In `src/pipelines/training_pipeline.py` (near the bottom), `batch_size` is set to `2` to be safe. Increase this (e.g., 8, 16, 32) until you hit a CUDA "Out of Memory" error to fully saturate your VRAM.
3. **Increase `num_workers`:** In `src/pipelines/utils/dataset_loader.py`, you can increase `num_workers` (default is 4) to 8 or 16 if your CPU can fetch data faster for the GPU.

## Training Outputs
During training, the pipeline will automatically create two folders in the project root:
- `checkpoints/`: Contains the saved models (`best_model.pt`, `checkpoint_epoch_X.pt`, `final_model.pt`).
- `metrics/`: Contains training history (`training_history.csv` and `training_history.json`).
