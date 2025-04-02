## 🎹 High-Resolution Sustain Pedal Depth Estimation

This repository contains code for training a Transformer-based model to detect piano sustain pedal usage from audio.

### 📂 Project Structure

```
.
├── train_basic.py                # Main training script
├── inference_basic.py            # Main inference & metrics calculation script
├── calculate_metric.py           # Helper functions related to evaluation metrics
├── requirements.txt              # Python dependencies for the project
├── sample_data/                  # JSON file lists for training, validation, and test data
├── src/
│   ├── model_basic.py            # Transformer-based model for pedal detection
│   ├── dataset_basic.py          # PyTorch dataset class for pedal data
│   ├── trainer_basic.py          # Trainer with MSE loss for pedal depth estimation
│   ├── trainer_bce.py            # Trainer with BCE loss for pedal depth estimation
│   ├── utils.py                  # Utility functions
│   ├── cnn_block.py              # Convolutional blocks
│   ├── transformer.py            # Transformer
│   ├── dirs.py                   # Directory and path utilities
```

### 🚀 Getting Started

#### 1. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### 2. **Prepare data**
Place your `.h5` data files in a directory (e.g., `/path/to/data/`) and update paths in `sample_data/train.json`, `sample_data/val.json` and `sample_data/test.json`.

#### 3. **Run training**
```bash
python train_basic.py \
  --data_dir /path/to/data \
  --datasets r0-pf1 \
  --save_dir results \
  --loss_function mse
```

Some useful training options:

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint_path` | Resume training from checkpoint | `None` |
| `--data_dir` | Path to H5 feature data | `/path/to/data/` |
| `--datasets` | Dataset names to include | `["r0-pf1"]` |
| `--save_dir` | Output directory for logs and checkpoints | `results` |
| `--loss_function` | Loss type (`mse` or `bce`) | `mse` |
| `--batch_size` | Batch size | `24` |
| `--train_rand_sample` | Use random sampling | `False` |