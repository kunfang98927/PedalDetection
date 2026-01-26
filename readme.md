## 🎹 Evaluating High-Resolution Piano Sustain Pedal Depth Estimation

This repository accompanies the ICASSP 2026 paper on **musically informed evaluation** for sustain pedal depth estimation. It includes training/inference code for Transformer-based models and an evaluation suite that extends frame-level metrics with **action-level** and **gesture-level** analyses.

**Paper status:** Accepted to ICASSP 2026.

### 📂 Project Structure

```
.
├── train.py                      # Main training script
├── inference.py                  # Inference & metrics calculation
├── calculate_metric.py           # Helper functions related to evaluation metrics
├── requirements.txt              # Python dependencies for the project
├── sample_data/                  # JSON file lists for training, validation, and test data
├── src/
│   ├── model.py                  # Transformer-based model for pedal detection
│   ├── dataset.py                # Dataset class for pedal data
│   ├── trainer.py                # Trainer with MSE loss for pedal depth estimation
│   ├── trainer_bce.py            # Trainer with BCE loss for pedal depth estimation
│   ├── utils.py                  # Utility functions
│   ├── multi_input_layer.py      # Multi-input fusion layers
│   ├── transformer.py            # Transformer
│   ├── dirs.py                   # Directory and path utilities
├── eval/
│   ├── action_analysis.py        # Action-level segmentation + metrics
│   ├── gesture.py                # Gesture extraction utilities
│   ├── gesture_analysis_5pt.py   # 5-point gesture metric
│   ├── gesture_analysis_fft.py   # Fourier-based gesture metric
├── plot/
│   ├── plot_dist.py              # Distribution plots for actions/gestures
```

### ✨ Highlights (ICASSP 2026)

- **Three-level evaluation:** frame, **action**, and **gesture** levels.
- **Action-level:** segment pedal curves into press/hold/release and evaluate alignment.
- **Gesture-level:** compare complete press–release cycles and their contour similarity.
- **Model variants:** audio (binary), audio, and audio+midi under a unified Transformer architecture.

### 🚀 Getting Started

#### 1. **Install dependencies**

```bash
pip install -r requirements.txt
```

#### 2. **Prepare data**

Place your `.h5` data files in a directory (e.g., `/path/to/data/`) and update the paths in `sample_data/train.json`, `sample_data/val.json`, and `sample_data/test.json`.

### 🧪 Reproducing ICASSP 2026 Metrics

To reproduce the paper’s evaluation, use **three model variants**:

- **audio (binary)**
- **audio**
- **audio+midi**

**Checkpoints (Hugging Face):** https://huggingface.co/KunFang/PedalDetection/tree/main

#### Inference

After downloading checkpoints, run [inference.py](inference.py) to obtain per-frame pedal depth predictions (saved as .npy) and frame-level metrics (saved as .txt).

##### Inference: audio (binary)

```bash
python inference.py --data_dir /path/to/data --dataset r0-pf1 \
  --checkpoint_path /path/to/checkpoint.pt --norm_feat \
  --hidden_dim 384 --cnn_dim 256 --mfcc_dim 128 --loss_function bce
```

##### Inference: audio

```bash
python inference.py --data_dir /path/to/data --dataset r0-pf1 \
  --checkpoint_path /path/to/checkpoint.pt --norm_feat \
  --hidden_dim 384 --cnn_dim 256 --mfcc_dim 128 --loss_function mse
```

##### Inference: audio+midi

```bash
python inference.py --data_dir /path/to/data --dataset r0-pf1 \
  --checkpoint_path /path/to/checkpoint.pt --ex_midi /path/to/midi.h5
  --norm_feat \
  --hidden_dim 384 --cnn_dim 256 --mfcc_dim 128 --midi_dim 96 --loss_function mse \
  --use_midi --use_dynamic
```

#### Action/Gesture Evaluation

To compute the action-level and gesture-level metrics proposed in the paper, run the scripts in [eval](eval).

Finally, generate action/gesture distribution figures using:

```bash
python plot/plot_dist.py /path/to/results
```

#### Training From Scratch

If you want to train the three model variants yourself, run the commands below.

##### Training: audio (binary)

```bash
python train.py --batch_size 32 --eval_steps 200 --feature_dim 249 \
  --max_frame 500 --data_dir /path/to/data --datasets r0-pf1 --save_dir /path/to/save_dir \
  --logging_steps 20 --loss_function bce \
  --norm_feat \
  --hidden_dim 384 \
  --cnn_dim 256 \
  --mfcc_dim 128
```

##### Training: audio

```bash
python train.py --batch_size 32 --eval_steps 200 --feature_dim 249 \
  --max_frame 500 --data_dir /path/to/data --datasets r0-pf1 --save_dir /path/to/save_dir \
  --logging_steps 20 --loss_function mse \
  --norm_feat \
  --hidden_dim 384 \
  --cnn_dim 256 \
  --mfcc_dim 128
```

##### Training: audio+midi

```bash
python train_extra.py --batch_size 32 --eval_steps 200 --feature_dim 249 \
  --max_frame 500 --data_dir /path/to/data --ex_midi /path/to/midi.npy \
  --datasets r0-pf1 --save_dir /path/to/save_dir \
  --logging_steps 20 --loss_function mse \
  --norm_feat \
  --hidden_dim 384 \
  --use_midi \
  --use_dynamic \
  --cnn_dim 256 \
  --mfcc_dim 128 \
  --midi_dim 96
```

### 📚 Cite the paper

```bibtex
@inproceedings{Zhang2026pedal,
  title={Evaluating High-Resolution Piano Sustain Pedal Depth Estimation with Musically Informed Metrics},
  author={Hanwen Zhang and Kun Fang and Ziyu Wang and Ichiro Fujinaga},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```
