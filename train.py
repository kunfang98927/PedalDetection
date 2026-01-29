import os
import torch
import shutil
import argparse
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.model import PedalDetectionModel

from src.dataset import PedalDataset
from src.trainer import PedalTrainerBasic
from src.trainer_bce import PedalTrainerBCE
from src.utils import get_label_bin_edges
import functools

torch.autograd.set_detect_anomaly(True)


print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Pedal model training arguments")

    # Checkpoint path
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file (default: None)",
    )

    # Data directory
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/kunfang/pedal_data/data/",
        help="Directory where the H5 files are stored",
    )

    # Datasets
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["r0-pf1"],
        help="Datasets list (default: ['r0-pf1'])",
    )  # command line: --datasets r1-pf0 r0-pf1

    # Save directory
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ckpt",
        help="Directory to save the model checkpoints and logs (default: ckpt)",
    )

    # Feature dimensions and training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for training (default: 24)",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=1,
        help="Evaluate every N epochs (default: 1)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=-1,
        help="Evaluate every N steps (default: -1)",
    )
    parser.add_argument(
        "--feature_dim", type=int, default=249, help="Feature dimension (default: 249)"
    )
    parser.add_argument(
        "--max_frame", type=int, default=500, help="Maximum frame length (default: 500)"
    )
    parser.add_argument(
        "--num_samples_per_clip",
        type=int,
        default=10,
        help="Number of samples per clip (default: 10)",
    )
    parser.add_argument(
        "--num_classes", type=int, default=1, help="Number of classes (default: 1)"
    )
    parser.add_argument(
        "--train_rand_sample",
        action="store_true",
        help="Randomly sample train dataset (default: False)",
    )

    # Pedal ratios
    parser.add_argument(
        "--global_pedal_ratio",
        type=float,
        default=0.2,
        help="Global pedal ratio (default: 0.1)",
    )
    parser.add_argument(
        "--pedal_value_ratio",
        type=float,
        default=0.6,
        help="Pedal value ratio (default: 0.4)",
    )
    parser.add_argument(
        "--pedal_onset_ratio",
        type=float,
        default=0.1,
        help="Pedal onset ratio (default: 0.2)",
    )
    parser.add_argument(
        "--pedal_offset_ratio",
        type=float,
        default=0.1,
        help="Pedal offset ratio (default: 0.2)",
    )
    parser.add_argument(
        "--on_off_threshold",
        type=float,
        default=64,
        help="Onset offset threshold (default: 64)",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="mse",
        choices=["bce", "mse"],
        help="Loss function to use (default: mse)",
    )
    parser.add_argument(
        "--data_subset",
        type=float,
        default=1.0,
        help="Choose a subset of the dataset (default: 1.0, i.e., use the full dataset)",
    )
    parser.add_argument(
        "--actual_epoch",
        type=float,
        default=10,
        help="Actual epoch to be trained (default: 10)",
    )
    parser.add_argument(
        "--norm_feat",
        action='store_true',
        default=False,
        help="normalize the input features per track (default: False)",
    )
    parser.add_argument(
        "--use_midi",
        action='store_true',
        default=False,
        help="use midi as additional input (default: False)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="hidden dimension per modality (default: 128)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="onecycle",
        choices=["step", "onecycle"],
        help="Type of lr scheduler to use (default: onecycle)",
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, default=5, 
        help="Log every N steps"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for data loading (default: 16, and 0 for no parallel loading)",
    )
    parser.add_argument(
        "--use_dynamic",
        action='store_true',
        default=False,
        help="use dynamic information/pitch velocity (default: False)",
    )
    parser.add_argument(
        "--ex_midi",
        type=str,
        default="",
        help="file name of the external midi to substitute the existing midi (default: empty string, i.e., not using external midi)",
        )
    parser.add_argument(
        "--cnn_dim",
        type=int,
        default=256,
        help="dimension of cnn output (default: 256)",
    )
    parser.add_argument(
        "--mfcc_dim",
        type=int,
        default=128,
        help="dimension of mfcc output (default: 128)",
    )
    parser.add_argument(
        "--midi_dim",
        type=int,
        default=128,
        help="dimension of midi output (default: 128)",
    )

    return parser.parse_args()


def mix_dataset(
    features_real,
    labels_real,
    metadata_real,
    features_synth,
    labels_synth,
    metadata_synth,
):
    print("Mixing real and synthetic audio datasets...")
    print("Real metadata:", len(metadata_real), metadata_real[0])
    print("Synthetic metadata:", len(metadata_synth), metadata_synth[0])

    midi_files_real = [m[1] for m in metadata_real]
    midi_files_synth = [m[1] for m in metadata_synth]
    midi_files_real = set(midi_files_real)
    midi_files_synth = set(midi_files_synth)
    common_midi_files = midi_files_real.intersection(midi_files_synth)
    print("Common midi files:", len(common_midi_files))

    features = []
    labels = []
    metadata = []
    for i, (f, l, m) in enumerate(zip(features_real, labels_real, metadata_real)):
        if m[1] in common_midi_files:
            features.append(f)
            labels.append(l)
            metadata.append(m)
    for i, (f, l, m) in enumerate(zip(features_synth, labels_synth, metadata_synth)):
        if m[1] in common_midi_files:
            features.append(f)
            labels.append(l)
            metadata.append(m)
    print("Mixed dataset size:", len(features))

    return features, labels, metadata


def main():

    args = parse_args()

    # Print the parsed arguments (you can also use these in your training code)
    print(f"Checkpoint Path: {args.checkpoint_path}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Eval Epochs: {args.eval_epochs}")
    print(f"Eval Steps: {args.eval_steps}")
    print(f"Feature Dimension: {args.feature_dim}")
    print(f"Max Frame: {args.max_frame}")
    print(f"Num Samples per Clip: {args.num_samples_per_clip}")
    print(f"Num Classes: {args.num_classes}")
    print(f"Train Random Sample: {args.train_rand_sample}")
    print(f"Global Pedal Ratio: {args.global_pedal_ratio}")
    print(f"Pedal Value Ratio: {args.pedal_value_ratio}")
    print(f"Pedal Onset Ratio: {args.pedal_onset_ratio}")
    print(f"Pedal Offset Ratio: {args.pedal_offset_ratio}")
    print(f"On Off Threshold: {args.on_off_threshold}")
    print(f"Loss Function: {args.loss_function}")

    checkpoint_path = args.checkpoint_path
    data_dir = args.data_dir
    datasets = args.datasets
    save_dir = args.save_dir
    batch_size = args.batch_size
    eval_epochs = args.eval_epochs
    eval_steps = args.eval_steps
    feature_dim = args.feature_dim
    max_frame = args.max_frame
    num_classes = args.num_classes
    global_pedal_ratio = args.global_pedal_ratio
    pedal_value_ratio = args.pedal_value_ratio
    pedal_onset_ratio = args.pedal_onset_ratio
    pedal_offset_ratio = args.pedal_offset_ratio
    on_off_threshold = args.on_off_threshold
    loss_function = args.loss_function
    subset_ratio = args.data_subset
    actual_epoch = args.actual_epoch
    if_normalize_features = args.norm_feat
    use_midi = args.use_midi
    hidden_dim = args.hidden_dim
    lr_scheduler = args.lr_scheduler
    logging_steps = args.logging_steps
    num_workers = args.num_workers
    use_dynamic = args.use_dynamic
    ex_midi = args.ex_midi
    ex_pedal = args.ex_pedal
    cnn_dim = args.cnn_dim
    mfcc_dim = args.mfcc_dim
    midi_dim = args.midi_dim        

    # Label bin edges, train and val
    label_bin_edges = get_label_bin_edges(num_classes)
    val_label_bin_edges = get_label_bin_edges(2)

    # Copy this file to save_dir
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy("train.py", os.path.join(save_dir, "train"))

    # write the arguments to a yaml file
    with open(f"{save_dir}/config.yaml", "w") as f:
        f.write(f"checkpoint_path: {args.checkpoint_path}\n")
        f.write(f"data_dir: {args.data_dir}\n")
        f.write(f"datasets: {args.datasets}\n")
        f.write(f"save_dir: {args.save_dir}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"eval_epochs: {args.eval_epochs}\n")
        f.write(f"eval_steps: {args.eval_steps}\n")
        f.write(f"feature_dim: {args.feature_dim}\n")
        f.write(f"max_frame: {args.max_frame}\n")
        f.write(f"num_samples_per_clip: {args.num_samples_per_clip}\n")
        f.write(f"num_classes: {args.num_classes}\n")
        f.write(f"train_rand_sample: {args.train_rand_sample}\n")
        f.write(f"global_pedal_ratio: {args.global_pedal_ratio}\n")
        f.write(f"pedal_value_ratio: {args.pedal_value_ratio}\n")
        f.write(f"pedal_onset_ratio: {args.pedal_onset_ratio}\n")
        f.write(f"pedal_offset_ratio: {args.pedal_offset_ratio}\n")
        f.write(f"on_off_threshold: {args.on_off_threshold}\n")
        f.write(f"loss_function: {args.loss_function}\n")

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        data_list_path="../pedal-icassp/PedalDetection/sample_data/train.json",
        data_list_path="sample_data/train.json",
        data_dir=args.data_dir,
        num_samples_per_clip=args.num_samples_per_clip,
        max_frame=args.max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.70,
        split="train",
        datasets=args.datasets,
        randomly_sample=args.train_rand_sample,
        feature_dim=feature_dim,
        on_off_threshold=on_off_threshold,
        normalize_features=if_normalize_features,
        midi=use_midi,
        dynamic=use_dynamic,
        external_midi=ex_midi,
        pred_pedal=ex_pedal,
    )
    val_dataset = PedalDataset(
        data_list_path="sample_data/val.json",
        data_dir=data_dir,
        num_samples_per_clip=5,  # num_samples_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.0,
        split="validation",
        datasets=[df for df in datasets if "pf0" not in df],  # not evaluate pf=0
        randomly_sample=False,
        feature_dim=feature_dim,
        on_off_threshold=on_off_threshold,
        normalize_features=if_normalize_features,
        midi=use_midi,
        dynamic=use_dynamic,
        external_midi=ex_midi
    )

    if subset_ratio < 1.0:
        print(f"Using {subset_ratio * 100:.1f}% of the training dataset")
        torch.manual_seed(106)  # Add this line for reproducibility
        num_train_samples = int(len(train_dataset) * subset_ratio)
        train_indices = torch.randperm(len(train_dataset))[:num_train_samples]
        # train_indices = get_stratified_indices(train_dataset, subset_ratio)
        train_dataset = Subset(train_dataset, train_indices)
        
        print(f"Using {subset_ratio * 100:.1f}% of the validation dataset")
        torch.manual_seed(106)  # Add this line for reproducibility
        num_val_samples = int(len(val_dataset) * subset_ratio)
        val_indices = torch.randperm(len(val_dataset))[:num_val_samples]
        # val_indices = get_stratified_indices(val_dataset, subset_ratio)
        val_dataset = Subset(val_dataset, val_indices)


    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Set device (and note multi-GPU availability)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Model
    model = PedalDetectionModel(
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=8,
        predict_global_pedal=True if global_pedal_ratio > 0 else False,
        predict_pedal_onset=True if pedal_onset_ratio > 0 else False,
        predict_pedal_offset=True if pedal_offset_ratio > 0 else False,
        use_midi=use_midi,
        cnn_dim=cnn_dim,
        mfcc_dim=mfcc_dim,
        midi_dim=midi_dim,
    )
    
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # DataLoader
    if device.startswith("cuda"):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=device,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=device,
        )
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   

    # Optimizer and Scheduler
    # Original
    if lr_scheduler == "step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    else:
    # BERT-specific OneCycle LR
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Calculate total steps
        total_steps = int(len(train_dataloader) * actual_epoch)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,                    
            total_steps=total_steps,
            pct_start=0.1,                 
            anneal_strategy='cos',
            div_factor=25.0,  # Start at max_lr/25 = 2e-5
            final_div_factor=100  
        )

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("The device is", device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_global_step = checkpoint.get("global_step", -1)
        # global_step is defined as:
        # The number of batches already processed, 
        # i.e., the index of the last batch that was completed
        batches_per_epoch = len(train_dataloader)
        
        saved_epoch = checkpoint.get("epoch", 0)
        if start_global_step != -1:
            start_epoch = start_global_step // batches_per_epoch
            step_in_epoch = start_global_step % batches_per_epoch
            # sanity check
            print(f"Checkpoint epoch: {saved_epoch+1}, start global step: {start_global_step}, batches per epoch: {batches_per_epoch}")
            print(f"Resuming training from epoch {start_epoch+1}, batch {step_in_epoch+1} (global step {start_global_step+1})...")
        else:
            # If no global step saved, fallback to checkpoint epoch logic
            start_epoch = saved_epoch
            step_in_epoch = 0
            print(f"Checkpoint epoch: {saved_epoch+1}, start global step: {start_global_step}, batches per epoch: {batches_per_epoch}")
            print(f"Resuming training from epoch {start_epoch+1}, batch {step_in_epoch+1} (no global step found)...")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0
        step_in_epoch = 0
        start_global_step = -1




    # Multi-GPU support using DataParallel
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model).to(device)
    elif device.startswith("cuda"):
        print("Using 1 GPU.")
        model = model.to(device)
    else:
        print("Using CPU.")
        model = model.to(device)

    # Trainer
    trainer_params = {
        "model": model,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "eval_epochs": eval_epochs,
        "save_total_limit": 10,
        "num_train_epochs": 50,
        "val_label_bin_edges": val_label_bin_edges,
        "save_dir": save_dir,
        "log_dir": log_dir
    }
    trainer = None
    if loss_function == "bce":
        trainer = PedalTrainerBCE(**trainer_params)
    elif loss_function == "mse":
        trainer_params["use_midi"] = use_midi
        trainer = PedalTrainerBasic(**trainer_params)

    # Train the model
    trainer.train(
        global_pedal_ratio=global_pedal_ratio,
        pedal_value_ratio=pedal_value_ratio,
        pedal_onset_ratio=pedal_onset_ratio,
        pedal_offset_ratio=pedal_offset_ratio,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
        step_in_epoch=step_in_epoch,
    )

def get_stratified_indices(dataset, subset_ratio, seed=42, pedal_key='pedal_target'):
    """Get stratified sample indices maintaining pedal on/off balance"""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Quick analysis on a sample to estimate balance (faster for large datasets)
    analysis_size = min(500, len(dataset))  
    analysis_indices = torch.randperm(len(dataset))[:analysis_size]
    
    pedal_on_indices = []
    pedal_off_indices = []
    
    print(f"  Analyzing pedal distribution from {analysis_size} samples...")
    
    for i in analysis_indices:
        try:
            sample = dataset[i.item()]
            pedal_values = sample[pedal_key]
            avg_pedal = torch.mean(pedal_values.float()).item()
            
            if avg_pedal > 0.5:  # Mostly pedal on
                pedal_on_indices.append(i.item())
            else:  # Mostly pedal off  
                pedal_off_indices.append(i.item())
                
        except Exception:
            # If error, just add to pedal_off group
            pedal_off_indices.append(i.item())
    
    # Calculate target sample sizes
    total_samples = int(len(dataset) * subset_ratio)
    pedal_on_ratio = len(pedal_on_indices) / analysis_size
    
    pedal_on_target = int(total_samples * pedal_on_ratio)
    pedal_off_target = total_samples - pedal_on_target
    
    print(f"  Target: {pedal_on_target} pedal-on, {pedal_off_target} pedal-off samples")
    
    # Now sample from full dataset maintaining the ratio
    # Get all indices and shuffle
    all_indices = torch.randperm(len(dataset))
    
    selected_pedal_on = []
    selected_pedal_off = []
    
    # Go through shuffled indices and categorize until we have enough
    for i in all_indices:
        if len(selected_pedal_on) >= pedal_on_target and len(selected_pedal_off) >= pedal_off_target:
            break
            
        try:
            sample = dataset[i.item()]
            pedal_values = sample[pedal_key]
            avg_pedal = torch.mean(pedal_values.float()).item()
            
            if avg_pedal > 0.5 and len(selected_pedal_on) < pedal_on_target:
                selected_pedal_on.append(i.item())
            elif avg_pedal <= 0.5 and len(selected_pedal_off) < pedal_off_target:
                selected_pedal_off.append(i.item())
                
        except Exception:
            if len(selected_pedal_off) < pedal_off_target:
                selected_pedal_off.append(i.item())
    
    # Combine and shuffle final selection
    final_indices = selected_pedal_on + selected_pedal_off
    np.random.shuffle(final_indices)
    
    print(f"  Selected: {len(selected_pedal_on)} pedal-on, {len(selected_pedal_off)} pedal-off")
    
    return final_indices

if __name__ == "__main__":
    main()
