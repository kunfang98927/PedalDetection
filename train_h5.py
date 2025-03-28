import os
import torch
import shutil
import argparse
from torch.utils.data import DataLoader
from src.model1 import PedalDetectionModelwithCNN
from src.model2 import PedalDetectionModelwithCNN1
from src.model3 import PedalDetectionModelContrastive
from src.model2_room_cond import PedalDetectionModelwithCNN_RoomCond
from src.dataset_h5 import PedalDataset, PedalRoomContrastiveDataset
from src.trainer2 import PedalTrainer2
from src.utils import get_label_bin_edges
import functools
import wandb


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
        default=["r1-pf0", "r0-pf1"],
        help="Datasets list (default: ['r1-pf0', 'r0-pf1'])",
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
        "--feature_dim", type=int, default=229, help="Feature dimension (default: 249)"
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
        "--room_ratio", type=float, default=0.0, help="Room ratio (default: 0.0)"
    )
    parser.add_argument(
        "--contrastive_ratio",
        type=float,
        default=0.0,
        help="Contrastive ratio (default: 0.1)",
    )
    parser.add_argument(
        "--room_contrastive_ratio",
        type=float,
        default=0.1,
        help="Room contrastive ratio (default: 0.1)",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="model2",
        help="Model version (default: model2)",
    )
    parser.add_argument(
        "--room_classes",
        type=int,
        default=4,
        help="Number of room classes (default: 4)",
    )
    parser.add_argument(
        "--on_off_threshold",
        type=float,
        default=64,
        help="Onset offset threshold (default: 64)",
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
    print(f"Room Ratio: {args.room_ratio}")
    print(f"Contrastive Ratio: {args.contrastive_ratio}")
    print(f"Room Contrastive Ratio: {args.room_contrastive_ratio}")
    print(f"Model Version: {args.model_version}")
    print(f"Room Classes: {args.room_classes}")
    print(f"On Off Threshold: {args.on_off_threshold}")

    checkpoint_path = args.checkpoint_path
    data_dir = args.data_dir
    datasets = args.datasets
    save_dir = args.save_dir
    batch_size = args.batch_size
    eval_epochs = args.eval_epochs
    eval_steps = args.eval_steps
    feature_dim = args.feature_dim
    max_frame = args.max_frame
    num_samples_per_clip = args.num_samples_per_clip
    num_classes = args.num_classes
    train_rand_sample = args.train_rand_sample
    global_pedal_ratio = args.global_pedal_ratio
    pedal_value_ratio = args.pedal_value_ratio
    pedal_onset_ratio = args.pedal_onset_ratio
    pedal_offset_ratio = args.pedal_offset_ratio
    room_ratio = args.room_ratio
    contrastive_ratio = args.contrastive_ratio
    room_contrastive_ratio = args.room_contrastive_ratio
    model_version = args.model_version
    room_classes = args.room_classes
    on_off_threshold = args.on_off_threshold

    # WandB
    wandb.init(
        project=f"pedal-{save_dir}", 
        entity="fangk3740-mcgill-university",
        config={
            "checkpoint_path": checkpoint_path,
            "data_dir": data_dir,
            "datasets": datasets,
            "save_dir": save_dir,
            "batch_size": batch_size,
            "eval_epochs": eval_epochs,
            "eval_steps": eval_steps,
            "feature_dim": feature_dim,
            "max_frame": max_frame,
            "num_samples_per_clip": num_samples_per_clip,
            "num_classes": num_classes,
            "train_rand_sample": train_rand_sample,
            "global_pedal_ratio": global_pedal_ratio,
            "pedal_value_ratio": pedal_value_ratio,
            "pedal_onset_ratio": pedal_onset_ratio,
            "pedal_offset_ratio": pedal_offset_ratio,
            "room_ratio": room_ratio,
            "contrastive_ratio": contrastive_ratio,
            "room_contrastive_ratio": room_contrastive_ratio,
            "model_version": model_version,
            "room_classes": room_classes,
            "on_off_threshold": on_off_threshold,
        }
    )

    # Label bin edges, train and val
    label_bin_edges = get_label_bin_edges(num_classes)
    val_label_bin_edges = get_label_bin_edges(2)

    # Copy this file to save_dir
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy("train_h5.py", os.path.join(save_dir, "train_h5"))

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
        f.write(f"room_ratio: {args.room_ratio}\n")
        f.write(f"contrastive_ratio: {args.contrastive_ratio}\n")
        f.write(f"room_contrastive_ratio: {args.room_contrastive_ratio}\n")
        f.write(f"model_version: {args.model_version}\n")
        f.write(f"room_classes: {args.room_classes}\n")
        f.write(f"on_off_threshold: {args.on_off_threshold}\n")

    # Dataset and DataLoader
    if args.room_contrastive_ratio > 0.0:
        print("[INFO] Using PedalRoomContrastiveDataset for triplet contrastive training")
        train_dataset = PedalRoomContrastiveDataset(
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
        )
    else:
        train_dataset = PedalDataset(
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
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Set device (and note multi-GPU availability)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Model
    model = None
    if model_version == "model1":
        model = PedalDetectionModelwithCNN(
            input_dim=feature_dim,
            hidden_dim=256,
            num_heads=8,
            ff_dim=1024,  # 256,
            num_layers=8,
            num_classes=num_classes,
        )
    elif model_version == "model2":
        model = PedalDetectionModelwithCNN1(
            input_dim=feature_dim,
            hidden_dim=256,
            num_heads=8,
            ff_dim=1024,  # 256,
            num_layers=8,
            num_classes=num_classes,
            predict_global_pedal=True if global_pedal_ratio > 0 else False,
            predict_pedal_onset=True if pedal_onset_ratio > 0 else False,
            predict_pedal_offset=True if pedal_offset_ratio > 0 else False,
            predict_room=True if room_ratio > 0 else False,
            room_classes=room_classes,
        )
    elif model_version == "model3":
        model = PedalDetectionModelContrastive(
            input_dim=feature_dim,
            hidden_dim=256,
            num_heads=8,
            ff_dim=1024,  # 256,
            num_layers=8,
            num_classes=num_classes,
            predict_global_pedal=True if global_pedal_ratio > 0 else False,
            predict_pedal_onset=True if pedal_onset_ratio > 0 else False,
            predict_pedal_offset=True if pedal_offset_ratio > 0 else False,
            predict_room=True if room_ratio > 0 else False,
        )
    elif model_version == "model_condition":
        model = PedalDetectionModelwithCNN_RoomCond(
            input_dim=feature_dim,
            hidden_dim=256,
            num_heads=8,
            ff_dim=1024,  # 256,
            num_layers=8,
            num_classes=num_classes,
            predict_global_pedal=True if global_pedal_ratio > 0 else False,
            predict_pedal_onset=True if pedal_onset_ratio > 0 else False,
            predict_pedal_offset=True if pedal_offset_ratio > 0 else False,
            predict_room=True if room_ratio > 0 else False,
            room_classes=room_classes,
        )
    print(model)
    # print trainable parameters number
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("The device is", device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        # Manually move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]  # Resume from the next epoch
        start_global_step = checkpoint["global_step"]
        start_epoch = start_epoch + 1 if start_global_step == -1 else start_epoch
        if start_global_step != -1:
            print(f"Resuming training from epoch {start_epoch}, global step {start_global_step}...")
        else:
            print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0
        start_global_step = -1

    # Manually step the scheduler based on the epoch number
    num_iterations = start_global_step if start_global_step != -1 else start_epoch * len(train_dataset)
    for e in range(num_iterations):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {e}: lr={optimizer.param_groups[0]['lr']}")

    # DataLoader
    if device.startswith("cuda"):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            pin_memory_device=device,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            pin_memory_device=device,
        )
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   

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
    trainer = PedalTrainer2(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logging_steps=5,
        eval_steps=eval_steps,
        eval_epochs=eval_epochs,
        save_total_limit=10,
        num_train_epochs=50,
        val_label_bin_edges=val_label_bin_edges,
        save_dir=save_dir,
        log_dir=log_dir,
    )

    # Train the model
    trainer.train(
        global_pedal_ratio=global_pedal_ratio,
        pedal_value_ratio=pedal_value_ratio,
        pedal_onset_ratio=pedal_onset_ratio,
        pedal_offset_ratio=pedal_offset_ratio,
        room_ratio=room_ratio,
        contrastive_ratio=contrastive_ratio,
        room_contrastive_ratio=room_contrastive_ratio,
        start_epoch=start_epoch,
        start_global_step=start_global_step,
    )

    # wandb_run.finish()


if __name__ == "__main__":
    main()
