import os
import torch
import shutil
from torch.utils.data import DataLoader
from src.model1 import PedalDetectionModelwithCNN
from src.dataset_h5 import PedalDataset
from src.trainer2 import PedalTrainer2
from src.utils import get_label_bin_edges
import functools

print = functools.partial(print, flush=True)


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

    checkpoint_path = None
    # checkpoint_path = "ckpt_0311_10per-clip-500frm_bs16-8h-2xdata-val1-6loss/model_epoch_7_step_7950_val_loss_0.1267_f1_0.7976_mae_0.2177.pt"

    # Feature dimension
    batch_size = 16
    feature_dim = 249  # 128 (spectrogram) + 13 (mfcc)
    max_frame = 500
    num_samples_per_clip = 10
    num_classes = 1

    global_pedal_ratio = 0.1
    pedal_value_ratio = 0.4
    pedal_onset_ratio = 0.2
    pedal_offset_ratio = 0.2
    room_ratio = 0.0
    contrastive_ratio = 0.1

    data_filter = [
        "r1-pf0",
        # "r1-pf1",
        # "r2-pf1",
        # "r3-pf1",
        "r0-pf1",
    ]

    label_bin_edges = get_label_bin_edges(num_classes)
    val_label_bin_edges = get_label_bin_edges(2)

    # Checkpoint save path
    save_dir = f"ckpt_0312_{num_samples_per_clip}per-clip-{max_frame}frm_bs{batch_size}-8h-2xdata-5loss"

    # Copy this file to save_dir
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("train_h5.py", os.path.join(save_dir, "train_h5"))

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        data_path="sample_data/train.json",
        num_samples_per_clip=num_samples_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
        split="train",
        data_filter=data_filter,
    )
    val_dataset = PedalDataset(
        data_path="sample_data/val.json",
        num_samples_per_clip=None,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.0,
        split="validation",
        data_filter=[df for df in data_filter if "pf0" not in df],  # not evaluate pf=0
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Set device (and note multi-GPU availability)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Model
    model = PedalDetectionModelwithCNN(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=8,
        ff_dim=1024,  # 256,
        num_layers=8,
        num_classes=num_classes,
    )
    print(model)

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

        start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    # Manually step the scheduler based on the epoch number
    for e in range(start_epoch):
        optimizer.step()
        scheduler.step()
        print(f"Epoch {e}: lr={optimizer.param_groups[0]['lr']}")

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

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
        logging_steps=10,
        eval_steps=200,
        eval_epochs=-1,
        save_total_limit=10,
        num_train_epochs=50,
        val_label_bin_edges=val_label_bin_edges,
        save_dir=save_dir,
    )

    # Train the model
    trainer.train(
        global_pedal_ratio=global_pedal_ratio,
        pedal_value_ratio=pedal_value_ratio,
        pedal_onset_ratio=pedal_onset_ratio,
        pedal_offset_ratio=pedal_offset_ratio,
        room_ratio=room_ratio,
        contrastive_ratio=contrastive_ratio,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
