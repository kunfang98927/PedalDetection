import os
import torch
import shutil
import numpy as np
from src.model1 import PedalDetectionModelwithCNN
from src.dataset import PedalDataset
from src.trainer1 import PedalTrainer1
from src.utils import (
    load_data,
    split_data,
    get_label_bin_edges,
    load_data_real_audio,
    split_data_real_audio,
)
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
    # checkpoint_path = "ckpt_0303_10per-clip-500frm_bs32_onoff_sbatch/model_epoch_460_val_loss_0.1984_f1_0.7202_mae_0.1631.pt"

    data_version_synth = "_kong1012room1_synth_20250217"  # "_kong504room3synth0220"  # "_4096_full_room1" # "_4096_NormPerFeat" # "_real_audio_4096"
    data_version_real = "_kong508room1real20250217"  # "_real_audio_4096"

    # Feature dimension
    batch_size = 32
    feature_dim = 249  # 128 (spectrogram) + 13 (mfcc)
    max_frame = 500
    num_samples_per_clip = 10
    num_classes = 1

    low_res_pedal_ratio = 0.1
    pedal_value_ratio = 0.4
    pedal_onset_ratio = 0.2
    pedal_offset_ratio = 0.2
    room_ratio = 0.0
    contrastive_ratio = 0.1

    pedal_factor = [0.0]  # pedal factor for synthetic data
    room_acoustics = [1.0]

    label_bin_edges = get_label_bin_edges(num_classes)
    val_label_bin_edges = get_label_bin_edges(2)

    # Checkpoint save path
    label_bin_edge_str = str(label_bin_edges[1]) + "-" + str(label_bin_edges[-2])
    factor_str = "&".join([str(f) for f in pedal_factor])
    save_dir = f"ckpt_0306_{num_samples_per_clip}per-clip-{max_frame}frm_bs{batch_size}_onoff-bce_sbatch"

    # Copy this file to save_dir
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("train1.py", os.path.join(save_dir, "train1"))

    # Data path
    data_path_synth = f"data/processed_data{data_version_synth}.npz"
    data_path_real = f"data/processed_data{data_version_real}.npz"

    # Load mix data
    features_real, labels_real, metadata_real = load_data_real_audio(
        data_path_real, label_bin_edges
    )
    train_features_real, train_labels_real, train_metadata_real = split_data_real_audio(
        features_real, labels_real, metadata_real, split="train"
    )
    val_features, val_labels, val_metadata = split_data_real_audio(
        features_real, labels_real, metadata_real, split="val"
    )  # only use real data as validation set
    features_synth, labels_synth, metadata_synth = load_data(
        data_path_synth, label_bin_edges, pedal_factor, room_acoustics
    )
    train_features_synth, train_labels_synth, train_metadata_synth = (
        split_data_real_audio(
            features_synth, labels_synth, metadata_synth, split="train"
        )
    )
    # features_synth_1, labels_synth_1, metadata_synth_1 = load_data(
    #     data_path_synth,
    #     label_bin_edges,
    #     pedal_factor=[1.0],
    #     room_acoustics=room_acoustics,
    # )
    # val_features_synth, val_labels_synth, val_metadata_synth = split_data_real_audio(
    #     features_synth_1, labels_synth_1, metadata_synth_1, split="val"
    # )
    print("Real audio train dataset size:", len(train_features_real))
    print("Synthetic audio train dataset size:", len(train_features_synth))
    # train_features, train_labels, train_metadata = (
    #     train_features_synth,
    #     train_labels_synth,
    #     train_metadata_synth,
    # )
    # val_features, val_labels, val_metadata = (
    #     val_features_synth,
    #     val_labels_synth,
    #     val_metadata_synth,
    # )
    # Mix data
    train_features, train_labels, train_metadata = mix_dataset(
        train_features_real,
        train_labels_real,
        train_metadata_real,
        train_features_synth,
        train_labels_synth,
        train_metadata_synth,
    )
    print("Mix Train dataset size:", len(train_features))

    # # Load data
    # if "real" in data_version:
    #     features, labels, metadata = load_data_real_audio(data_path, label_bin_edges)
    #     train_features, train_labels, train_metadata = split_data_real_audio(
    #         features, labels, metadata, split="train"
    #     )
    #     val_features, val_labels, val_metadata = split_data_real_audio(
    #         features, labels, metadata, split="val"
    #     )
    # elif "full" in data_version:
    #     features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
    #     train_features, train_labels, train_metadata = split_data_real_audio(
    #         features, labels, metadata, split="train"
    #     )
    #     val_features, val_labels, val_metadata = split_data_real_audio(
    #         features, labels, metadata, split="val"
    #     )
    # else:
    #     features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
    #     (
    #         train_features,
    #         val_features,
    #         _,
    #         train_labels,
    #         val_labels,
    #         _,
    #         train_metadata,
    #         val_metadata,
    #         _,
    #     ) = split_data(
    #         features, labels, metadata, val_size=0.15, test_size=0.15, random_state=100
    #     )

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        features=train_features,
        labels=train_labels,
        metadata=train_metadata,
        num_samples_per_clip=num_samples_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
        split="train",
    )
    val_dataset = PedalDataset(
        features=val_features,
        labels=val_labels,
        metadata=val_metadata,
        num_samples_per_clip=num_samples_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.05,
        split="validation",
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

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

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Trainer
    trainer = PedalTrainer1(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logging_steps=10,
        eval_epochs=10,
        save_total_limit=10,
        num_train_epochs=500,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        val_label_bin_edges=val_label_bin_edges,
        save_dir=save_dir,
    )

    # Train the model
    trainer.train(
        low_res_pedal_ratio=low_res_pedal_ratio,
        pedal_value_ratio=pedal_value_ratio,
        pedal_onset_ratio=pedal_onset_ratio,
        pedal_offset_ratio=pedal_offset_ratio,
        room_ratio=room_ratio,
        contrastive_ratio=contrastive_ratio,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
