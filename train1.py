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

    data_version_synth = "_kong504room3synth0220"  # "_4096_full_room1" # "_4096_NormPerFeat" # "_real_audio_4096"
    data_version_real = "_kong508room1real20250217"  # "_real_audio_4096"

    # Feature dimension
    batch_size = 256
    feature_dim = 249  # 128 (spectrogram) + 13 (mfcc)
    max_frame = 100
    num_samples_per_clip = 50
    num_classes = 1

    low_res_pedal_ratio = 0.5
    pedal_value_ratio = 0.5
    pedal_onset_ratio = 0.0
    pedal_offset_ratio = 0.0
    room_ratio = 0.0
    contrastive_ratio = 0.5

    pedal_factor = [0.0]  # pedal factor for synthetic data
    room_acoustics = [1.0]

    label_bin_edges = get_label_bin_edges(num_classes)
    val_label_bin_edges = get_label_bin_edges(3)

    # Checkpoint save path
    label_bin_edge_str = str(label_bin_edges[1]) + "-" + str(label_bin_edges[-2])
    factor_str = "&".join([str(f) for f in pedal_factor])
    save_dir = f"ckpt-mse-0222-mf100-room3-kongfeat-cnn"
    # save_dir = f"ckpt_{num_samples_per_clip}per-clip-{num_classes}cls-data{data_version}-{max_frame}frm_p{pedal_value_ratio}-r{room_ratio}-c{contrastive_ratio}_{label_bin_edge_str}_bs{batch_size}_fctr{factor_str}"

    # Copy this file to save_dir
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("train.py", os.path.join(save_dir, "train"))

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
    print("Real audio train dataset size:", len(train_features_real))
    print("Synthetic audio train dataset size:", len(train_features_synth))
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
        overlap_ratio=0.2,
        split="validation",
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Model
    model = PedalDetectionModelwithCNN(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=8,
        ff_dim=1024, #256,
        num_layers=8,
        num_classes=num_classes,
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

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
        save_total_limit=20,
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
    )


if __name__ == "__main__":
    main()
