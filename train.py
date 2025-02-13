import os
import torch
import shutil
from src.model import PedalDetectionModel
from src.dataset import PedalDataset
from src.trainer import PedalTrainer
from src.utils import load_data, split_data


def main():

    data_version = "2"

    # Feature dimension
    batch_size = 64
    feature_dim = 141  # 128 (spectrogram) + 13 (mfcc)
    max_frame = 50
    num_classes = 128
    pedal_ratio = 0.6
    room_ratio = 0.0
    contrastive_ratio = 0.4
    pedal_factor = [1.0]
    room_acoustics = [1.0]

    label_bin_edges = []
    if num_classes == 3:
        label_bin_edges = [0, 11, 95, 128]
    elif num_classes == 4:
        label_bin_edges = [0, 11, 60, 95, 128]
    elif num_classes == 2:
        label_bin_edges = [0, 11, 128]
    elif num_classes == 128:
        label_bin_edges = range(129)

    # Checkpoint save path
    label_bin_edge_str = str(label_bin_edges[1]) + "-" + str(label_bin_edges[-2])
    factor_str = "&".join([str(f) for f in pedal_factor])
    save_dir = f"room1_ckpt_1000per-clip-{num_classes}cls-data{data_version}-{max_frame}frm_p{pedal_ratio}-r{room_ratio}-c{contrastive_ratio}_{label_bin_edge_str}_bs{batch_size}_fctr{factor_str}"

    # Copy this file to save_dir
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("train.py", os.path.join(save_dir, "train"))

    # Data path
    data_path = f"data/processed_data{data_version}.npz"

    # Load data
    features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
    (
        train_features,
        val_features,
        _,
        train_labels,
        val_labels,
        _,
        train_metadata,
        val_metadata,
        _,
    ) = split_data(
        features, labels, metadata, val_size=0.15, test_size=0.15, random_state=100
    )

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        features=train_features,
        labels=train_labels,
        metadata=train_metadata,
        num_samples_per_clip=1000,
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
        num_samples_per_clip=100,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
        split="validation",
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Model
    model = PedalDetectionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=8,
        ff_dim=256,
        num_layers=8,
        num_classes=num_classes,
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Trainer
    trainer = PedalTrainer(
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
        num_train_epochs=200,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        save_dir=save_dir,
    )

    # Train the model
    trainer.train(
        pedal_ratio=pedal_ratio,
        room_ratio=room_ratio,
        contrastive_ratio=contrastive_ratio,
    )


if __name__ == "__main__":
    main()
