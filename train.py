import torch
from src.model import PedalDetectionModel
from src.dataset import PedalDataset
from src.trainer import PedalTrainer
from src.utils import load_data, split_data


def main():

    data_version = "2"

    # Feature dimension
    batch_size = 32
    feature_dim = 141 # 128 (spectrogram) + 13 (mfcc)
    max_frame = 500
    num_classes = 3
    pedal_ratio = 0.5
    room_ratio = 0.1
    contrastive_ratio = 0.4
    pedal_factor = [1.0]
    label_bin_edges = []
    if num_classes == 3:
        label_bin_edges = [0, 10, 95, 128]
    elif num_classes == 4:
        label_bin_edges = [0, 10, 60, 100, 128]
    elif num_classes == 2:
        label_bin_edges = [0, 10, 128]

    # Data path
    data_path = f"data/processed_data{data_version}.npz"

    # Load data
    features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor)
    train_features, val_features, _, train_labels, val_labels, _, train_metadata, val_metadata, _ = split_data(
        features, labels, metadata, val_size=0.1, test_size=0.1, random_state=42
    )

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        features=train_features,
        labels=train_labels,
        metadata=train_metadata,
        num_samples_per_clip=10,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
    )
    val_dataset = PedalDataset(
        features=val_features,
        labels=val_labels,
        metadata=val_metadata,
        num_samples_per_clip=10,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Model
    model = PedalDetectionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=8,
        ff_dim=256,
        num_layers=4,
        num_classes=num_classes,
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
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
        save_total_limit=5,
        num_train_epochs=500,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        save_dir=f"checkpoints-{num_classes}class-data{data_version}-{max_frame}frame_p{pedal_ratio}-r{room_ratio}-c{contrastive_ratio}_10-95_bs{batch_size}_factor1.0",
    )

    # Train the model
    trainer.train(pedal_ratio=pedal_ratio, room_ratio=room_ratio, contrastive_ratio=contrastive_ratio)


if __name__ == "__main__":
    main()
