import torch
from src.model import PedalDetectionModel
from src.dataset import PedalDataset
from src.trainer import PedalTrainer
from src.utils import load_data, split_data


def main():

    # Feature dimension
    feature_dim = 141 # 128 (spectrogram) + 13 (mfcc)
    max_frame = 1000

    # Data path
    data_path = "data/processed_data.npz"

    # Load data
    features, labels, metadata = load_data(data_path)
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
    )
    val_dataset = PedalDataset(
        features=val_features,
        labels=val_labels,
        metadata=val_metadata,
        num_samples_per_clip=10,
        max_frame=max_frame,
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # Model
    model = PedalDetectionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=2,
        ff_dim=256,
        num_layers=4,
        num_classes=128,
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
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
        train_batch_size=64,
        val_batch_size=64,
        save_dir="checkpoints-0",
    )

    # Train the model
    trainer.train(alpha=1.0, beta=0.0)


if __name__ == "__main__":
    main()
