import torch
from src.model import PedalDetectionModel
from src.dataset import PedalDataset
from src.trainer import PedalTrainer
from src.utils import prepare_data
from sklearn.model_selection import train_test_split


def main():
    # Set parameters
    seq_len = 1000
    feature_dim = 64
    max_acoustic_setting_value = 5
    acoustic_setting_dim = 3
    data_samples = 640

    # Prepare data
    spectrograms, labels, acoustic_settings = prepare_data(
        seq_len,
        feature_dim,
        data_samples,
        max_acoustic_setting_value,
        acoustic_setting_dim,
    )

    # Split data into train and validation sets
    (
        train_spectrograms,
        val_spectrograms,
        train_labels,
        val_labels,
        train_acoustic_settings,
        val_acoustic_settings,
    ) = train_test_split(
        spectrograms, labels, acoustic_settings, test_size=0.2, random_state=42
    )

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        train_spectrograms,
        train_labels,
        train_acoustic_settings,
    )
    val_dataset = PedalDataset(val_spectrograms, val_labels, val_acoustic_settings)

    # Model
    model = PedalDetectionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        num_heads=2,
        ff_dim=256,
        num_layers=4,
        num_classes=3,
    )
    print(model)

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
        num_train_epochs=200,
        train_batch_size=32,
        val_batch_size=32,
        save_dir="checkpoints-12",
    )

    # Train the model
    trainer.train(alpha=0.8, beta=0.2)


if __name__ == "__main__":
    main()
