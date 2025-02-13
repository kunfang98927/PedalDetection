import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from src.model import PedalDetectionModel
from src.dataset import PedalDataset
from src.utils import load_data, split_data, get_label_bin_edges


def load_model(
    checkpoint_path,
    input_dim,
    hidden_dim,
    num_heads,
    ff_dim,
    num_layers,
    num_classes,
    device="cpu",
):
    model = PedalDetectionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def infer(model, feature, loss_mask, device="cpu"):
    feature = feature.to(device)
    with torch.no_grad():
        p_v_logits, p_on_logits, p_off_logits, room_logits, latent_repr = model(feature, loss_mask=loss_mask)
        p_v_preds = torch.argmax(torch.softmax(p_v_logits, dim=-1), dim=-1)
        p_on_preds = torch.sigmoid(p_on_logits)
        p_off_preds = torch.sigmoid(p_off_logits)
        room_preds = torch.argmax(torch.softmax(room_logits, dim=-1), dim=-1)
    return (
        p_v_preds.squeeze(0).cpu().numpy(),
        p_on_preds.squeeze(0).cpu().numpy(),
        p_off_preds.squeeze(0).cpu().numpy(),
        room_preds.squeeze(0).cpu().numpy()
    )


def main():
    # Parameters
    checkpoint_path = "ckpt-on-thres11/model_epoch_100_val_loss_0.1368_val_pedal_f1_0.0812.pt"
    feature_dim = 141
    max_frame = 500
    hidden_dim = 256
    num_heads = 8
    ff_dim = 256
    num_layers = 8
    num_classes = 2
    num_sample_per_clip = 10
    pedal_factor = [1.0]
    room_acoustics = [1.0]

    label_bin_edges = get_label_bin_edges(num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(
        checkpoint_path,
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        device=device,
    )

    # Data path
    data_path = "data/processed_data2.npz"

    # Load data
    features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
    _, _, test_features, _, _, test_labels, _, _, test_metadata = split_data(
        features, labels, metadata, val_size=0.15, test_size=0.15, random_state=100
    )

    # Dataset and DataLoader
    test_dataset = PedalDataset(
        features=test_features,
        labels=test_labels,
        metadata=test_metadata,
        num_samples_per_clip=num_sample_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
        split="test",
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Perform inference
    pedal_onset_maes = []
    pedal_offset_maes = []

    img_count = 0
    for inputs, p_v_labels, p_on_labels, p_off_labels, room_labels, midi_ids, pedal_factors in test_dataloader:
        inputs, p_v_labels, p_on_labels, p_off_labels = inputs.to(device), p_v_labels.to(device), p_on_labels.to(device), p_off_labels.to(device)
        room_labels, midi_ids, pedal_factors = room_labels.to(device), midi_ids.to(device), pedal_factors.to(device)

        loss_mask = p_v_labels != -1
        p_v_preds, p_on_preds, p_off_preds, room_preds = infer(model, inputs, loss_mask, device=device)

        p_on_labels = p_on_labels.cpu().numpy()
        p_off_labels = p_off_labels.cpu().numpy()
        p_on_labels = p_on_labels.squeeze()
        p_off_labels = p_off_labels.squeeze()
        p_on_preds = p_on_preds.squeeze()
        p_off_preds = p_off_preds.squeeze()

        # Measure pedal onset prediction
        pedal_onset_mae = mean_absolute_error(p_on_labels, p_on_preds)
        p_on_threshold = 0.5
        p_on_preds[p_on_preds >= p_on_threshold] = 1
        p_on_preds[p_on_preds < p_on_threshold] = 0
        p_on_labels[p_on_labels >= p_on_threshold] = 1
        p_on_labels[p_on_labels < p_on_threshold] = 0
        pedal_onset_maes.append(pedal_onset_mae)

        # Measure pedal offset prediction
        pedal_offset_mae = mean_absolute_error(p_off_labels, p_off_preds)
        p_off_threshold = 0.5
        p_off_preds[p_off_preds >= p_off_threshold] = 1
        p_off_preds[p_off_preds < p_off_threshold] = 0
        p_off_labels[p_off_labels >= p_off_threshold] = 1
        p_off_labels[p_off_labels < p_off_threshold] = 0
        pedal_offset_maes.append(pedal_offset_mae)

        # Plot pedal onset and offset, prediction vs. soft labels
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(p_on_labels, label="Pedal Onset Labels")
        plt.plot(p_on_preds, label="Pedal Onset Predictions")
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Onset")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(p_off_labels, label="Pedal Offset Labels")
        plt.plot(p_off_preds, label="Pedal Offset Predictions")
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Offset")
        plt.legend()
        plt.tight_layout()
        # save the plot
        plt.savefig(f"pedal_pred_{img_count}.png")
        plt.close()
        img_count += 1

    # Pedal value
    print("Total Frames:", len(pedal_onset_maes))
    print("Average Pedal Onset MAE:", sum(pedal_onset_maes) / len(pedal_onset_maes))
    print("Average Pedal Offset MAE:", sum(pedal_offset_maes) / len(pedal_offset_maes))


if __name__ == "__main__":
    main()
