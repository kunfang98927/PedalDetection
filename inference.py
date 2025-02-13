import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.model import PedalDetectionModel
from src.utils import load_data, split_data, visualize_attention, visualize_clusters


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
        pedal_outputs, room_outputs, latent_repr = model(feature, loss_mask=loss_mask)
        pedal_predictions = torch.argmax(torch.softmax(pedal_outputs, dim=-1), dim=-1)
        room_predictions = torch.argmax(torch.softmax(room_outputs, dim=-1), dim=-1)
    return (
        pedal_predictions.squeeze(0).cpu().numpy(),
        room_predictions.squeeze(0).cpu().numpy(),
        latent_repr.cpu().numpy(),
    )


def main():
    # Parameters
    checkpoint_path = "room1_ckpt_100per-clip-128cls-data2-100frm_p0.6-r0.0-c0.4_1-127_bs32_fctr1.0/model_epoch_190_val_loss_2.6313_val_pedal_f1_0.2405.pt"
    # checkpoint_path = "test-room1-ckpt_rand-smpl_500per-clip-3cls-data2-50frm_p0.5-r0.1-c0.4_11-95_bs128_fctr1.0/model_epoch_30_val_loss_0.6139_val_pedal_f1_0.5476.pt"
    # checkpoint_path = "test-ckpt_rand-smpl_10per-clip-3cls-data2-500frm_p0.5-r0.0-c0.4_11-95_bs32_fctr1.0/model_epoch_60_val_loss_0.5979_val_pedal_f1_0.5810.pt"
    # checkpoint_path = "sbatch-ckpt_rand-smpl_10per-clip-3cls-data2-500frm_p0.5-r0.1-c0.4_11-95_bs32_fctr0.5&1.0&2.0/model_epoch_50_val_loss_0.5235_val_pedal_f1_0.6062.pt"
    # checkpoint_path = "sbatch-ckpt_rand-smpl_10per-clip-3cls-data2-500frm_p0.5-r0.1-c0.4_11-95_bs32_fctr0.0&0.5&1.0&2.0/model_epoch_60_val_loss_0.5131_val_pedal_f1_0.5797.pt"
    # checkpoint_path = "sbatch-ckpt_rand-smpl_50per-clip-3cls-data2-500frm_p0.5-r0.1-c0.4_11-95_bs32_fctr1.0/model_epoch_30_val_loss_0.6691_val_pedal_f1_0.4854.pt"
    # checkpoint_path = "sbatch-ckpt_rand-smpl_10per-clip-3cls-data2-500frm_p0.5-r0.1-c0.4_11-95_bs32_fctr1.0/model_epoch_70_val_loss_0.6700_val_pedal_f1_0.3661.pt"
    feature_dim = 141
    max_frame = 100
    hidden_dim = 256
    num_heads = 8
    ff_dim = 256
    num_layers = 8
    num_classes = 128
    num_sample_per_clip = 10
    pedal_factor = [1.0]
    room_acoustics = [1.0]

    label_bin_edges = []
    if num_classes == 3:
        label_bin_edges = [0, 11, 95, 128]
    elif num_classes == 4:
        label_bin_edges = [0, 11, 60, 100, 128]
    elif num_classes == 2:
        label_bin_edges = [0, 11, 128]
    elif num_classes == 128:
        label_bin_edges = range(129)
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

    # Perform inference
    label_ratio = 1.0
    pedal_accs = []
    room_accs = []
    latent_reprs = []
    pedal_predictions = []
    room_predictions = []
    label_regions = []

    img_count = 0
    for i in range(num_sample_per_clip):
        for feature, label, metadata in zip(test_features, test_labels, test_metadata):
            quantized_label = np.digitize(label, label_bin_edges) - 1

            # Segment by max_frame
            start_frame = np.random.randint(0, feature.shape[1] - max_frame)
            selected_feature = feature[:, start_frame : start_frame + max_frame].T
            selected_label = quantized_label[start_frame : start_frame + max_frame]

            selected_feature = torch.tensor(
                selected_feature, dtype=torch.float32
            ).unsqueeze(0)

            # only get the middle region of the label
            label_start = int((1 - label_ratio) / 2 * max_frame)
            label_end = int((1 + label_ratio) / 2 * max_frame)
            label_masked = np.full(selected_label.shape, -1)
            selected_label = selected_label[label_start:label_end]
            label_masked[label_start:label_end] = selected_label
            loss_mask = label_masked != -1
            loss_mask = (
                torch.tensor(loss_mask, dtype=torch.bool).unsqueeze(0).to(device)
            )

            pedal_prediction, room_prediction, latent_repr = infer(
                model, selected_feature, loss_mask, device=device
            )

            pedal_prediction = pedal_prediction[label_start:label_end]
            latent_repr = latent_repr[:, label_start:label_end]

            pedal_acc = (pedal_prediction == selected_label).sum() / len(selected_label)
            pedal_accs.append(pedal_acc)
            room_acc = (room_prediction == (metadata[0] - 1)).sum() / len(
                selected_label
            )
            room_accs.append(room_acc)
            latent_reprs.append(latent_repr)
            pedal_predictions.append(pedal_prediction)
            room_predictions.append(room_prediction)
            label_regions.append(selected_label)

            # plt.figure(figsize=(8, 6))
            # plt.subplot(3, 1, 1)
            # plt.plot(selected_label, label="True(quantized)")
            # plt.plot(pedal_prediction, label="Predicted")
            # plt.xlabel("Frame Index")
            # plt.ylabel("Pedal Label")
            # plt.ylim(-0.1, 2.1)
            # plt.legend()

            # plt.subplot(3, 1, 2)
            # plt.plot(label[start_frame : start_frame + max_frame], label="Label")
            # plt.xlabel("Frame Index")
            # plt.ylabel("Pedal Label")
            # plt.legend()

            # plt.subplot(3, 1, 3)
            # plt.imshow(selected_feature.T, aspect="auto", origin="lower")
            # plt.xlabel("Frame Index")
            # plt.ylabel("Feature Index")

            # plt.savefig(f"imgs/inference_{img_count}_{metadata[1]}_{metadata[2]}.png")
            # plt.close()
            # img_count += 1

    # # visualize attention weights
    # visualize_attention(
    #     model,
    #     num_layers,
    #     num_heads,
    #     save_path="inference_results/attention_plot.png",
    # )

    # visualize clusters
    print(len(latent_reprs), latent_reprs[0].shape)
    latent_reprs = np.concatenate(np.array(latent_reprs), axis=0)
    print(latent_reprs.shape)
    latent_reprs = latent_reprs.reshape(-1, latent_reprs.shape[-1])
    pedal_predictions = np.concatenate(np.array(pedal_predictions), axis=0)
    label_regions = np.concatenate(np.array(label_regions), axis=0)
    print(latent_reprs.shape, pedal_predictions.shape, label_regions.shape)
    label_count = {0: 0, 1: 0, 2: 0}
    for label, pred in zip(label_regions, pedal_predictions):
        if label == 0.0:
            label_count[0] += 1
        elif label == 1.0:
            label_count[1] += 1
        elif label == 2.0:
            label_count[2] += 1
    print(label_count)

    # Pedal classification report
    print("Average Pedal Accuracy:", sum(pedal_accs) / len(pedal_accs))
    cr_pedal = classification_report(label_regions, pedal_predictions)
    print(cr_pedal)

    # Room classification report
    print("Average Room Accuracy:", sum(room_accs) / len(room_accs))
    cr_room = classification_report(
        [metadata[0] - 1 for metadata in test_metadata] * num_sample_per_clip,
        room_predictions,
        target_names=["dry room", "clean studio", "concert hall"],
    )
    print(cr_room)

    # Save classification reports in same file
    with open(
        "inference_results/classification_reports.txt",
        "w",
    ) as f:
        f.write("Pedal Classification Report:\n")
        f.write(cr_pedal)
        f.write("\n\n")
        f.write("Room Classification Report:\n")
        f.write(cr_room)

    # save confusion matrix
    pedal_cm = confusion_matrix(label_regions, pedal_predictions)
    room_cm = confusion_matrix(
        [metadata[0] - 1 for metadata in test_metadata] * num_sample_per_clip,
        room_predictions,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(pedal_cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.title("Pedal Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.subplot(1, 2, 2)
    sns.heatmap(room_cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.title("Room Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig("inference_results/confusion_matrix.png")

    visualize_clusters(
        latent_reprs,
        pedal_predictions,
        label_regions,
        save_path="inference_results/clusters.png",
    )


if __name__ == "__main__":
    main()
