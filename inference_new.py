import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, f1_score, mean_squared_error
from src.model import PedalDetectionModel
from src.dataset import PedalDataset
from src.utils import load_data, split_data, get_label_bin_edges, load_data_real_audio, split_data_real_audio, plot_pedal_pred


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
        low_res_p_logits, p_v_logits, p_on_logits, p_off_logits, room_logits, latent_repr = model(feature, loss_mask=loss_mask)
        p_v_logits = p_v_logits[loss_mask]
        p_on_logits = p_on_logits[loss_mask]
        p_off_logits = p_off_logits[loss_mask]
        low_res_p_v_preds = low_res_p_logits # torch.argmax(torch.softmax(low_res_p_logits, dim=-1), dim=-1)
        p_v_preds = p_v_logits # torch.argmax(torch.softmax(p_v_logits, dim=-1), dim=-1)
        p_on_preds = torch.sigmoid(p_on_logits)
        p_off_preds = torch.sigmoid(p_off_logits)
        room_preds = torch.argmax(torch.softmax(room_logits, dim=-1), dim=-1)
    return (
        low_res_p_v_preds.squeeze(0).cpu().numpy(),
        p_v_preds.squeeze(0).cpu().numpy(),
        p_on_preds.squeeze(0).cpu().numpy(),
        p_off_preds.squeeze(0).cpu().numpy(),
        room_preds.squeeze(0).cpu().numpy()
    )


def main():
    # Parameters
    checkpoint_path = "ckpt-mse-2fac-200fr-cntxt-fullroom1-big-mixres-1/model_epoch_60_val_loss_0.0458_val_f1_0.7385.pt"
    # checkpoint_path = "ckpt-mse-2fac-100fr-cntxt-fullroom1/model_epoch_70_val_loss_0.0461_val_f1_0.8458.pt"
    # checkpoint_path = "ckpt-mse-2fac-100fr-cntxt-real/model_epoch_160_val_loss_0.1238_val_f1_0.6838.pt" # best model (real audio)
    # checkpoint_path = "ckpt-mse-2fac-100fr-cntxt/model_epoch_160_val_loss_0.0519_val_f1_0.8261.pt" # best model
    # checkpoint_path = "ckpt-real/model_epoch_50_val_loss_0.0733_val_f1_0.5906.pt" # best model (real audio)
    # checkpoint_path = "ckpt-test-mse-aug-newdata-2factor-labelratio1-mf100/model_epoch_60_val_loss_0.0191_val_f1_0.8895.pt" # best model
    # checkpoint_path = "ckpt-test-mse-aug-newdata-2factor-labelratio1/model_epoch_80_val_loss_0.0328_val_f1_0.7634.pt" # best model
    # checkpoint_path = "ckpt-test-mse-aug-newdata-2factor/model_epoch_120_val_loss_0.0378_val_f1_0.7176.pt" # best model
    # checkpoint_path = "ckpt-test-mse-aug-newdata/model_epoch_60_val_loss_0.0364_val_f1_0.8220.pt"
    # checkpoint_path = "ckpt-test-mse-1/model_epoch_40_val_loss_0.0783_val_f1_0.5631.pt"
    # checkpoint_path = "ckpt-test-2/model_epoch_180_val_loss_0.0251_val_low-res-pedal_f1_0.7205.pt"
    feature_dim = 141
    max_frame = 200
    hidden_dim = 1024 #256
    num_heads = 8
    ff_dim = 1024 #256
    num_layers = 12 #8
    num_classes = 1 # 1 stands for MSE loss
    num_sample_per_clip = None
    pedal_factor = [1.0]
    room_acoustics = [1.0]

    label_bin_edges = get_label_bin_edges(num_classes)
    inf_label_bin_edges = [0, 11, 128]
    # inf_label_bin_edges = [0, 16, 32, 48, 64, 80, 96, 112, 128]

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
    # data_path = "data/processed_data_4096_NormPerFeat.npz"
    # data_path = "data/processed_data_real_audio_4096.npz"
    data_path = "data/processed_data_4096_full_room1.npz"

    # Load data
    if "real" in data_path:
        features, labels, metadata = load_data_real_audio(data_path, label_bin_edges)
        features, labels, metadata = split_data_real_audio(features, labels, metadata, split="test", max_num_samples=None)
        print("Split(Test) dataset size:", len(features), len(labels), len(metadata))
    elif "full" in data_path:
        features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
        features, labels, metadata = split_data_real_audio(features, labels, metadata, split="test", max_num_samples=None)
    else:
        features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
        _, _, features, _, _, labels, _, _, metadata = split_data(
            features, labels, metadata, val_size=0.15, test_size=0.15, random_state=100
        )

    # Dataset and DataLoader
    test_dataset = PedalDataset(
        features=features,
        labels=labels,
        metadata=metadata,
        num_samples_per_clip=num_sample_per_clip,
        max_frame=max_frame,
        label_ratio=0.6,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.4,
        split="test",
    )
    print("Test dataset size:", len(test_dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Perform inference
    pedal_onset_maes = []
    pedal_offset_maes = []

    quantized_all_low_res_p_labels = []
    quantized_all_low_res_p_preds = []
    all_low_res_p_labels = []
    all_low_res_p_preds = []

    quantized_all_p_v_labels = []
    quantized_all_p_v_preds = []
    all_p_v_labels = []
    all_p_v_preds = []
    for inputs, low_res_p_labels, p_v_labels, p_on_labels, p_off_labels, room_labels, midi_ids, pedal_factors in test_dataloader:
        inputs, low_res_p_labels, p_v_labels, p_on_labels, p_off_labels = inputs.to(device), low_res_p_labels.to(device), p_v_labels.to(device), p_on_labels.to(device), p_off_labels.to(device)
        room_labels, midi_ids, pedal_factors = room_labels.to(device), midi_ids.to(device), pedal_factors.to(device)

        loss_mask = p_v_labels != -1
        low_res_p_preds, p_v_preds, p_on_preds, p_off_preds, room_preds = infer(model, inputs, loss_mask, device=device)

        # apply loss_mask
        p_v_labels = p_v_labels[loss_mask]
        p_on_labels = p_on_labels[loss_mask]
        p_off_labels = p_off_labels[loss_mask]

        # Low resolution pedal value prediction
        low_res_p_labels = low_res_p_labels.cpu().numpy().squeeze()
        low_res_p_preds = low_res_p_preds.squeeze()
        low_res_p_preds[low_res_p_preds < 0] = 0
        low_res_p_preds[low_res_p_preds > 1] = 1

        all_low_res_p_labels.append(low_res_p_labels)
        all_low_res_p_preds.append(low_res_p_preds)

        quantized_low_res_p_labels = np.digitize(low_res_p_labels * 127, inf_label_bin_edges) - 1
        quantized_low_res_p_preds = np.digitize(low_res_p_preds * 127, inf_label_bin_edges) - 1

        quantized_all_low_res_p_labels.append(quantized_low_res_p_labels)
        quantized_all_low_res_p_preds.append(quantized_low_res_p_preds)

        # Pedal value prediction
        p_v_labels = p_v_labels.cpu().numpy().squeeze()
        p_v_preds = p_v_preds.squeeze()
        p_v_preds[p_v_preds < 0] = 0
        p_v_preds[p_v_preds > 1] = 1

        all_p_v_labels.append(p_v_labels)
        all_p_v_preds.append(p_v_preds)

        quantized_p_v_labels = np.digitize(p_v_labels * 127, inf_label_bin_edges) - 1
        quantized_p_v_preds = np.digitize(p_v_preds * 127, inf_label_bin_edges) - 1

        quantized_all_p_v_labels.append(quantized_p_v_labels)
        quantized_all_p_v_preds.append(quantized_p_v_preds)

        # p_on_labels = p_on_labels.cpu().numpy()
        # p_off_labels = p_off_labels.cpu().numpy()
        # p_on_labels = p_on_labels.squeeze()
        # p_off_labels = p_off_labels.squeeze()
        # p_on_preds = p_on_preds.squeeze()
        # p_off_preds = p_off_preds.squeeze()

        # # Measure pedal onset prediction
        # pedal_onset_mae = mean_absolute_error(p_on_labels, p_on_preds)
        # p_on_threshold = 0.5
        # p_on_preds[p_on_preds >= p_on_threshold] = 1
        # p_on_preds[p_on_preds < p_on_threshold] = 0
        # p_on_labels[p_on_labels >= p_on_threshold] = 1
        # p_on_labels[p_on_labels < p_on_threshold] = 0
        # pedal_onset_maes.append(pedal_onset_mae)

        # # Measure pedal offset prediction
        # pedal_offset_mae = mean_absolute_error(p_off_labels, p_off_preds)
        # p_off_threshold = 0.5
        # p_off_preds[p_off_preds >= p_off_threshold] = 1
        # p_off_preds[p_off_preds < p_off_threshold] = 0
        # p_off_labels[p_off_labels >= p_off_threshold] = 1
        # p_off_labels[p_off_labels < p_off_threshold] = 0
        # pedal_offset_maes.append(pedal_offset_mae)

        # # Plot pedal onset and offset, prediction vs. soft labels
        # plt.figure(figsize=(12, 6))
        # plt.subplot(2, 1, 1)
        # plt.plot(p_on_labels, label="Pedal Onset Labels")
        # plt.plot(p_on_preds, label="Pedal Onset Predictions")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Pedal Onset")
        # plt.legend()
        # plt.subplot(2, 1, 2)
        # plt.plot(p_off_labels, label="Pedal Offset Labels")
        # plt.plot(p_off_preds, label="Pedal Offset Predictions")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Pedal Offset")
        # plt.legend()
        # plt.tight_layout()
        # # save the plot
        # plt.savefig(f"pedal_pred_{img_count}.png")
        # plt.close()
        # img_count += 1

    print(len(all_p_v_labels), len(all_p_v_preds))
    np.save("p_v_labels_test_set_synth.npy", all_p_v_labels)
    np.save("p_v_preds_test_set_synth.npy", all_p_v_preds)
    all_p_v_labels = np.concatenate(all_p_v_labels)
    all_p_v_preds = np.concatenate(all_p_v_preds)
    print(all_p_v_labels.shape, all_p_v_preds.shape)

    # store the results
    np.save("global_p_labels_test_set_synth.npy", all_low_res_p_labels)
    np.save("global_p_preds_test_set_synth.npy", all_low_res_p_preds)

    # Measure pedal value prediction: MAE (all_low_res_p_labels, all_low_res_p_preds)
    low_res_p_mae = mean_absolute_error(all_low_res_p_labels, all_low_res_p_preds)
    print("Low Res Pedal Value MAE:", low_res_p_mae)

    # Measure pedal value prediction: MSE (all_low_res_p_labels, all_low_res_p_preds)
    low_res_p_mse = mean_squared_error(all_low_res_p_labels, all_low_res_p_preds)
    print("Low Res Pedal Value MSE:", low_res_p_mse)

    # Measure pedal value prediction: f1 score
    low_res_p_f1 = f1_score(quantized_all_low_res_p_labels, quantized_all_low_res_p_preds, average="weighted")
    print("Low Res Pedal Value F1 Score:", low_res_p_f1)

    # classification report
    print("Classification Report:")
    print(classification_report(quantized_all_low_res_p_labels, quantized_all_low_res_p_preds))
    # confusion matrix
    cm = confusion_matrix(quantized_all_low_res_p_labels, quantized_all_low_res_p_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # save the plot
    plt.savefig(f"low_res_{len(inf_label_bin_edges)-1}.png")
    plt.close()

    # # Pedal value
    # print("Total Frames:", len(pedal_onset_maes))
    # print("Average Pedal Onset MAE:", sum(pedal_onset_maes) / len(pedal_onset_maes))
    # print("Average Pedal Offset MAE:", sum(pedal_offset_maes) / len(pedal_offset_maes))

    plot_pedal_pred(all_low_res_p_labels, all_low_res_p_preds, "low_res")
    plot_pedal_pred(all_p_v_labels, all_p_v_preds, "p_v", img_num_frames=500)


if __name__ == "__main__":
    main()
