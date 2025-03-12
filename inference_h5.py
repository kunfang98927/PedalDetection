import os
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
)
from src.model1 import PedalDetectionModelwithCNN
from src.dataset_h5 import PedalDataset
from src.utils import (
    load_data,
    split_data,
    get_label_bin_edges,
    load_data_real_audio,
    split_data_real_audio,
    plot_pedal_pred,
)


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
    model = PedalDetectionModelwithCNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def infer(model, feature, loss_mask, device="cpu"):
    feature = feature.to(device)
    with torch.no_grad():
        (
            global_p_logits,
            p_v_logits,
            p_on_logits,
            p_off_logits,
            room_logits,
            latent_repr,
        ) = model(feature, loss_mask=loss_mask)

        # pedal value for each frame
        p_v_logits = p_v_logits[loss_mask]
        p_v_preds = p_v_logits

        # pedal onset
        p_on_logits = p_on_logits[loss_mask]
        p_on_preds = torch.sigmoid(p_on_logits)

        # pedal offset
        p_off_logits = p_off_logits[loss_mask]
        p_off_preds = torch.sigmoid(p_off_logits)

        # global pedal value
        global_p_v_preds = global_p_logits

        # room
        room_preds = torch.argmax(torch.softmax(room_logits, dim=-1), dim=-1)

    return (
        global_p_v_preds.squeeze().cpu().numpy(),
        p_v_preds.squeeze().cpu().numpy(),
        p_on_preds.squeeze().cpu().numpy(),
        p_off_preds.squeeze().cpu().numpy(),
        room_preds.squeeze().cpu().numpy(),
    )


def main():
    # Parameters
    # checkpoint_path = "ckpt_0303_10per-clip-500frm_bs32_no-onoff_sbatch/model_epoch_450_val_loss_0.1921_f1_0.6449_mae_0.1736.pt"
    # checkpoint_path = "ckpt_0306_10per-clip-500frm_bs32_onoff_sbatch_resume/model_epoch_500_val_loss_0.6021_f1_0.8045_mae_0.1605.pt" # best 1st model
    # checkpoint_path = "ckpt_0306_10per-clip-500frm_bs32_onoff-bce_sbatch/model_epoch_370_val_loss_0.1002_f1_0.7635_mae_0.1544.pt"  # best 2nd model
    checkpoint_path = "ckpt_0312_10per-clip-500frm_bs16-8h-2xdata-val1-6loss-resume/model_epoch_8_step_1200_val_loss_0.1165_f1_0.8290_mae_0.2024.pt"

    # Get the name of the checkpoint
    ckpt_name = checkpoint_path.split("/")[-1]
    result_dir = checkpoint_path.split("/")[0]
    result_dir = f"{result_dir}/results"
    print(f"Result directory: {result_dir}")
    os.makedirs(result_dir, exist_ok=True)
    report_path = f"{result_dir}/report.txt"
    with open(report_path, "a") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")

    feature_dim = 249
    max_frame = 500
    hidden_dim = 256  # 256
    num_heads = 8
    ff_dim = 1024  # 256
    num_layers = 8
    num_classes = 1  # 1 stands for MSE loss

    label_bin_edges = get_label_bin_edges(num_classes)
    inf_label_bin_edges = [0, 64, 128]
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

    # Data loading
    test_dataset = PedalDataset(
        data_path="sample_data/test.json",
        num_samples_per_clip=1,  # num_samples_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.05,
        split="test",
        data_filter=["r0-pf1"],  # ["r1-pf1", "r2-pf1", "r3-pf1", "r0-pf1"],
        # num_examples=20,
    )
    print("Test dataset size:", len(test_dataset))

    # Set device (and note multi-GPU availability)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    # Perform inference
    quantized_all_global_p_labels = []
    quantized_all_global_p_preds = []
    all_global_p_labels = []
    all_global_p_preds = []

    quantized_all_p_v_labels = []
    quantized_all_p_v_preds = []
    all_p_v_labels = []
    all_p_v_preds = []

    all_p_onset_labels = []
    all_p_onset_preds = []

    all_p_offset_labels = []
    all_p_offset_preds = []

    all_room_labels = []
    all_room_preds = []

    for (
        inputs,
        global_p_labels,
        p_v_labels,
        p_on_labels,
        p_off_labels,
        room_labels,
        midi_ids,
        pedal_factors,
    ) in tqdm(test_dataloader):
        inputs, global_p_labels, p_v_labels, p_on_labels, p_off_labels = (
            inputs.to(device),
            global_p_labels.to(device),
            p_v_labels.to(device),
            p_on_labels.to(device),
            p_off_labels.to(device),
        )
        room_labels, midi_ids, pedal_factors = (
            room_labels.to(device),
            midi_ids.to(device),
            pedal_factors.to(device),
        )

        loss_mask = p_v_labels != -1
        global_p_preds, p_v_preds, p_on_preds, p_off_preds, room_preds = infer(
            model, inputs, loss_mask, device=device
        )

        # Apply loss_mask
        p_v_labels = p_v_labels[loss_mask]
        p_on_labels = p_on_labels[loss_mask]
        p_off_labels = p_off_labels[loss_mask]

        # Global pedal value prediction
        global_p_labels = global_p_labels.cpu().numpy().squeeze()
        global_p_preds = global_p_preds.squeeze()
        global_p_preds[global_p_preds < 0] = 0
        global_p_preds[global_p_preds > 1] = 1

        all_global_p_labels.append(global_p_labels)
        all_global_p_preds.append(global_p_preds)

        quantized_global_p_labels = (
            np.digitize(global_p_labels * 127, inf_label_bin_edges) - 1
        )
        quantized_global_p_preds = (
            np.digitize(global_p_preds * 127, inf_label_bin_edges) - 1
        )

        quantized_all_global_p_labels.append(quantized_global_p_labels)
        quantized_all_global_p_preds.append(quantized_global_p_preds)

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

        # Pedal onset/offset prediction
        p_on_labels = p_on_labels.cpu().numpy().squeeze()
        p_off_labels = p_off_labels.cpu().numpy().squeeze()
        p_on_preds = p_on_preds.squeeze()
        p_off_preds = p_off_preds.squeeze()
        p_on_preds[p_on_preds < 0] = 0
        p_on_preds[p_on_preds > 1] = 1
        p_off_preds[p_off_preds < 0] = 0
        p_off_preds[p_off_preds > 1] = 1

        all_p_onset_labels.append(p_on_labels)
        all_p_onset_preds.append(p_on_preds)

        all_p_offset_labels.append(p_off_labels)
        all_p_offset_preds.append(p_off_preds)

        # Room prediction
        room_labels = room_labels.cpu().numpy().squeeze()
        room_preds = room_preds.squeeze()

        all_room_labels.append(room_labels)
        all_room_preds.append(room_preds)

    # pedal value
    all_p_v_labels = np.concatenate(all_p_v_labels)
    all_p_v_preds = np.concatenate(all_p_v_preds)
    print(all_p_v_labels.shape, all_p_v_preds.shape)
    np.save(
        f"{result_dir}/p_v_labels_test_set_0310.npy",
        all_p_v_labels,
    )
    np.save(
        f"{result_dir}/p_v_preds_test_set_0310.npy",
        all_p_v_preds,
    )

    # pedal onset
    all_p_onset_labels = np.concatenate(all_p_onset_labels)
    all_p_onset_preds = np.concatenate(all_p_onset_preds)
    print(all_p_onset_labels.shape, all_p_onset_preds.shape)
    np.save(
        f"{result_dir}/p_onset_labels_test_set_0310.npy",
        all_p_onset_labels,
    )
    np.save(
        f"{result_dir}/p_onset_preds_test_set_0310.npy",
        all_p_onset_preds,
    )

    # pedal offset
    all_p_offset_labels = np.concatenate(all_p_offset_labels)
    all_p_offset_preds = np.concatenate(all_p_offset_preds)
    print(all_p_offset_labels.shape, all_p_offset_preds.shape)
    np.save(
        f"{result_dir}/p_offset_labels_test_set_0310.npy",
        all_p_offset_labels,
    )
    np.save(
        f"{result_dir}/p_offset_preds_test_set_0310.npy",
        all_p_offset_preds,
    )

    # global pedal value
    all_global_p_labels = np.concatenate(all_global_p_labels)
    all_global_p_preds = np.concatenate(all_global_p_preds)
    print(all_global_p_labels.shape, all_global_p_preds.shape)
    np.save(
        f"{result_dir}/global_p_labels_test_set_0310.npy",
        all_global_p_labels,
    )
    np.save(
        f"{result_dir}/global_p_preds_test_set_0310.npy",
        all_global_p_preds,
    )

    # room
    all_room_labels = np.concatenate(all_room_labels)
    all_room_preds = np.concatenate(all_room_preds)
    print(all_room_labels.shape, all_room_preds.shape)
    np.save(
        f"{result_dir}/room_labels_test_set_0310.npy",
        all_room_labels,
    )
    np.save(
        f"{result_dir}/room_preds_test_set_0310.npy",
        all_room_preds,
    )

    # Measure pedal value prediction: MAE (all_global_p_labels, all_global_p_preds)
    global_p_mae = mean_absolute_error(all_global_p_labels, all_global_p_preds)
    print("Global Pedal Value MAE:", global_p_mae)

    # Measure pedal value prediction: MSE (all_global_p_labels, all_global_p_preds)
    global_p_mse = mean_squared_error(all_global_p_labels, all_global_p_preds)
    print("Global Pedal Value MSE:", global_p_mse)

    # Measure pedal value prediction: f1 score
    quantized_all_global_p_labels = np.concatenate(quantized_all_global_p_labels)
    quantized_all_global_p_preds = np.concatenate(quantized_all_global_p_preds)
    print(quantized_all_global_p_labels.shape, quantized_all_global_p_preds.shape)
    global_p_f1 = f1_score(
        quantized_all_global_p_labels,
        quantized_all_global_p_preds,
        average="weighted",
    )
    print("Global Pedal Value F1 Score:", global_p_f1)

    # classification report
    print("Classification Report:")
    cr = classification_report(
        quantized_all_global_p_labels, quantized_all_global_p_preds
    )
    print(cr)

    with open(report_path, "a") as f:
        f.write(f"Global Pedal Value MAE: {global_p_mae}\n")
        f.write(f"Global Pedal Value MSE: {global_p_mse}\n")
        f.write(f"Global Pedal Value F1 Score: {global_p_f1}\n")
        f.write("Classification Report:\n")
        f.write(cr)

    # # confusion matrix
    # cm = confusion_matrix(quantized_all_global_p_labels, quantized_all_global_p_preds)
    # plt.figure(figsize=(6, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix")
    # plt.tight_layout()
    # # save the plot
    # plt.savefig(f"global_{len(inf_label_bin_edges)-1}{ckpt_name}.png")
    # plt.close()

    plot_pedal_pred(
        all_global_p_labels,
        all_global_p_preds,
        f"{result_dir}/pedal_pred-global_{ckpt_name}.png",
        img_num_frames=800,
    )
    plot_pedal_pred(
        all_p_v_labels,
        all_p_v_preds,
        f"{result_dir}/pedal_pred-p_v_{ckpt_name}.png",
        img_num_frames=800,
    )
    plot_pedal_pred(
        all_p_onset_labels,
        all_p_onset_preds,
        f"{result_dir}/pedal_pred-p_on_{ckpt_name}.png",
        img_num_frames=800,
    )
    plot_pedal_pred(
        all_p_offset_labels,
        all_p_offset_preds,
        f"{result_dir}/pedal_pred-p_off_{ckpt_name}.png",
        img_num_frames=800,
    )


if __name__ == "__main__":
    main()
