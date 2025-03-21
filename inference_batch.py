import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    classification_report,
    f1_score,
    mean_squared_error,
)
from src.model2 import PedalDetectionModelwithCNN1
from src.dataset_h5 import PedalDataset
from src.utils import (
    get_label_bin_edges,
    plot_pedal_pred,
)
from calculate_metric import calculate_result, plot_all_pred

import functools

print = functools.partial(print, flush=True)


def load_model(
    checkpoint_path,
    input_dim,
    hidden_dim,
    num_heads,
    ff_dim,
    num_layers,
    num_classes,
    device="cpu",
    predict_global_pedal=True,
    predict_pedal_onset=False,
    predict_pedal_offset=False,
    predict_room=False,
):
    model = PedalDetectionModelwithCNN1(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        predict_global_pedal=predict_global_pedal,
        predict_pedal_onset=predict_pedal_onset,
        predict_pedal_offset=predict_pedal_offset,
        predict_room=predict_room,
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

    parser = argparse.ArgumentParser(description="Pedal model inference arguments")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--predict_global_pedal",
        type=bool,
        default=True,
        help="Predict global pedal value",
    )
    parser.add_argument(
        "--predict_pedal_onset",
        type=bool,
        default=True,
        help="Predict pedal onset",
    )
    parser.add_argument(
        "--predict_pedal_offset",
        type=bool,
        default=True,
        help="Predict pedal offset",
    )
    parser.add_argument(
        "--predict_room",
        type=bool,
        default=False,
        help="Predict room",
    )

    args = parser.parse_args()
    # Parameters
    checkpoint_path = args.checkpoint_path
    datasets = [args.dataset]
    predict_global_pedal = args.predict_global_pedal
    predict_pedal_onset = args.predict_pedal_onset
    predict_pedal_offset = args.predict_pedal_offset
    predict_room = args.predict_room

    # Get the name of the checkpoint
    ckpt_name = checkpoint_path.split("/")[-1]
    result_dir = checkpoint_path.split("/")[0]
    result_dir = f"{result_dir}/results"
    print(f"Result directory: {result_dir}")
    os.makedirs(result_dir, exist_ok=True)
    report_path = f"{result_dir}/report-{datasets[0]}-{ckpt_name}.txt"
    with open(report_path, "a") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"datasets: {datasets}\n")
        f.write(f"predict_global_pedal: {predict_global_pedal}\n")
        f.write(f"predict_pedal_onset: {predict_pedal_onset}\n")
        f.write(f"predict_pedal_offset: {predict_pedal_offset}\n")
        f.write(f"predict_room: {predict_room}\n")

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
        predict_global_pedal=predict_global_pedal,
        predict_pedal_onset=predict_pedal_onset,
        predict_pedal_offset=predict_pedal_offset,
        predict_room=predict_room,
    )
    # print model trainable parameters number
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )
    # write model parameters to report
    with open(report_path, "a") as f:
        f.write(
            f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.\n"
        )

    # Data loading
    test_dataset = PedalDataset(
        data_list_path="sample_data/test.json",
        data_dir="/scratch/kunfang/pedal_data/data/",
        num_samples_per_clip=1,  # num_samples_per_clip,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.0,
        split="test",
        datasets=datasets,
        # num_examples=20,
        randomly_sample=False,
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
        num_workers=16,
        pin_memory=True,
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
        loss_mask,
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
        f"{result_dir}/p_v_labels_test_set_{datasets[0]}.npy",
        all_p_v_labels,
    )
    np.save(
        f"{result_dir}/p_v_preds_test_set_{datasets[0]}.npy",
        all_p_v_preds,
    )

    # pedal onset
    all_p_onset_labels = np.concatenate(all_p_onset_labels)
    all_p_onset_preds = np.concatenate(all_p_onset_preds)
    print(all_p_onset_labels.shape, all_p_onset_preds.shape)
    np.save(
        f"{result_dir}/p_onset_labels_test_set_{datasets[0]}.npy",
        all_p_onset_labels,
    )
    np.save(
        f"{result_dir}/p_onset_preds_test_set_{datasets[0]}.npy",
        all_p_onset_preds,
    )

    # pedal offset
    all_p_offset_labels = np.concatenate(all_p_offset_labels)
    all_p_offset_preds = np.concatenate(all_p_offset_preds)
    print(all_p_offset_labels.shape, all_p_offset_preds.shape)
    np.save(
        f"{result_dir}/p_offset_labels_test_set_{datasets[0]}.npy",
        all_p_offset_labels,
    )
    np.save(
        f"{result_dir}/p_offset_preds_test_set_{datasets[0]}.npy",
        all_p_offset_preds,
    )

    # global pedal value
    all_global_p_labels = np.concatenate(all_global_p_labels)
    all_global_p_preds = np.concatenate(all_global_p_preds)
    print(all_global_p_labels.shape, all_global_p_preds.shape)
    np.save(
        f"{result_dir}/global_p_labels_test_set_{datasets[0]}.npy",
        all_global_p_labels,
    )
    np.save(
        f"{result_dir}/global_p_preds_test_set_{datasets[0]}.npy",
        all_global_p_preds,
    )

    # room
    all_room_labels = np.concatenate(all_room_labels)
    all_room_preds = np.concatenate(all_room_preds)
    print(all_room_labels.shape, all_room_preds.shape)
    np.save(
        f"{result_dir}/room_labels_test_set_{datasets[0]}.npy",
        all_room_labels,
    )
    np.save(
        f"{result_dir}/room_preds_test_set_{datasets[0]}.npy",
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
        quantized_all_global_p_labels, quantized_all_global_p_preds, digits=4
    )
    print(cr)

    with open(report_path, "a") as f:
        f.write(f"Global Pedal Value MAE: {global_p_mae}\n")
        f.write(f"Global Pedal Value MSE: {global_p_mse}\n")
        f.write(f"Global Pedal Value F1 Score: {global_p_f1}\n")
        f.write("Classification Report:\n")
        f.write(cr)

    plot_pedal_pred(
        all_global_p_labels,
        all_global_p_preds,
        f"{result_dir}/pedal_pred-global_{ckpt_name}.png",
        img_num_frames=1000,
    )
    plot_pedal_pred(
        all_p_v_labels,
        all_p_v_preds,
        f"{result_dir}/pedal_pred-p_v_{ckpt_name}.png",
        img_num_frames=1000,
    )
    plot_pedal_pred(
        all_p_onset_labels,
        all_p_onset_preds,
        f"{result_dir}/pedal_pred-p_on_{ckpt_name}.png",
        img_num_frames=1000,
    )
    plot_pedal_pred(
        all_p_offset_labels,
        all_p_offset_preds,
        f"{result_dir}/pedal_pred-p_off_{ckpt_name}.png",
        img_num_frames=1000,
    )

    # Calculate evaluation metrics for pedal value, pedal onset, and pedal offset
    all_p_v_labels = all_p_v_labels.flatten()
    all_p_v_preds = all_p_v_preds.flatten()
    all_p_onset_labels = all_p_onset_labels.flatten()
    all_p_onset_preds = all_p_onset_preds.flatten()
    all_p_offset_labels = all_p_offset_labels.flatten()
    all_p_offset_preds = all_p_offset_preds.flatten()

    calculate_result(all_p_v_labels, all_p_v_preds, "p_v", report_path)
    calculate_result(all_p_onset_labels, all_p_onset_preds, "p_on", report_path)
    calculate_result(all_p_offset_labels, all_p_offset_preds, "p_off", report_path)

    plot_all_pred(
        all_p_v_labels,
        all_p_v_preds,
        all_p_onset_labels,
        all_p_onset_preds,
        all_p_offset_labels,
        all_p_offset_preds,
        "p_v_on_off",
        img_num_frames=1000,
        save_dir=result_dir,
    )

if __name__ == "__main__":
    main()
