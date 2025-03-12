import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
)
from src.utils import plot_pedal_pred


def plot_all_pred(
    p_v_labels,
    p_v_preds,
    p_on_labels,
    p_on_preds,
    p_off_labels,
    p_off_preds,
    title,
    img_num_frames=100,
    img_num_plots=2,
):

    # plot the ground truth and prediction: all_low_res_p_labels, all_low_res_p_preds, in the same plot
    plt.figure(figsize=(20, 9))

    for i in range(0, img_num_plots * 3, 3):

        start_frame = np.random.randint(0, len(p_v_labels) - img_num_frames)
        end_frame = start_frame + img_num_frames

        # p_v
        plt.subplot(img_num_plots * 3, 1, i + 1)
        plt.plot(
            p_v_labels[start_frame:end_frame],
            label="Ground Truth",
            alpha=0.8,
            linestyle="dashed",
        )
        plt.plot(p_v_preds[start_frame:end_frame], label="Predictions", alpha=0.8)
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Value")
        plt.legend()

        # p_on
        plt.subplot(img_num_plots * 3, 1, i + 2)
        plt.plot(
            p_on_labels[start_frame:end_frame],
            label="Ground Truth",
            alpha=0.8,
            linestyle="dashed",
        )
        plt.plot(p_on_preds[start_frame:end_frame], label="Predictions", alpha=0.8)
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Onset")
        plt.legend()

        # p_off
        plt.subplot(img_num_plots * 3, 1, i + 3)
        plt.plot(
            p_off_labels[start_frame:end_frame],
            label="Ground Truth",
            alpha=0.8,
            linestyle="dashed",
        )
        plt.plot(p_off_preds[start_frame:end_frame], label="Predictions", alpha=0.8)
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Offset")
        plt.legend()

    plt.tight_layout()
    # save the plot
    plt.savefig(f"pedal_pred-{title}.png")
    plt.close()


def calculate_classification_metrics(label_bin_edges, all_labels, all_preds):

    print("\nLabel bin edges:", label_bin_edges)

    quantized_low_res_p_labels = np.digitize(all_labels * 127, label_bin_edges) - 1
    quantized_low_res_p_preds = np.digitize(all_preds * 127, label_bin_edges) - 1

    f1 = f1_score(
        quantized_low_res_p_labels, quantized_low_res_p_preds, average="weighted"
    )
    cr = classification_report(quantized_low_res_p_labels, quantized_low_res_p_preds)

    print("F1:", f1)
    print("Classification Report:")
    print(cr)

    return f1, cr


def event_f1_score(pred_events, label_events, tolerance=70):
    """
    Compute precision, recall, and F1 score for events.

    A predicted event is a true positive (TP) if it is within `tolerance` (in ms)
    of a label event. Each label event can only be matched once.

    Parameters:
      pred_events (np.array): 1D array of predicted event times (in ms).
      label_events (np.array): 1D array of ground-truth event times (in ms).
      tolerance (float): Maximum allowed time difference (in ms) to consider a match.

    Returns:
      precision (float): TP / (TP + FP)
      recall (float): TP / (TP + FN)
      f1 (float): Harmonic mean of precision and recall.
    """
    # Ensure events are sorted
    pred_events = np.sort(pred_events)
    label_events = np.sort(label_events)

    TP = 0
    matched = np.zeros(len(label_events), dtype=bool)

    for pred in pred_events:
        # Compute absolute time differences with all label events
        diffs = np.abs(label_events - pred)
        # Get the index of the closest label event
        idx = np.argmin(diffs)
        if diffs[idx] <= tolerance and not matched[idx]:
            TP += 1
            matched[idx] = True  # mark this label as matched

    FP = len(pred_events) - TP
    FN = len(label_events) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return precision, recall, f1


def calculate_result(all_labels, all_preds, title="p_v", report_path=None):
    print(f"+++++++++++++++++++++ {title} +++++++++++++++++++++")
    print(all_labels.shape, all_preds.shape)
    print("MAE:", mean_absolute_error(all_labels, all_preds))
    print("MSE:", mean_squared_error(all_labels, all_preds))

    with open(report_path, "a") as f:
        print(f"+++++++++++++++++++++ {title} +++++++++++++++++++++")
        f.write(f"MAE: {mean_absolute_error(all_labels, all_preds)}\n")
        f.write(f"MSE: {mean_squared_error(all_labels, all_preds)}\n")

    if title == "p_v":
        label_bin_edges_list = [
            [0, 11, 128],
            [0, 11, 95, 128],
            [0, 64, 128],
            [0, 32, 64, 96, 128],
        ]
        for label_bin_edges in label_bin_edges_list:
            f1, cr = calculate_classification_metrics(
                label_bin_edges, all_labels, all_preds
            )
            with open(report_path, "a") as f:
                f.write(f"Label bin edges: {label_bin_edges}\n")
                f.write(f"F1: {f1}\n")
                f.write(f"Classification Report:\n{cr}\n")

    elif title == "p_on" or title == "p_off":
        for i in range(0, len(all_preds), 500):
            seg_min = all_preds[i : i + 500].min()
            seg_max = all_preds[i : i + 500].max()
            all_preds[i : i + 500] = (all_preds[i : i + 500] - seg_min) / (
                seg_max - seg_min
            )
        all_preds_ = np.zeros_like(all_preds)
        for i in range(1, len(all_preds) - 1):
            if all_preds[i] > all_preds[i - 1] and all_preds[i] > all_preds[i + 1]:
                if all_preds[i] > 0.0:
                    all_preds_[i] = 1
        label_event_times = np.where(all_labels == 1)[0]
        pred_event_times = np.where(all_preds_ == 1)[0]
        print("pred", pred_event_times)
        print("label", label_event_times)
        precision, recall, f1 = event_f1_score(
            pred_event_times, label_event_times, tolerance=7
        )
        with open(report_path, "a") as f:
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1: {f1}\n")
    elif title == "room":
        print("room")
        print(all_labels)
        print(all_preds)
        print(confusion_matrix(all_labels, all_preds))
        print(classification_report(all_labels, all_preds))
        with open(report_path, "a") as f:
            f.write(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}\n")
            f.write(
                f"Classification Report:\n{classification_report(all_labels, all_preds)}\n"
            )


def sliding_window_normalize(a, window_size):
    """
    Normalize each element in the array 'a' using a sliding window.

    For each element, the window is centered on that element (with adjustments at the boundaries),
    and the element is normalized as:
        normalized_value = (a[i] - local_min) / (local_max - local_min)

    Parameters:
    - a: 1D NumPy array.
    - window_size: Size of the sliding window (should be an odd number for symmetry).

    Returns:
    - A new NumPy array with the normalized values.
    """
    n = len(a)
    normalized = np.zeros(n, dtype=float)
    half_window = window_size // 2

    for i in range(n):
        # Determine the window boundaries
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        local_window = a[start:end]

        local_min = local_window.min()
        local_max = local_window.max()

        # Avoid division by zero if the window has constant values.
        if local_max - local_min != 0:
            normalized[i] = (a[i] - local_min) / (local_max - local_min)
        else:
            normalized[i] = 0.0  # Alternatively, you can set this to 0.5

    return normalized


def main():

    ckpt_dir = "ckpt_0312_10per-clip-500frm_bs16-8h-2xdata-val1-6loss-resume/results"
    report_path = f"{ckpt_dir}/report.txt"

    all_p_v_labels = np.load(f"{ckpt_dir}/p_v_labels_test_set_0310.npy")
    all_p_v_preds = np.load(f"{ckpt_dir}/p_v_preds_test_set_0310.npy")

    all_on_labels = np.load(f"{ckpt_dir}/p_onset_labels_test_set_0310.npy")
    all_on_preds = np.load(f"{ckpt_dir}/p_onset_preds_test_set_0310.npy")

    all_off_labels = np.load(f"{ckpt_dir}/p_offset_labels_test_set_0310.npy")
    all_off_preds = np.load(f"{ckpt_dir}/p_offset_preds_test_set_0310.npy")

    # all_room_labels = np.load(
    #     f"{ckpt_dir}/room_labels_test_set_0310.npy"
    # )
    # all_room_preds = np.load(
    #     f"{ckpt_dir}/room_preds_test_set_0310.npy"
    # )

    all_p_v_labels = all_p_v_labels.flatten()
    all_p_v_preds = all_p_v_preds.flatten()
    all_on_labels = all_on_labels.flatten()
    all_on_preds = all_on_preds.flatten()
    all_off_labels = all_off_labels.flatten()
    all_off_preds = all_off_preds.flatten()
    # all_room_labels = all_room_labels.flatten()
    # all_room_preds = all_room_preds.flatten()

    print(all_p_v_labels.shape, all_p_v_preds.shape)
    print(all_on_labels.shape, all_on_preds.shape)
    print(all_off_labels.shape, all_off_preds.shape)
    # print(all_room_labels.shape, all_room_preds.shape)

    calculate_result(all_p_v_labels, all_p_v_preds, "p_v", report_path)
    calculate_result(all_on_labels, all_on_preds, "p_on", report_path)
    calculate_result(all_off_labels, all_off_preds, "p_off", report_path)
    # calculate_result(all_room_labels, all_room_preds, "room", report_path)

    plot_all_pred(
        all_p_v_labels,
        all_p_v_preds,
        all_on_labels,
        all_on_preds,
        all_off_labels,
        all_off_preds,
        "p_v_on_off",
        img_num_frames=2000,
    )


if __name__ == "__main__":
    main()
