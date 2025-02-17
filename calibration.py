import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from skimage.exposure import match_histograms


def histogram_match(y_synth, y_real, save_output=True):
    """
    Matches the histogram of the synthetic predictions to the real predictions.

    Args:
        y_synth (numpy.ndarray): Synthetic predictions.
        y_real (numpy.ndarray): Real predictions.
        save_output (bool): Whether to save the mapped output as .npy.

    Returns:
        y_real_mapped (numpy.ndarray): Mapped real predictions.
    """

    # Apply histogram matching
    y_real_mapped = match_histograms(y_real, y_synth)

    # Save the mapped values if required
    if save_output:
        np.save("y_synth_mapped.npy", y_real_mapped)
        print("Saved mapped real audio predictions as y_real_mapped.npy")

    return y_real_mapped


def plot_histogram(preds_real, preds_synth, label_bin_edges):
    # quantize
    quantized_preds_real = np.digitize(preds_real, label_bin_edges) - 1
    quantized_preds_synth = np.digitize(preds_synth, label_bin_edges) - 1

    # For real audio and synthetic audio, plot the histogram of the class count
    class_count_real = np.zeros(len(label_bin_edges) - 1)
    class_count_synth = np.zeros(len(label_bin_edges) - 1)
    for i in range(len(label_bin_edges) - 1):
        class_count_real[i] = np.sum(quantized_preds_real == i)
        class_count_synth[i] = np.sum(quantized_preds_synth == i)
    # print("Class count in real audio:", class_count_real)
    # print("Class count in synthetic audio:", class_count_synth)
    # print("Class count in real audio (normalized):", class_count_real / np.sum(class_count_real))
    # print("Class count in synthetic audio (normalized):", class_count_synth / np.sum(class_count_synth))

    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    titles = ["Real audio", "Synthetic audio"]
    for i, class_count in enumerate([class_count_real, class_count_synth]):
        axs[i].bar(range(len(label_bin_edges) - 1), class_count)
        axs[i].set_title("Class count in " + titles[i])
        axs[i].set_xlabel("Class")
        axs[i].set_ylabel("Count")
    plt.tight_layout()

    # save figure
    plt.savefig("histogram.png")


def main():

    labels_real = "result-for-calibration/p_v_labels_val_set_real.npy"
    labels_synth = "result-for-calibration/p_v_labels_val_set_synth.npy"
    preds_real = "result-for-calibration/p_v_preds_val_set_real.npy"
    preds_synth = "result-for-calibration/p_v_preds_val_set_synth.npy"

    labels_real = np.load(labels_real) # (seg_num, 120)
    labels_synth = np.load(labels_synth) # (seg_num, 120)
    preds_real = np.load(preds_real) # (seg_num, 120)
    preds_synth = np.load(preds_synth) # (seg_num, 120)

    print(labels_real.shape, preds_real.shape)
    print(labels_synth.shape, preds_synth.shape)

    mapped_preds_real = histogram_match(preds_synth, preds_real, save_output=True)
    print(mapped_preds_real.shape)
    plot_histogram(mapped_preds_real, preds_synth, label_bin_edges=range(129))

    labels_real = labels_real.flatten()
    labels_synth = labels_synth.flatten()
    preds_real = preds_real.flatten()
    preds_synth = preds_synth.flatten()
    mapped_preds_real = mapped_preds_real.flatten()

    # calculate MSE and MAE
    mse_real = mean_squared_error(labels_real, preds_real)
    mae_real = mean_absolute_error(labels_real, preds_real)
    mse_synth = mean_squared_error(labels_synth, preds_synth)
    mae_synth = mean_absolute_error(labels_synth, preds_synth)
    mse_mapped = mean_squared_error(labels_real, mapped_preds_real)
    mae_mapped = mean_absolute_error(labels_real, mapped_preds_real)
    print("MSE for real audio:", mse_real)
    print("MAE for real audio:", mae_real)
    print("MSE for synthetic audio:", mse_synth)
    print("MAE for synthetic audio:", mae_synth)
    print("MSE for mapped real audio:", mse_mapped)
    print("MAE for mapped real audio:", mae_mapped)

    # plot the histogram
    label_bin_edges = [0, 11, 128]
    # quantize the labels
    quantized_labels_real = np.digitize(labels_real * 127., label_bin_edges) - 1
    quantized_labels_synth = np.digitize(labels_synth * 127., label_bin_edges) - 1
    quantized_preds_real = np.digitize(preds_real * 127., label_bin_edges) - 1
    quantized_preds_synth = np.digitize(preds_synth * 127., label_bin_edges) - 1
    quantized_mapped_preds_real = np.digitize(mapped_preds_real * 127., label_bin_edges) - 1

    # confusion matrix
    print("Confusion matrix for real audio:")
    print(confusion_matrix(quantized_labels_real, quantized_preds_real))
    print("Confusion matrix for synthetic audio:")
    print(confusion_matrix(quantized_labels_synth, quantized_preds_synth))
    print("Confusion matrix for mapped real audio:")
    print(confusion_matrix(quantized_labels_real, quantized_mapped_preds_real))

    # classification report
    print("Classification report for real audio:")
    print(classification_report(quantized_labels_real, quantized_preds_real))
    print("Classification report for synthetic audio:")
    print(classification_report(quantized_labels_synth, quantized_preds_synth))
    print("Classification report for mapped real audio:")
    print(classification_report(quantized_labels_real, quantized_mapped_preds_real))


if __name__ == "__main__":
    main()
