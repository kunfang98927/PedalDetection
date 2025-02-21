import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix, f1_score, mean_squared_error
from src.utils import plot_pedal_pred


def calculate_classification_metrics(label_bin_edges, all_labels, all_preds):

    print("\nLabel bin edges:", label_bin_edges)

    quantized_low_res_p_labels = np.digitize(all_labels * 127, label_bin_edges) - 1
    quantized_low_res_p_preds = np.digitize(all_preds * 127, label_bin_edges) - 1

    print("F1:", f1_score(quantized_low_res_p_labels, quantized_low_res_p_preds, average="weighted"))
    print(classification_report(quantized_low_res_p_labels, quantized_low_res_p_preds))

    cm = confusion_matrix(quantized_low_res_p_labels, quantized_low_res_p_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    # save the plot
    plt.savefig(f"low_res_{len(label_bin_edges)-1}.png")
    plt.close()

def main():

    all_labels = np.load("p_v_labels_test_set_real-ckpt360-mixdata-cntrstloss-mf100-sbatch.npy")
    all_preds = np.load("p_v_preds_test_set_real-ckpt360-mixdata-cntrstloss-mf100-sbatch.npy")

    all_labels = all_labels.flatten()
    all_preds = all_preds.flatten()

    print("MAE:", mean_absolute_error(all_labels, all_preds))
    print("MSE:", mean_squared_error(all_labels, all_preds))

    calculate_classification_metrics([0, 11, 128], all_labels, all_preds)
    calculate_classification_metrics([0, 11, 95, 128], all_labels, all_preds)
    # calculate_classification_metrics([0, 11, 53, 95, 128], all_labels, all_preds)
    # calculate_classification_metrics([0, 11, 32, 53, 74, 95, 128], all_labels, all_preds)
    # calculate_classification_metrics([0, 11, 21, 32, 42, 53, 63, 74, 84, 95, 106, 117, 128], all_labels, all_preds)

    calculate_classification_metrics([0, 64, 128], all_labels, all_preds)
    calculate_classification_metrics([0, 32, 64, 96, 128], all_labels, all_preds)
    # calculate_classification_metrics([0, 16, 32, 48, 64, 80, 96, 112, 128], all_labels, all_preds)
    # calculate_classification_metrics([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128], all_labels, all_preds)

    plot_pedal_pred(all_labels, all_preds, "p_v", img_num_frames=800)

if __name__ == "__main__":
    main()



