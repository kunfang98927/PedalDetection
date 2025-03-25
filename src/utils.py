import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_pedal_pred(labels, preds, save_path, img_num_frames=100, img_num_plots=6):

    # plot the ground truth and prediction: all_low_res_p_labels, all_low_res_p_preds, in the same plot
    plt.figure(figsize=(20, 9))

    for i in range(img_num_plots):
        plt.subplot(img_num_plots, 1, i + 1)
        # randomly select a start frame
        start_frame = np.random.randint(0, len(labels) - img_num_frames)
        end_frame = start_frame + img_num_frames
        plt.plot(
            labels[start_frame:end_frame],
            label="Ground Truth",
            alpha=0.8,
            linestyle="dashed",
        )
        plt.plot(preds[start_frame:end_frame], label="Predictions", alpha=0.8)
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Value")
        plt.legend()
    plt.tight_layout()
    # save the plot
    plt.savefig(save_path)
    plt.close()


def calculate_pedal_onset_offset(quantized_pedal_value, on_off_threshold=64):
    """
    Calculate the pedal onset and offset from the pedal value.

    Args:
        pedal_value (numpy.ndarray): Pedal value of shape (seq_len, ).

    Returns:
        pedal_onset (numpy.ndarray): Pedal onset of shape (seq_len, ).
            pedal_onset[i] = 1 if pedal_value[i] > pedal_value[i-1] and pedal_value[i-1] == 0
        pedal_offset (numpy.ndarray): Pedal offset of shape (seq_len, ).
            pedal_offset[i] = 1 if pedal_value[i] < pedal_value[i-1] and pedal_value[i] == 0
    """
    pedal_onset = np.zeros_like(quantized_pedal_value)
    pedal_offset = np.zeros_like(quantized_pedal_value)
    for i in range(1, len(quantized_pedal_value)):
        if (
            quantized_pedal_value[i] > quantized_pedal_value[i - 1]
            and quantized_pedal_value[i - 1] <= on_off_threshold
            and quantized_pedal_value[i] > on_off_threshold
        ):
            pedal_onset[i] = 1
        if (
            quantized_pedal_value[i] < quantized_pedal_value[i - 1]
            and quantized_pedal_value[i] <= on_off_threshold
            and quantized_pedal_value[i - 1] > on_off_threshold
        ):
            pedal_offset[i] = 1
    return pedal_onset, pedal_offset


def calculate_soft_regresion_label(label, window=5):
    """
    Calculate the soft regression label from the pedal onset.

    Args:
        label (numpy.ndarray): Label of shape (seq_len, ).
        window (int): The window size for the soft regression label.

    Returns:
        soft_regression_label (numpy.ndarray): Soft regression label of shape (seq_len, ).
    """
    soft_regression_label = np.zeros_like(label, dtype=np.float32)
    for i, l in enumerate(label):
        if l == 1:
            for j in range(-window, window + 1):
                if 0 <= i + j < len(soft_regression_label):
                    soft_regression_label[i + j] = 1 - abs(j) / window
    return soft_regression_label


def calculate_low_res_pedal_value(
    selected_pedal_value, quantized_pedal_value, label_start, label_end, label_bin_edges
):
    """
    Calculate the low resolution pedal value from the  pedal value.

    Args:
        selected_pedal_value (numpy.ndarray): Pedal value of shape (seq_len, ).
        quantized_pedal_value (numpy.ndarray): Quantized pedal value of shape (seq_len, ).
        label_start (int): The start index of the label region.
        label_end (int): The end index of the label region.
        label_bin_edges (list): The bin edges for quantizing the pedal value.

    Returns:
        low_res_pedal_value: Mean of the quantized pedal values from label_start to label_end.
    """
    low_res_pedal_value = np.mean(selected_pedal_value[label_start:label_end])
    if len(label_bin_edges) == 2:
        return low_res_pedal_value / 127.0

    # quantize the pedal value
    for i in range(len(label_bin_edges) - 1):
        if label_bin_edges[i] <= low_res_pedal_value < label_bin_edges[i + 1]:
            quantized_low_res_pedal_value = i
            break

    # statistics of quantized pedal values in each bin, for example, if bin_edges = [0, 11, 95, 128]
    # the quantized pedal values in each bin are [0, 1, 2]
    # then calculate the frequency of each quantized pedal value in each bin
    # and return [0.7, 0.2, 0.1] as the low_res_soft_pedal_value
    low_res_soft_pedal_value = np.zeros(len(label_bin_edges) - 1)
    for i in range(len(label_bin_edges) - 1):
        low_res_soft_pedal_value[i] = np.mean(
            quantized_pedal_value[label_start:label_end] == i
        )
    # print("\n", low_res_soft_pedal_value)
    # print(quantized_low_res_pedal_value, low_res_pedal_value, selected_pedal_value[label_start:label_end])

    return quantized_low_res_pedal_value  # , low_res_soft_pedal_value


def get_label_bin_edges(num_classes):
    label_bins = {
        1: [0, 128],
        2: [0, 64, 128],
        3: [0, 11, 95, 128],
        4: [0, 11, 60, 95, 128],
        128: list(range(129)),
    }
    return label_bins.get(num_classes, None)


def load_data(data_path, label_bin_edges, pedal_factor, room_acoustics):
    """
    Load the processed data from the given path.

    Args:
        data_path (str): Path to the processed data file.

    Returns:
        features (numpy.ndarray): Spectrogram features of shape (num_samples, seq_len, feature_dim).
        labels (numpy.ndarray): Ground truth labels of shape (num_samples, seq_len).
        metadata (dict): Metadata dictionary.
    """
    data = np.load(data_path, allow_pickle=True)
    print("Data keys:", data.files)
    features = data["features"]
    average_labels = data["average_labels"]
    instant_values = data["instant_values"]
    metadata = data["metadata"]

    print("Features shape:", features.shape, features[0].shape)
    print(
        "Average labels shape:",
        average_labels.shape,
        average_labels[0].shape,
        average_labels[0][:, 0],
        average_labels[0][:, 1],
        average_labels[0][:, 2],
    )
    print(
        "Instant values shape:",
        instant_values.shape,
        instant_values[0].shape,
        instant_values[0][:, 0],
        instant_values[0][:, 1],
        instant_values[0][:, 2],
    )
    print(
        "Metadata shape:",
        metadata.shape,
        metadata[0],
        metadata[1],
        metadata[2],
        metadata[-2],
        metadata[-1],
    )

    # pedal_labels: shape: (num_samples, ); pedal_labels[0].shape: (seq_len, )
    pedal_labels = np.ndarray((len(instant_values)), dtype=object)
    for i, label in enumerate(instant_values):
        pedal_labels[i] = label[1]

    # only keep the data with metadata[2] in [-1, 0.5, 1.]
    pedal_factor_mask = np.isin(metadata[:, 2], pedal_factor)
    features = features[pedal_factor_mask]
    pedal_labels = pedal_labels[pedal_factor_mask]
    metadata = metadata[pedal_factor_mask]

    # only keep the data with metadata[0] in [1., 2., 3.] (room acoustics)
    room_acoustics_mask = np.isin(metadata[:, 0], room_acoustics)
    features = features[room_acoustics_mask]
    pedal_labels = pedal_labels[room_acoustics_mask]
    metadata = metadata[room_acoustics_mask]

    print("Features shape:", features.shape)
    print("pedal_labels shape:", pedal_labels.shape, pedal_labels[0].shape)
    print("Metadata shape:", metadata.shape)

    # calculate the number of samples for each class
    class_count = np.zeros(len(label_bin_edges) - 1)
    for i, label in enumerate(pedal_labels):
        unique, counts = np.unique(label, return_counts=True)
        for j, c in zip(unique, counts):
            for k in range(len(label_bin_edges) - 1):
                if label_bin_edges[k] <= j < label_bin_edges[k + 1]:
                    class_count[k] += c
    # calculate the percentage of each class
    class_count = class_count / np.sum(class_count)
    print("Class count:", class_count)

    return features, pedal_labels, metadata


def load_data_real_audio(data_path, label_bin_edges):
    """
    Load the processed data from the given path.

    Args:
        data_path (str): Path to the processed data file.

    Returns:
        features (numpy.ndarray): Spectrogram features of shape (num_samples, seq_len, feature_dim).
        labels (numpy.ndarray): Ground truth labels of shape (num_samples, seq_len).
        metadata (dict): Metadata dictionary.
    """
    data = np.load(data_path, allow_pickle=True)
    print("Data keys:", data.files)
    features = data["features"]
    average_labels = data["average_labels"]
    instant_values = data["instant_values"]
    metadata = data["metadata"]

    print("Features shape:", features.shape, features[0].shape)
    print(
        "Average labels shape:",
        average_labels.shape,
        average_labels[0].shape,
        average_labels[0][:, 0],
        average_labels[0][:, 1],
        average_labels[0][:, 2],
    )
    print(
        "Instant values shape:",
        instant_values.shape,
        instant_values[0].shape,
        instant_values[0][:, 0],
        instant_values[0][:, 1],
        instant_values[0][:, 2],
    )
    print("Metadata shape:", metadata.shape, metadata[0])

    # pedal_labels: shape: (num_samples, ); pedal_labels[0].shape: (seq_len, )
    pedal_labels = np.ndarray((len(instant_values)), dtype=object)
    for i, label in enumerate(instant_values):
        pedal_labels[i] = label[1]

    print("Features shape:", features.shape)
    print("pedal_labels shape:", pedal_labels.shape, pedal_labels[0].shape)
    print("Metadata shape:", metadata.shape)

    # calculate the number of samples for each class
    class_count = np.zeros(len(label_bin_edges) - 1)
    for i, label in enumerate(pedal_labels):
        unique, counts = np.unique(label, return_counts=True)
        for j, c in zip(unique, counts):
            for k in range(len(label_bin_edges) - 1):
                if label_bin_edges[k] <= j < label_bin_edges[k + 1]:
                    class_count[k] += c
    # calculate the percentage of each class
    class_count = class_count / np.sum(class_count)
    print("Class count:", class_count)

    return features, pedal_labels, metadata


def split_data(
    features, labels, metadata, val_size=0.1, test_size=0.1, random_state=42
):
    """
    Split the data into training and validation sets according to piece ids.
    """
    piece_ids = metadata[:, 1]
    unique_piece_ids = np.unique(piece_ids)

    np.random.seed(random_state)
    val_piece_ids = np.random.choice(
        unique_piece_ids, size=int(val_size * len(unique_piece_ids)), replace=False
    )
    test_piece_ids = np.random.choice(
        np.setdiff1d(unique_piece_ids, val_piece_ids),
        size=int(test_size * len(unique_piece_ids)),
        replace=False,
    )
    train_piece_ids = np.setdiff1d(
        unique_piece_ids, np.concatenate([val_piece_ids, test_piece_ids])
    )
    print(
        f"Train Piece IDs: {train_piece_ids}",
        f"Val Piece IDs: {val_piece_ids}",
        f"Test Piece IDs: {test_piece_ids}",
    )

    train_indices = np.where(np.isin(piece_ids, train_piece_ids))[0]
    val_indices = np.where(np.isin(piece_ids, val_piece_ids))[0]
    test_indices = np.where(np.isin(piece_ids, test_piece_ids))[0]

    train_features = features[train_indices]
    val_features = features[val_indices]
    test_features = features[test_indices]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    train_metadata = metadata[train_indices]
    val_metadata = metadata[val_indices]
    test_metadata = metadata[test_indices]

    return (
        train_features,
        val_features,
        test_features,
        train_labels,
        val_labels,
        test_labels,
        train_metadata,
        val_metadata,
        test_metadata,
    )


def split_data_real_audio(
    features, labels, metadata, split="test", max_num_samples=None
):
    """
    Split the data into training and validation sets according to piece ids.
    """
    split2id = {"train": 0, "val": 1, "test": 2}
    split_id = metadata[:, -1]  # train: 0, val: 1, test: 2
    # filter the data according to the split
    split_mask = split_id == split2id[split]
    split_features = features[split_mask]
    split_labels = labels[split_mask]
    split_metadata = metadata[split_mask]

    # randomly select max_num_samples samples
    if max_num_samples is not None and max_num_samples < len(split_features):
        indices = np.random.choice(len(split_features), max_num_samples, replace=False)
        split_features = split_features[indices]
        split_labels = split_labels[indices]
        split_metadata = split_metadata[indices]

    return split_features, split_labels, split_metadata


def visualize_clusters(latent_repr, predictions, annotations, save_path=None):
    """
    Visualize the latent representation with classification predictions and annotations.

    Args:
        latent_repr (numpy.ndarray): Latent representation of shape (seq_len, feature_dim).
        predictions (numpy.ndarray): Classification predictions for each frame (seq_len,).
        annotations (numpy.ndarray): Ground truth annotations for each frame (seq_len,).
    """

    # Step 1: Dimensionality Reduction
    pca = PCA(n_components=2)
    reduced_latent_repr = pca.fit_transform(latent_repr)

    # Step 2: Plot predicted clusters
    plt.figure(figsize=(15, 5))

    # Plot predictions
    plt.subplot(1, 2, 1)
    for label in np.unique(annotations):
        points = reduced_latent_repr[annotations == label]
        plt.scatter(points[:, 0], points[:, 1], label=f"label={label}", s=2, alpha=0.5)
    plt.title("Predicted Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    # Step 3: Highlight misclassifications
    plt.subplot(1, 2, 2)
    misclassified = predictions != annotations
    if np.any(misclassified):
        plt.scatter(
            reduced_latent_repr[:, 0],
            reduced_latent_repr[:, 1],
            c="gray",
            alpha=0.5,
            label="Correctly Classified",
            s=2,
        )
        plt.scatter(
            reduced_latent_repr[misclassified, 0],
            reduced_latent_repr[misclassified, 1],
            c="red",
            label="Misclassified",
            alpha=0.6,
            s=2,
        )
        plt.title("Misclassified Frames Highlighted")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved cluster plot to {save_path}")
    else:
        plt.savefig("cluster_plot.png")
        print("Saved cluster plot to cluster_plot.png")

    plt.close()


def draw(data, ax):
    sns.heatmap(data, ax=ax, cmap="viridis", cbar=False, square=True)


def visualize_attention(model, num_layers, num_heads, save_path=None):
    """
    Visualize the attention weights as a heatmap.
    """

    fig, axs = plt.subplots(num_heads, num_layers, figsize=(20, 20))
    for layer in range(num_layers):
        for h in range(num_heads):
            draw(model.layers[layer].self_attn.attn[0, h].data, ax=axs[h, layer])
            axs[h, layer].set_title(f"Layer {layer + 1} - Head {h + 1}")
            axs[h, layer].set_xlabel("Key Tokens")
            axs[h, layer].set_ylabel("Query Tokens")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved attention plot to {save_path}")
    else:
        plt.savefig("attention_plot.png")
        print("Saved attention plot to attention_plot.png")

        plt.close()
