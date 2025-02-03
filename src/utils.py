import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def load_data(data_path):
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
    features = data["features"]
    labels = data["labels"]
    metadata = data["metadata"]
    return features, labels, metadata


def split_data(features, labels, metadata, val_size=0.1, test_size=0.1, random_state=42):
    """
    Split the data into training and validation sets according to piece ids.
    """
    piece_ids = metadata[:, 1]
    unique_piece_ids = np.unique(piece_ids)

    np.random.seed(random_state)
    val_piece_ids = np.random.choice(unique_piece_ids, size=int(val_size * len(unique_piece_ids)), replace=False)
    test_piece_ids = np.random.choice(
        np.setdiff1d(unique_piece_ids, val_piece_ids), size=int(test_size * len(unique_piece_ids)), replace=False
    )
    train_piece_ids = np.setdiff1d(unique_piece_ids, np.concatenate([val_piece_ids, test_piece_ids]))
    print(f"Train Piece IDs: {train_piece_ids}", f"Val Piece IDs: {val_piece_ids}", f"Test Piece IDs: {test_piece_ids}")

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

    return train_features, val_features, test_features, \
        train_labels, val_labels, test_labels, train_metadata, val_metadata, test_metadata


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

    fig, axs = plt.subplots(num_heads, num_layers, figsize=(20, 10))
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