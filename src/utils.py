import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


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


def prepare_data(
    seq_len=1000,
    feature_dim=64,
    data_samples=64,
    max_acoustic_setting_value=5,
    acoustic_setting_dim=3,
):
    # Sample data (replace with actual data loading logic)
    spectrograms = [np.random.rand(seq_len, feature_dim) for _ in range(data_samples)]
    acoustic_settings = [
        np.random.randint(0, max_acoustic_setting_value, size=acoustic_setting_dim)
        for _ in range(data_samples)
    ]  # Shape: [acoustic_setting_dim]

    labels = []
    for spectrogram in spectrograms:
        label = np.zeros(seq_len)
        for i in range(seq_len):
            start_id = max(0, i - 2)
            end_id = min(seq_len, i + 2)
            mean_frame = np.mean(spectrogram[start_id:end_id])
            right_eps = 0.020
            left_eps = 0.002
            if mean_frame > 0.5 + right_eps:
                label[i] = 2
            elif mean_frame < 0.5 - left_eps:
                label[i] = 0
            else:
                label[i] = 1
        labels.append(label)

    # count the number of each class
    class_counts = {0: 0, 1: 0, 2: 0}
    for label in labels:
        for i in range(seq_len):
            class_counts[label[i]] += 1
    print(class_counts)

    return spectrograms, labels, acoustic_settings
