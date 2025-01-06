import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.model import PedalDetectionModel
from src.utils import prepare_data, visualize_attention, visualize_clusters


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


def infer(model, spectrogram, device="cpu"):
    spectrogram = spectrogram.to(device)
    with torch.no_grad():
        outputs, latent_repr = model(spectrogram)
        predictions = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
    return predictions.squeeze(0).cpu().numpy(), latent_repr


def main():
    # Parameters
    checkpoint_path = "checkpoints-12/model_epoch_130_val_loss_0.3269_val_acc_0.8328.pt"
    feature_dim = 64
    hidden_dim = 256
    num_heads = 2
    ff_dim = 256
    num_layers = 4
    num_classes = 3
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

    # Prepare data
    spectrograms, labels, _ = prepare_data(data_samples=1)

    # Perform inference
    label_ratio = 0.5
    accs = []
    latent_reprs = []
    predictions = []
    label_regions = []
    for spectrogram, label in zip(spectrograms, labels):
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        prediction, latent_repr = infer(model, spectrogram, device=device)

        # only get the middle region of the label
        seq_len = label.shape[0]
        label_start = int((1 - label_ratio) / 2 * seq_len)
        label_end = int((1 + label_ratio) / 2 * seq_len)
        label = label[label_start:label_end]

        prediction = prediction[label_start:label_end]
        latent_repr = latent_repr[:, label_start:label_end]

        acc = (prediction == label).sum() / len(label)
        accs.append(acc)
        latent_reprs.append(latent_repr)
        predictions.append(prediction)
        label_regions.append(label)

    # visualize attention weights
    visualize_attention(
        model,
        num_layers,
        num_heads,
        save_path="inference_results/attention_plot_epoch130.png",
    )

    # visualize clusters
    latent_reprs = torch.cat(latent_reprs, dim=1).squeeze(0)
    predictions = np.concatenate(np.array(predictions), axis=0)
    label_regions = np.concatenate(np.array(label_regions), axis=0)
    visualize_clusters(
        latent_reprs,
        predictions,
        label_regions,
        save_path="inference_results/clusters_epoch130.png",
    )

    print("Average Accuracy:", sum(accs) / len(accs))
    print(
        classification_report(label_regions, predictions, target_names=["0", "1", "2"])
    )

    # save confusion matrix
    cm = confusion_matrix(label_regions, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, square=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("inference_results/confusion_matrix_epoch130.png")


if __name__ == "__main__":
    main()
