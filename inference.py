import torch
from src.model import PedalDetectionModel
from src.utils import prepare_data


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
        outputs, latent_repr, attn_weights_list = model(spectrogram)
        predictions = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
    return predictions.squeeze(0).cpu().numpy(), latent_repr, attn_weights_list


def main():
    # Parameters
    checkpoint_path = "checkpoints-2/model_epoch_80_val_loss_0.7233_val_acc_0.9340.pt"
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
    spectrograms, labels, _ = prepare_data()

    # Perform inference
    accs = []
    for spectrogram, label in zip(spectrograms, labels):
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        predictions, _, _ = infer(model, spectrogram, device=device)
        acc = (predictions == label).sum() / len(label)
        accs.append(acc)

    print("Average Accuracy:", sum(accs) / len(accs))


if __name__ == "__main__":
    main()
