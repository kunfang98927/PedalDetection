import numpy as np


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
            mean_frame = np.mean(spectrogram[i])
            if mean_frame > 0.53:
                label[i] = 2
            elif mean_frame < 0.47:
                label[i] = 0
            else:
                label[i] = 1
        labels.append(label)

    return spectrograms, labels, acoustic_settings
