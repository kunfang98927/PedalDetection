import torch
from torch.utils.data import Dataset
import numpy as np


class PedalDataset(Dataset):
    def __init__(
        self,
        spectrograms,
        labels,
        acoustic_settings,
        augmentations=None,
        label_ratio=0.5,
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.acoustic_settings = acoustic_settings
        self.augmentations = augmentations
        self.label_ratio = label_ratio

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]  # Shape: [seq_len, feature_dim]
        label = self.labels[idx]  # Shape: [seq_len]
        acoustic_setting = self.acoustic_settings[idx]

        if self.augmentations:
            spectrogram = self.augmentations(spectrogram)

        seq_len = label.shape[0]
        label_start = int((1 - self.label_ratio) / 2 * seq_len)
        label_end = int((1 + self.label_ratio) / 2 * seq_len)
        label_masked = np.full(label.shape, -1)
        label_masked[label_start:label_end] = label[label_start:label_end]

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label_masked = torch.tensor(label_masked, dtype=torch.long)
        acoustic_setting = torch.tensor(acoustic_setting, dtype=torch.float32)

        return spectrogram, label_masked, acoustic_setting
