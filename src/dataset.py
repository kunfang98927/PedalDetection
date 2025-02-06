import torch
from torch.utils.data import Dataset
import numpy as np


class PedalDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
        metadata,
        num_samples_per_clip=10,
        max_frame=500, # about 86.13 fps
        label_ratio=0.6,
    ):
        self.features = features
        self.labels = labels
        self.metadata = metadata
        self.num_samples_per_clip = num_samples_per_clip
        self.max_frame = max_frame
        self.label_ratio = label_ratio

    def __len__(self):
        return len(self.features) * self.num_samples_per_clip

    def __getitem__(self, idx):
        feat_idx = idx // self.num_samples_per_clip # Index of the feature
        seg_idx = idx % self.num_samples_per_clip # Index of the segment

        feature = self.features[feat_idx]  # Shape: [feature_dim, seq_len]
        label = self.labels[feat_idx]  # Shape: [seq_len]
        metadata = self.metadata[feat_idx] # Shape: [num_metadata]

        label = np.where(label <= 10, 0, label)
        label = np.where(label >= 100, 2, label)
        label = np.where((label > 10) & (label < 100), 1, label)

        synth_setting = metadata[0] - 1.0 # 1: dry room no reverb; 2: clean studio moderate reverb; 3: large concert hall max reverb
        piece_id = metadata[1] # piece id
        midi_id = metadata[2] # midi id

        # Randomly select self.max_frame frames from the sequence
        start_frame = np.random.randint(0, feature.shape[1] - self.max_frame)
        selected_feature = feature[:, start_frame:start_frame + self.max_frame].T
        selected_label = label[start_frame:start_frame + self.max_frame]

        # Mask beginning and end of the sequence, and keep the middle part for training
        label_start = int((1 - self.label_ratio) / 2 * self.max_frame)
        label_end = int((1 + self.label_ratio) / 2 * self.max_frame)
        label_masked = np.full(selected_label.shape, -1)
        label_masked[label_start:label_end] = selected_label[label_start:label_end]

        selected_feature = torch.tensor(selected_feature, dtype=torch.float32)
        label_masked = torch.tensor(label_masked, dtype=torch.long)

        return selected_feature, label_masked, synth_setting #, piece_id, midi_id
