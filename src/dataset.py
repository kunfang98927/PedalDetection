import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils import (
    calculate_pedal_onset_offset,
    calculate_soft_regresion_label,
    calculate_low_res_pedal_value,
)


def quantize_pedal_value(selected_pedal_value):
    quantized_pedal_value = np.zeros_like(selected_pedal_value, dtype=np.float32)

    # Apply piecewise conditions
    mask1 = selected_pedal_value < 11
    mask2 = (selected_pedal_value >= 11) & (selected_pedal_value < 25)
    mask3 = (selected_pedal_value >= 25) & (selected_pedal_value < 95)
    mask4 = selected_pedal_value >= 95

    quantized_pedal_value[mask1] = 0  # Selected < 11 -> 0
    quantized_pedal_value[mask2] = (selected_pedal_value[mask2] - 11) * (
        50 / (25 - 11)
    )  # Linear 0 to 50
    quantized_pedal_value[mask3] = 50 + (selected_pedal_value[mask3] - 25) * (
        77 / (95 - 25)
    )  # Linear 50 to 127
    quantized_pedal_value[mask4] = 127  # Selected >= 95 -> 127

    return quantized_pedal_value


class PedalDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
        metadata,
        num_samples_per_clip=10,
        max_frame=500,  # about 86.13 fps
        label_ratio=0.6,
        label_bin_edges=[0, 11, 95, 128],
        overlap_ratio=0.25,
        split="train",
    ):
        self.features = features
        print("self.features", len(self.features))
        self.labels = labels
        self.metadata = metadata
        self.num_samples_per_clip = num_samples_per_clip
        self.max_frame = max_frame
        self.label_ratio = label_ratio
        self.label_bin_edges = label_bin_edges
        self.overlap_ratio = overlap_ratio
        self.split = split

    def __len__(self):
        if self.split == "test" or self.split == "validation":
            return sum(
                [
                    (feature.shape[1] - self.max_frame)
                    // int(self.max_frame * (1 - self.overlap_ratio))
                    for feature in self.features
                ]
            )
        return len(self.features) * self.num_samples_per_clip

    def __getitem__(self, idx):
        if self.split == "test" or self.split == "validation":
            feat_idx = 0
            for feature in self.features:
                if idx < (feature.shape[1] - self.max_frame) // int(
                    self.max_frame * (1 - self.overlap_ratio)
                ):
                    break
                idx -= (feature.shape[1] - self.max_frame) // int(
                    self.max_frame * (1 - self.overlap_ratio)
                )
                feat_idx += 1
            seg_idx = idx
            if feat_idx >= len(self.features):
                raise IndexError
        else:
            feat_idx = idx // self.num_samples_per_clip  # Index of the feature
            seg_idx = idx % self.num_samples_per_clip  # Index of the segment

        feature = self.features[feat_idx]  # Shape: [feature_dim, seq_len]
        pedal_value = self.labels[feat_idx]  # Shape: [seq_len]
        metadata = self.metadata[feat_idx]  # Shape: [num_metadata]

        room_acoustics = (
            metadata[0] - 1.0
        )  # 1: dry room no reverb; 2: clean studio moderate reverb; 3: large concert hall max reverb
        midi_id = metadata[1]  # midi id
        pedal_factor = metadata[2] if len(metadata) > 2 else 1.0  # pedal factor

        # Randomly select self.max_frame frames from the sequence
        if self.split == "train":
            start_frame = np.random.randint(0, feature.shape[1] - self.max_frame)
            end_frame = start_frame + self.max_frame
        elif self.split == "test" or self.split == "validation":
            start_frame = int(seg_idx * (1 - self.overlap_ratio) * self.max_frame)
            end_frame = int(start_frame + self.max_frame)
            if end_frame - start_frame != self.max_frame or end_frame > len(
                pedal_value
            ):
                return self.__getitem__((idx + 1) % len(self))

        # Select the feature and label within the selected frames
        selected_feature = feature[:, start_frame:end_frame].T
        selected_pedal_value = pedal_value[start_frame:end_frame]

        if len(self.label_bin_edges) == 2:
            # quantized_pedal_value = quantize_pedal_value(selected_pedal_value)
            pedal_onset, pedal_offset = calculate_pedal_onset_offset(
                selected_pedal_value, 0
            )
            quantized_pedal_value = selected_pedal_value / 127.0
        else:
            quantized_pedal_value = (
                np.digitize(selected_pedal_value, self.label_bin_edges) - 1
            )
            pedal_onset, pedal_offset = calculate_pedal_onset_offset(
                quantized_pedal_value, on_off_threshold=self.label_bin_edges[1]
            )
        soft_pedal_onset = calculate_soft_regresion_label(pedal_onset)
        soft_pedal_offset = calculate_soft_regresion_label(pedal_offset)

        # # if "0" label is more than 30% of the sequence, skip this sample
        # if self.split == "train":
        #     if np.sum(quantized_pedal_value == 0) > self.max_frame * 0.3 or np.sum(quantized_pedal_value == 127) > self.max_frame * 0.3:
        #         return self.__getitem__((idx + 1) % len(self))

        # Mask beginning and end of the sequence, and keep the middle part for training
        label_start = int((1 - self.label_ratio) / 2 * self.max_frame)
        label_end = int((1 + self.label_ratio) / 2 * self.max_frame)
        quantized_pedal_value_masked = np.full(
            quantized_pedal_value.shape, -1, dtype=np.float32
        )
        quantized_pedal_value_masked[label_start:label_end] = quantized_pedal_value[
            label_start:label_end
        ]
        soft_pedal_onset_masked = np.full(soft_pedal_onset.shape, -1, dtype=np.float32)
        soft_pedal_onset_masked[label_start:label_end] = soft_pedal_onset[
            label_start:label_end
        ]
        soft_pedal_offset_masked = np.full(
            soft_pedal_offset.shape, -1, dtype=np.float32
        )
        soft_pedal_offset_masked[label_start:label_end] = soft_pedal_offset[
            label_start:label_end
        ]

        # print("shape", selected_pedal_value.shape, quantized_pedal_value_masked.shape, pedal_onset.shape, pedal_offset.shape, soft_pedal_onset.shape, soft_pedal_offset.shape)
        # s = np.random.randint(0, 100)
        # self.plot_all_labels(
        #     selected_pedal_value,
        #     quantized_pedal_value_masked,
        #     pedal_onset,
        #     pedal_offset,
        #     soft_pedal_onset,
        #     soft_pedal_offset,
        #     self.label_bin_edges,
        #     img_name=f"all_labels_{s}",
        # )
        # print(stop_here)

        low_res_label = calculate_low_res_pedal_value(
            selected_pedal_value,
            quantized_pedal_value,
            label_start,
            label_end,
            self.label_bin_edges,
        )

        selected_feature = torch.tensor(selected_feature, dtype=torch.float32)
        quantized_pedal_value_masked = torch.tensor(
            quantized_pedal_value_masked, dtype=torch.float32
        )
        low_res_label = torch.tensor(low_res_label, dtype=torch.float32)
        soft_pedal_onset_masked = torch.tensor(
            soft_pedal_onset_masked, dtype=torch.float32
        )
        soft_pedal_offset_masked = torch.tensor(
            soft_pedal_offset_masked, dtype=torch.float32
        )

        return (
            selected_feature,
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            room_acoustics,
            midi_id,
            pedal_factor,
        )

    def plot_all_labels(
        self,
        pedal_value,
        quantized_pedal_value,
        pedal_onset,
        pedal_offset,
        soft_pedal_onset,
        soft_pedal_offset,
        label_bin_edges,
        img_name,
    ):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8), dpi=100)
        plt.subplot(6, 1, 1)
        plt.plot(pedal_value, label="True")
        plt.title("True pedal value")
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal Value")
        plt.ylim(0, 128)
        # plot horizontal lines for bin edges
        for bin_edge in label_bin_edges[1:-1]:
            plt.axhline(y=bin_edge, color="r", linestyle="--")
        plt.subplot(6, 1, 2)
        plt.plot(quantized_pedal_value * 127, label="Quantized")
        plt.ylim(0, 128)
        plt.title("Quantized pedal value")
        plt.xlabel("Frame Index")
        plt.ylabel("Quantized Pedal Value")
        plt.subplot(6, 1, 3)
        plt.plot(pedal_onset, label="Onset")
        plt.title("Pedal onset")
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal onset")
        plt.subplot(6, 1, 4)
        for i, onset in enumerate(soft_pedal_onset):
            plt.vlines(i, 0, onset, linewidth=0.5, colors="C0")
            plt.scatter(i, onset, s=1, c="C0")
        plt.title("Soft pedal onset")
        plt.xlabel("Frame Index")
        plt.ylabel("Soft pedal onset")
        plt.subplot(6, 1, 5)
        plt.plot(pedal_offset, label="Offset")
        plt.title("Pedal offset")
        plt.xlabel("Frame Index")
        plt.ylabel("Pedal offset")
        plt.subplot(6, 1, 6)
        for i, offset in enumerate(soft_pedal_offset):
            plt.vlines(i, 0, offset, linewidth=0.5, colors="C0")
            plt.scatter(i, offset, s=1, c="C0")
        plt.title("Soft pedal offset")
        plt.xlabel("Frame Index")
        plt.ylabel("Soft pedal offset")
        plt.tight_layout()
        # save the plot
        plt.savefig(f"{img_name}.png")
        plt.close()
