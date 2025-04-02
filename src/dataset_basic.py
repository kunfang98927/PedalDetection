import os
import json
import h5py
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.utils import (
    calculate_pedal_onset_offset,
    calculate_soft_regresion_label,
    calculate_low_res_pedal_value,
)

class PedalDataset(Dataset):
    def __init__(
        self,
        data_list_path,  # Path to the JSON file (e.g., train.json, val.json, or test.json)
        data_dir="/scratch/kunfang/pedal_data/data/",
        datasets=["r0-pf1"],
        num_samples_per_clip=10,
        max_frame=500,
        label_ratio=1.0,
        label_bin_edges=[0, 64, 128],
        overlap_ratio=0.25,
        split="train",
        num_examples=None,
        randomly_sample=False,
        feature_dim=229,
        on_off_threshold=64,
    ):
        """
        Args:
            data_list_path (str): Path to the JSON file (e.g., train.json, val.json, or test.json).
            data_dir (str): Directory where the H5 files are stored.
            num_samples_per_clip (int): Number of clips to sample per example (for training).
            max_frame (int): Length of the clip in frames.
            label_ratio (float): Portion of the clip where the loss is computed.
            label_bin_edges (list): Used for quantizing pedal values.
            overlap_ratio (float): Overlap ratio for sliding window.
            split (str): "train", "validation", or "test".
        """
        with open(data_list_path, "r") as f:
            self.examples = json.load(f)
            print(f"Loaded {len(self.examples)} examples from {data_list_path}")

        # filter the examples
        self.examples = [
            ex
            for ex in self.examples
            if datasets is None or self.filter_examples(ex, datasets)
        ]
        print(f"Filtered examples: {len(self.examples)}")
        if num_examples is not None:
            # randomly select num_examples
            np.random.seed(0)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_examples]
            print(f"Randomly selected examples: {len(self.examples)}")
            # print out all midi_ids
            selected_midi_ids = set([ex["midi_id"] for ex in self.examples])
            print(f"Selected midi_ids: {selected_midi_ids}")

        self.data_dir = data_dir
        self.num_samples_per_clip = num_samples_per_clip
        self.max_frame = max_frame
        self.label_ratio = label_ratio
        self.label_bin_edges = label_bin_edges
        self.overlap_ratio = overlap_ratio
        self.split = split.lower()
        self.randomly_sample = randomly_sample
        self.feature_dim = feature_dim
        self.on_off_threshold = on_off_threshold

        # Open all H5 files and store them in a dictionary.
        self.h5fs = {}
        for ex in self.examples:
            file_path = os.path.join(data_dir, ex["file_path"])
            if file_path not in self.h5fs:
                self.h5fs[file_path] = h5py.File(file_path, "r")
                print(f"Opened {file_path}")

        # if split is validation, only validate on pedal factor 1
        if self.split == "validation":
            self.examples = [
                ex for ex in self.examples if ex["pedal_factor"] == 1
            ]
            print(f"Filtered examples for validation: {len(self.examples)}")

        # Precompute number of segments per example if not randomly sampling.
        if not self.randomly_sample:
            self.segments_per_example = self.precompute_segments_per_example(self.examples)

        # Print some information.
        print(
            f"Loaded {len(self.examples)} examples from {data_list_path} for split: {self.split}"
        )

    def precompute_segments_per_example(self, examples):
        segments_per_example = []
        for ex in examples:
            num_frames = ex["num_frames"]
            if num_frames > self.max_frame:
                # Compute number of segments with a sliding window.
                step = int(self.max_frame * (1 - self.overlap_ratio))
                segments = math.ceil((num_frames - self.max_frame) / step) + 1
            else:
                segments = 0
            segments_per_example.append({"num_segments": segments})
        return segments_per_example

    def filter_examples(self, ex, datasets):
        """
        Filter examples based on the provided datasets.

        Args:
            ex (dict): Example to filter.
            datasets: ["r1-pf1", "r2-pf1", "r3-pf1", "r0-pf1"].
        """
        if datasets is None:
            return True
        room_id = str(ex["room_id"])
        pedal_factor = str(ex["pedal_factor"])
        if f"r{room_id}-pf{pedal_factor}" in datasets:
            return True
        else:
            return False

    def __len__(self):
        if not self.randomly_sample:
            return sum([seg["num_segments"] for seg in self.segments_per_example])
        else:
            return len(self.examples) * self.num_samples_per_clip
        
    def fetch_segment(self, file_path, example_index, start_frame, end_frame):

        # Slice the feature and pedal arrays.
        selected_feature = self.h5fs[file_path]["features"][str(example_index)][
            :self.feature_dim, start_frame:end_frame
        ].T  # [max_frame, feature_dim]
        selected_pedal_value = self.h5fs[file_path]["instant_values"][
            str(example_index)
        ][1][start_frame:end_frame]

        # Process labels.
        pedal_onset, pedal_offset = calculate_pedal_onset_offset(
            selected_pedal_value, on_off_threshold=self.on_off_threshold
        )
        if len(self.label_bin_edges) == 2: # regression
            quantized_pedal_value = selected_pedal_value / 127.0
        else: # classification
            quantized_pedal_value = np.digitize(
                selected_pedal_value, self.label_bin_edges
            ) - 1
            quantized_pedal_value = np.clip(quantized_pedal_value, 0, len(self.label_bin_edges) - 2)
        # Convert pedal values to soft labels.
        soft_pedal_onset = calculate_soft_regresion_label(pedal_onset)
        soft_pedal_offset = calculate_soft_regresion_label(pedal_offset)

        # Create masks so that loss is computed only on a central portion of the clip.
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

        low_res_label = calculate_low_res_pedal_value(
            selected_pedal_value,
            quantized_pedal_value,
            label_start,
            label_end,
            self.label_bin_edges,
        )

        # Convert everything to torch tensors.
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

        # Create loss mask.
        loss_mask = torch.zeros(self.max_frame, dtype=torch.float32)
        loss_mask[label_start:label_end] = 1.0

        if loss_mask.sum() == 0:
            print(self.split, "[0 - Warning] Empty mask detected!")

        # pad the feature to max_frame
        if selected_feature.shape[0] < self.max_frame:
            if selected_feature.shape[0] == 0:
                print(self.split, "[0 - Warning] Empty feature detected!")
            (selected_feature, quantized_pedal_value_masked,
            soft_pedal_onset_masked, soft_pedal_offset_masked, loss_mask) = self.pad_data(
                selected_feature,
                quantized_pedal_value_masked,
                soft_pedal_onset_masked,
                soft_pedal_offset_masked,
                loss_mask
            )
            if loss_mask.sum() == 0:
                print(self.split, "[1 - Warning] Empty mask detected!")

        return (
            selected_feature,
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask,
        )

    def __getitem__(self, idx):
        # Map global idx to a specific example and segment.
        if not self.randomly_sample:
            running = 0
            for i, example in enumerate(self.segments_per_example):
                seg_count = example["num_segments"]
                if idx < running + seg_count:
                    ex_idx = i
                    seg_idx = idx - running
                    break
                running += seg_count
            else:
                raise IndexError("Index out of range in sliding window mode.")
        else:
            ex_idx = idx // self.num_samples_per_clip
            seg_idx = None  # For training, we'll sample a random segment.

        # Get JSON info for this example.
        ex_info = self.examples[ex_idx]
        file_path = os.path.join(self.data_dir, ex_info["file_path"])
        example_index = ex_info["example_index"]
        num_frames = ex_info["num_frames"]

        # Determine start_frame and end_frame.
        if self.randomly_sample:
            if num_frames > self.max_frame:
                start_frame = np.random.randint(0, num_frames - self.max_frame)
            else:
                start_frame = 0
            end_frame = start_frame + self.max_frame
        else: # sliding window
            step = int(self.max_frame * (1 - self.overlap_ratio))
            start_frame = seg_idx * step
            end_frame = min(start_frame + self.max_frame, num_frames)

        # Fetch the segment.
        (
            selected_feature,
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask,
        ) = self.fetch_segment(file_path, example_index, start_frame, end_frame)

        # protect from empty mask or feature
        if loss_mask.sum() == 0 or selected_feature.shape[0] == 0:
            print("[Warning] Empty mask or feature detected!", loss_mask.sum(), selected_feature.shape[0])
            return self.__getitem__(np.random.randint(0, len(self)))

        if len(self.label_bin_edges) == 3:
            # assert that quantized_pedal_value_masked and low_res_label should only be 0 or 1
            assert not torch.any(
                (quantized_pedal_value_masked != -1) & (quantized_pedal_value_masked != 0) & (quantized_pedal_value_masked != 1)
            ), f"Quantized pedal value should be -1, 0, or 1. Found: {quantized_pedal_value_masked}"
            assert not torch.any(
                (low_res_label != -1) & (low_res_label != 0) & (low_res_label != 1)
            ), f"Low res label should be -1, 0, or 1. Found: {low_res_label}"

        return (
            selected_feature,
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask
        )
    
    def pad_data(self, selected_feature, quantized_pedal_value_masked,
                    soft_pedal_onset_masked, soft_pedal_offset_masked, loss_mask):
        pad_length = self.max_frame - selected_feature.shape[0]
        if pad_length == self.max_frame:
            print("[Warning] pad_length", pad_length)
        selected_feature = F.pad(selected_feature, (0, 0, 0, pad_length), "constant", 0)
        quantized_pedal_value_masked = F.pad(quantized_pedal_value_masked, (0, pad_length), "constant", -1)
        soft_pedal_onset_masked = F.pad(soft_pedal_onset_masked, (0, pad_length), "constant", -1)
        soft_pedal_offset_masked = F.pad(soft_pedal_offset_masked, (0, pad_length), "constant", -1)
        if loss_mask.sum() == 0:
            print("[Warning] before loss_mask.shape", loss_mask.shape, loss_mask.sum())
        loss_mask[-pad_length:] = 0
        if loss_mask.sum() == 0:
            print("[Warning] after loss_mask.shape", loss_mask.shape, loss_mask.sum())
        return (
            selected_feature,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask
        )