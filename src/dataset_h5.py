import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import (
    calculate_pedal_onset_offset,
    calculate_soft_regresion_label,
    calculate_low_res_pedal_value,
)


class PedalDataset(Dataset):
    def __init__(
        self,
        data_path,  # Path to the JSON file (e.g., train.json, val.json, or test.json)
        num_samples_per_clip=10,
        max_frame=500,
        label_ratio=1.0,
        label_bin_edges=[0, 11, 95, 128],
        overlap_ratio=0.25,
        split="train",
        data_filter=None,
        num_examples=None,
    ):
        """
        Args:
            data_path (str): Path to the JSON file with a list of examples.
            num_samples_per_clip (int): Number of clips to sample per example (for training).
            max_frame (int): Length of the clip in frames.
            label_ratio (float): Portion of the clip where the loss is computed.
            label_bin_edges (list): Used for quantizing pedal values.
            overlap_ratio (float): Overlap ratio for sliding window (for val/test).
            split (str): "train", "validation", or "test".
        """
        with open(data_path, "r") as f:
            self.examples = json.load(f)
            print(f"Loaded {len(self.examples)} examples from {data_path}")

        # filter the examples
        self.examples = [
            ex
            for ex in self.examples
            if data_filter is None or self.filter_examples(ex, data_filter)
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

        self.num_samples_per_clip = num_samples_per_clip
        self.max_frame = max_frame
        self.label_ratio = label_ratio
        self.label_bin_edges = label_bin_edges
        self.overlap_ratio = overlap_ratio
        self.split = split.lower()

        # Open all H5 files and store them in a dictionary.
        self.h5fs = {}
        for ex in self.examples:
            file_path = ex["file_path"].replace("/scratch/kunfang/pedal_data/data/", "data/h5/")
            if file_path not in self.h5fs:
                self.h5fs[file_path] = h5py.File(file_path, "r")
                print(f"Opened {file_path}")

        # For validation/test, precompute number of segments per example.
        if self.split in ["test"]: #["validation", "test"]:
            self.segments_per_example = []
            for ex in self.examples:
                num_frames = ex["num_frames"]
                if num_frames > self.max_frame:
                    # Compute number of segments with a sliding window.
                    step = int(self.max_frame * (1 - self.overlap_ratio))
                    segments = (num_frames - self.max_frame) // step
                else:
                    segments = 0
                self.segments_per_example.append(segments)
        # Print some information.
        print(
            f"Loaded {len(self.examples)} examples from {data_path} for split: {self.split}"
        )

    def filter_examples(self, ex, data_filter):
        """
        Filter examples based on the provided data_filter.

        Args:
            ex (dict): Example to filter.
            data_filter: ["r1-pf1", "r2-pf1", "r3-pf1", "r0-pf1"].
        """
        if data_filter is None:
            return True
        room_id = str(ex["room_id"])
        pedal_factor = str(ex["pedal_factor"])
        if f"r{room_id}-pf{pedal_factor}" in data_filter:
            return True
        else:
            return False

    def __len__(self):
        if self.split in ["test"]: #["validation", "test"]:
            return sum(self.segments_per_example)
        else:
            return len(self.examples) * self.num_samples_per_clip

    def __getitem__(self, idx):
        # Map global idx to a specific example and segment.
        if self.split in ["test"]: #["validation", "test"]:
            running = 0
            for i, seg_count in enumerate(self.segments_per_example):
                if idx < running + seg_count:
                    ex_idx = i
                    seg_idx = idx - running
                    break
                running += seg_count
            else:
                raise IndexError("Index out of range in validation/test mode.")
        else:
            ex_idx = idx // self.num_samples_per_clip
            seg_idx = None  # For training, we'll sample a random segment.

        # Get JSON info for this example.
        ex_info = self.examples[ex_idx]
        file_path = ex_info["file_path"].replace("/scratch/kunfang/pedal_data/data/", "data/h5/")
        example_index = ex_info["example_index"]
        num_frames = ex_info["num_frames"]
        room_id = ex_info["room_id"]
        midi_id = ex_info["midi_id"]
        pedal_factor = ex_info["pedal_factor"]
        # print(file_path, example_index, num_frames, room_id, midi_id, pedal_factor)

        # Determine start_frame and end_frame.
        if self.split == "train" or self.split == "validation":
            if num_frames > self.max_frame:
                start_frame = np.random.randint(0, num_frames - self.max_frame)
            else:
                start_frame = 0
            end_frame = start_frame + self.max_frame
        else:
            # For test, use a sliding window.
            step = int(self.max_frame * (1 - self.overlap_ratio))
            start_frame = seg_idx * step
            # If the computed window exceeds the available frames, adjust.
            if start_frame + self.max_frame > num_frames:
                start_frame = num_frames - self.max_frame
            end_frame = start_frame + self.max_frame

        # Slice the feature and pedal arrays.
        selected_feature = self.h5fs[file_path]["features"][str(example_index)][
            :, start_frame:end_frame
        ].T  # [max_frame, feature_dim]
        selected_pedal_value = self.h5fs[file_path]["instant_values"][
            str(example_index)
        ][1][start_frame:end_frame]
        # print(idx, selected_feature.shape, selected_pedal_value.shape)

        # Process labels.
        # if len(self.label_bin_edges) == 2:
        pedal_onset, pedal_offset = calculate_pedal_onset_offset(
            selected_pedal_value, on_off_threshold=0
        )
        quantized_pedal_value = selected_pedal_value / 127.0
        # else:
        #     quantized_pedal_value = (
        #         np.digitize(selected_pedal_value, self.label_bin_edges) - 1
        #     )
        #     pedal_onset, pedal_offset = calculate_pedal_onset_offset(
        #         quantized_pedal_value, on_off_threshold=self.label_bin_edges[1]
        #     )
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

        return (
            selected_feature,
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            room_id,
            midi_id,
            pedal_factor,
        )
