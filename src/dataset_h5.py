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
        label_bin_edges=[0, 11, 95, 128],
        overlap_ratio=0.25,
        split="train",
        num_examples=None,
        randomly_sample=False,
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
            segments_per_example.append({
                # "file_path": ex["file_path"], 
                # "example_index": ex["example_index"],
                # "num_frames": num_frames,
                # "room_id": ex["room_id"],
                # "midi_id": ex["midi_id"],
                # "pedal_factor": ex["pedal_factor"],
                "num_segments": segments
            })
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
            :, start_frame:end_frame
        ].T  # [max_frame, feature_dim]
        selected_pedal_value = self.h5fs[file_path]["instant_values"][
            str(example_index)
        ][1][start_frame:end_frame]
        # print(idx, selected_feature.shape, selected_pedal_value.shape)

        # Process labels.
        pedal_onset, pedal_offset = calculate_pedal_onset_offset(
            selected_pedal_value, on_off_threshold=64
        )
        quantized_pedal_value = selected_pedal_value / 127.0
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

        # pad the feature to max_frame
        if selected_feature.shape[0] < self.max_frame:
            (selected_feature, quantized_pedal_value_masked,
            soft_pedal_onset_masked, soft_pedal_offset_masked, loss_mask) = self.pad_data(
                selected_feature,
                quantized_pedal_value_masked,
                soft_pedal_onset_masked,
                soft_pedal_offset_masked,
                loss_mask
            )

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
        room_id = ex_info["room_id"]
        midi_id = ex_info["midi_id"]
        pedal_factor = ex_info["pedal_factor"]

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

        return (
            selected_feature,
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask,
            room_id,
            midi_id,
            pedal_factor,
        )
    
    def pad_data(self, selected_feature, quantized_pedal_value_masked,
                    soft_pedal_onset_masked, soft_pedal_offset_masked, loss_mask):
        pad_length = self.max_frame - selected_feature.shape[0]
        selected_feature = F.pad(selected_feature, (0, 0, 0, pad_length), "constant", 0)
        quantized_pedal_value_masked = F.pad(quantized_pedal_value_masked, (0, pad_length), "constant", -1)
        soft_pedal_onset_masked = F.pad(soft_pedal_onset_masked, (0, pad_length), "constant", -1)
        soft_pedal_offset_masked = F.pad(soft_pedal_offset_masked, (0, pad_length), "constant", -1)
        loss_mask[-pad_length:] = 0
        return (
            selected_feature,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask
        )

class PedalRoomContrastiveDataset(PedalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.anchor_examples = [
            ex for ex in self.examples if ex["pedal_factor"] == 1 and ex["room_id"] != 0
        ]
        self.anchor_segments_per_example = self.precompute_segments_per_example(self.anchor_examples)

        print(f"Loaded {len(self.anchor_examples)} anchor examples in total {len(self.examples)}")

    def __len__(self):
        if not self.randomly_sample:
            return sum([seg["num_segments"] for seg in self.anchor_segments_per_example])
        else:
            return len(self.anchor_examples) * self.num_samples_per_clip

    def __getitem__(self, idx):
        # Map global idx to a specific example and segment.
        if not self.randomly_sample:
            running = 0
            for i, example in enumerate(self.anchor_segments_per_example):
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
        ex_info = self.anchor_examples[ex_idx]
        file_path = os.path.join(self.data_dir, ex_info["file_path"])
        example_index = ex_info["example_index"]
        num_frames = ex_info["num_frames"]
        anchor_room_id = ex_info["room_id"]
        anchor_midi_id = ex_info["midi_id"]
        anchor_pedal_factor = ex_info["pedal_factor"]

        # Determine start_frame and end_frame.
        if self.randomly_sample:
            if num_frames > self.max_frame:
                anchor_start_frame = np.random.randint(0, num_frames - self.max_frame)
            else:
                anchor_start_frame = 0
            anchor_end_frame = anchor_start_frame + self.max_frame
        else: # sliding window
            step = int(self.max_frame * (1 - self.overlap_ratio))
            anchor_start_frame = seg_idx * step
            anchor_end_frame = min(anchor_start_frame + self.max_frame, num_frames)

        # Fetch the segment.
        (
            anchor_feature,
            anchor_low_res_label,
            anchor_quantized_pedal_value_masked,
            anchor_soft_pedal_onset_masked,
            anchor_soft_pedal_offset_masked,
            anchor_loss_mask,
        ) = self.fetch_segment(file_path, example_index, anchor_start_frame, anchor_end_frame)

        anchor_sample = (
            anchor_feature, anchor_low_res_label, anchor_quantized_pedal_value_masked,
            anchor_soft_pedal_onset_masked, anchor_soft_pedal_offset_masked, anchor_loss_mask,
            anchor_room_id, anchor_midi_id, anchor_pedal_factor
        )

        # Select positive and negative samples.
        positive_sample = self.select_data_sample(anchor_room_id, anchor_midi_id, anchor_start_frame, anchor_end_frame, positive=True)
        negative_sample = self.select_data_sample(anchor_room_id, anchor_midi_id, anchor_start_frame, anchor_end_frame, positive=False)

        # check anchor, positive, and negative samples
        assert anchor_room_id != positive_sample[6] if positive_sample is not None else True
        assert anchor_room_id == negative_sample[6] if negative_sample is not None else True
        assert anchor_midi_id == positive_sample[7] == negative_sample[7] if positive_sample is not None and negative_sample is not None else True
        assert anchor_pedal_factor == positive_sample[8] if positive_sample is not None else True
        assert anchor_pedal_factor != negative_sample[8] if negative_sample is not None else True
        assert anchor_pedal_factor == 1 if positive_sample is not None else True

        positive_sample = positive_sample if positive_sample is not None else anchor_sample
        negative_sample = negative_sample if negative_sample is not None else anchor_sample

        # print("Low res label:", anchor_low_res_label, positive_sample[1], negative_sample[1])
        # for anchor_label, pos_label, neg_label in zip(anchor_quantized_pedal_value_masked, positive_sample[2], negative_sample[2]):
        #     print(anchor_label, pos_label, neg_label)

        # Return anchor plus positive and negative features for triplet
        return anchor_sample + positive_sample + negative_sample
    
    def select_data_sample(self, room_id, midi_id, start_frame, end_frame, positive=True):
        """
        Select a data sample for room contrastive sampling.

        For positive sampling, select a different room with the same midi_id, pedal_factor=1.
        For negative sampling, select the same room with pedal_factor=0.
        """
        if positive:
            # Select a different room with the same midi_id and pedal_factor=1.
            positive_candidates = []
            for ex in self.examples:
                if ex["room_id"] != room_id and ex["midi_id"] == midi_id and ex["pedal_factor"] == 1:
                    positive_candidates.append(ex)
            if len(positive_candidates) == 0:
                # If no positive sample is found, return the anchor sample.
                return None
            positive_example = np.random.choice(positive_candidates)
            positive_segment = self.fetch_segment(
                os.path.join(self.data_dir, positive_example["file_path"]),
                positive_example["example_index"], start_frame, end_frame
            )
            return positive_segment + (positive_example["room_id"], positive_example["midi_id"], positive_example["pedal_factor"])
        else:
            # Select the same room with pedal_factor=0.
            negative_candidates = []
            for ex in self.examples:
                if ex["room_id"] == room_id and ex["midi_id"] == midi_id and ex["pedal_factor"] == 0:
                    negative_candidates.append(ex)
            if len(negative_candidates) == 0:
                # If no negative sample is found, return the anchor sample.
                return None
            negative_example = np.random.choice(negative_candidates)
            negative_segment = self.fetch_segment(
                os.path.join(self.data_dir, negative_example["file_path"]),
                negative_example["example_index"], start_frame, end_frame
            )
            return negative_segment + (negative_example["room_id"], negative_example["midi_id"], negative_example["pedal_factor"])