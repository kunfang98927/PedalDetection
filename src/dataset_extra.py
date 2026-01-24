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
        feature_dim=249,
        on_off_threshold=64,
        normalize_features=False,  # True/False for per-feature normalization across the track
        midi=False, # read MIDI files as additional input too
        dynamic=True,
        external_midi="", # pass external midi files to substitute existing ones
        pred_pedal="", # pass external pedal files
        binarize_pedal_threshold=-1, # whether to binarize the pedal values to 0/1 according to this threshold
        unique_index=False, # whether to use unique index for external data
        pedal_latent=False, # whether to use latent representation of predicted pedal values
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
            normalize_features (bool): Whether to apply per-track feature normalization:
                - False: No normalization (default) 
                - True: Normalize each feature dimension using full track statistics
                        (preserves relative differences between clips from same track)
            midi (bool): Whether to include MIDI modality in the dataset.
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
        self.normalize_features = normalize_features
        
        # MIDI-related parameters
        self.midi = midi
        self.dynamic = dynamic
        self.ex_midi = external_midi
        self.pred_pedal=pred_pedal
        
        # Remove the requirement for MIDI when using predicted pedal
        # if self.pred_pedal != "" and not self.midi:
        #     raise ValueError("use_pred_pedal=True requires use_midi=True. Predicted pedal can only be used when MIDI is available.")
        
        print(f"Use dynamic (pitch velocity): {self.dynamic}")
        if binarize_pedal_threshold == -1:
            self.binarize_pedal = False
        else:
            self.binarize_pedal = True
            self.on_off_threshold = binarize_pedal_threshold
            print(f"Binarize given pedal predictions with threshold: {self.on_off_threshold}")
        if self.pred_pedal == "" and self.binarize_pedal:
            raise ValueError("binarize_pedal=True requires pred_pedal to be provided.")
        self.unique_index = unique_index
        if '885' in self.ex_midi or 'test' in self.ex_midi:
            self.unique_index = True       
        if self.unique_index:
            print("Using unique index for external data")
        self.latent_pedal = pedal_latent
       

        # Cache for track normalization statistics
        self.track_norm_cache = {} if normalize_features else None

        # Open all H5 files and store them in a dictionary.
        self.h5fs = {}
        for ex in self.examples:
            file_path = os.path.join(data_dir, ex["file_path"])
            if file_path not in self.h5fs:
                self.h5fs[file_path] = h5py.File(file_path, "r")
                print(f"Opened {file_path}")
                
                # Verify MIDI data exists if requested
                if self.midi:
                    if "midi_values" not in self.h5fs[file_path] and self.ex_midi == "":
                        raise ValueError(f"Warning: 'midi_values' not found in {file_path}")
                    elif "midi_values" not in self.h5fs[file_path] and self.ex_midi != "":
                        print(f"Warning: 'midi_values' not found in {file_path}, but external MIDI provided.")
                    else:
                        print(f"Found MIDI data in {file_path}")

        # Load external midi files if provided
        if self.ex_midi != "":
            print(f"Loading external MIDI files from {self.ex_midi}")
            external_file_path = os.path.join(data_dir, self.ex_midi)

            # Store external MIDI file with a recognizable key
            self.external_midi_key = "external_midi"
            self.h5fs[self.external_midi_key] = h5py.File(external_file_path, "r")
            print(f"Opened external MIDI file: {external_file_path}")

            if "midi_values" not in self.h5fs[self.external_midi_key]:
                raise ValueError(f"'midi_values' not found in external MIDI file: {external_file_path}")
            else:
                print(f"Found MIDI data in external MIDI file to replace existing MIDI data.")
        else:
            self.external_midi_key = None

        if self.pred_pedal != "":
            print(f"Loading external pedal prediction files from {self.pred_pedal}")
            external_file_path = os.path.join(data_dir, self.pred_pedal)

            # Store external MIDI file with a recognizable key
            self.predict_pedal_key = "predict_pedal"
            self.h5fs[self.predict_pedal_key] = h5py.File(external_file_path, "r")
            print(f"Opened external MIDI file: {external_file_path}")
            
            if self.latent_pedal:
                if "latent_repr" not in self.h5fs[self.predict_pedal_key]:
                    raise ValueError(f"'latent_repr' not found in external pedal file: {external_file_path}")
                else:
                    print(f"Found latent pedal data in external pedal file and use it to train.")
            else:
                if "pedal_values" not in self.h5fs[self.predict_pedal_key]:
                    raise ValueError(f"'pedal_values' not found in external MIDI file: {external_file_path}")
                else:
                    print(f"Found predicted pedal data in external pedal file and use it to train.")
        else:
            self.predict_pedal_key = None


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
        modalities = ["audio features (mel bins and MFCCs)"]
        if self.midi:
            if self.ex_midi != "":
                modalities.append("transcribed MIDI values (note)")
            else:
                modalities.append("gt MIDI values (note)")

        # Add pedal modality info regardless of MIDI
        if self.pred_pedal != "":
            if self.latent_pedal:
                modalities.append("Predicted pedal values (latent)")
            else:
                modalities.append("Predicted pedal values")
        
        print(
            f"Loaded {len(self.examples)} examples from {data_list_path} for split: {self.split}"
        )
        print(f"Using modalities: {', '.join(modalities)}")

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

    def get_track_normalization_stats(self, file_path, example_index):
        """
        Compute normalization statistics for the entire track.
        Results are cached to avoid recomputation.
        
        Args:
            file_path (str): Path to H5 file
            example_index (int): Index of the example in the H5 file
            
        Returns:
            tuple: (mean, std) tensors of shape [1, feature_dim]
        """
        # Create cache key
        cache_key = (file_path, example_index)
        
        # Return cached result if available
        if cache_key in self.track_norm_cache:
            return self.track_norm_cache[cache_key]
        
        # Get full track features
        full_features = self.h5fs[file_path]["features"][str(example_index)][
            :self.feature_dim, :
        ].T  # [full_time, feature_dim]
        
        # Convert to tensor
        full_features = torch.tensor(full_features, dtype=torch.float32)
        
        # Compute statistics across time dimension (dim=0)
        mean = full_features.mean(dim=0, keepdim=True)  # [1, feature_dim]
        std = full_features.std(dim=0, keepdim=True)    # [1, feature_dim]
        
        # Avoid division by zero (using same epsilon as original code)
        std = torch.clamp(std, min=1e-6)
        
        # Cache the result
        self.track_norm_cache[cache_key] = (mean, std)
        
        return mean, std

    def normalize_feature_tensor(self, features, mean, std):
        """
        Normalize features using pre-computed track-level statistics.
        
        Args:
            features (torch.Tensor): Features of shape [time_frames, feature_dim]
            mean (torch.Tensor): Mean statistics of shape [1, feature_dim]
            std (torch.Tensor): Std statistics of shape [1, feature_dim]
            
        Returns:
            torch.Tensor: Normalized features with same shape
        """
        # Handle empty features
        if features.numel() == 0:
            print("[Warning] Attempting to normalize empty features")
            return features
            
        # Apply normalization using track-level statistics
        features = (features - mean) / std
            
        return features
        
    def fetch_segment(self, file_path, example_index, start_frame, end_frame, room_id=None):

        # Slice the feature and pedal arrays.
        selected_feature = self.h5fs[file_path]["features"][str(example_index)][
            :self.feature_dim, start_frame:end_frame
        ].T  # [max_frame, feature_dim]
        selected_pedal_value = self.h5fs[file_path]["instant_values"][
            str(example_index)
        ][1][start_frame:end_frame]

        # Convert features to torch tensor first
        selected_feature = torch.tensor(selected_feature, dtype=torch.float32)
        # print(f"Feature shape before normalization: {selected_feature.shape}")
        
        # Apply normalization if specified (using full track statistics)
        if self.normalize_features:
            mean, std = self.get_track_normalization_stats(file_path, example_index)
            selected_feature = self.normalize_feature_tensor(selected_feature, mean, std)

        # Handle MIDI modality (no normalization - preserve semantic meaning)
        selected_midi = None
        selected_pred_pedal = None
        if self.midi:

            # Get MIDI data - use external if available, otherwise original
            if self.external_midi_key is not None:
                midi_source = self.h5fs[self.external_midi_key]
                # print(f"Using external MIDI data for example {example_index}")
            else:
                midi_source = self.h5fs[file_path]

            if self.unique_index:
                fetch_idx_key = f"{example_index}-r{room_id}"
            else: 
                fetch_idx_key = str(example_index)

            selected_midi_values = midi_source["midi_values"][
                fetch_idx_key][1,:,start_frame:end_frame]
        
            if not self.dynamic: # binarize the midi values if not using dynamics
                selected_midi_values = (selected_midi_values > 0).astype(int)

            selected_midi = selected_midi_values.T
            
            # Convert to tensor (no normalization)
            selected_midi = torch.tensor(selected_midi, dtype=torch.float32)
            # print(f"MIDI shape: {selected_midi.shape}")

        # Handle predicted pedal regardless of MIDI mode
        if self.predict_pedal_key is not None:
            
            if self.unique_index:
                fetch_idx_key = f"{example_index}-r{room_id}"
            else: 
                fetch_idx_key = str(example_index)
                
            if self.latent_pedal:
                selected_pred_pedal_values = self.h5fs[self.predict_pedal_key]["latent_repr"][
                    fetch_idx_key][:, start_frame:end_frame]
            else:
                selected_pred_pedal_values = self.h5fs[self.predict_pedal_key]["pedal_values"][
                    fetch_idx_key][start_frame:end_frame]
            
            
            selected_pred_pedal = selected_pred_pedal_values.T
            selected_pred_pedal = torch.tensor(selected_pred_pedal, dtype=torch.float32)
            
            if selected_pred_pedal.dim() == 1:
                selected_pred_pedal = selected_pred_pedal.unsqueeze(-1)

            # print(f"Predicted pedal shape: {selected_pred_pedal.shape}")
            if self.binarize_pedal:
                selected_pred_pedal = (selected_pred_pedal * 127 >= self.on_off_threshold).float()

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

        # Convert everything else to torch tensors.
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
            
            # Include MIDI in padding, which will be None if MIDI is not used
            (selected_feature, selected_midi, selected_pred_pedal, quantized_pedal_value_masked,
            soft_pedal_onset_masked, soft_pedal_offset_masked, loss_mask) = self.pad_data(
                selected_feature,
                selected_midi,
                selected_pred_pedal,
                quantized_pedal_value_masked,
                soft_pedal_onset_masked,
                soft_pedal_offset_masked,
                loss_mask
            )
            if loss_mask.sum() == 0:
                print(self.split, "[1 - Warning] Empty mask detected!")

        return (
            selected_feature,
            selected_midi,
            selected_pred_pedal,  # Return MIDI data
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
        # Always use original file path for pedal data and other non-MIDI features
        file_path = os.path.join(self.data_dir, ex_info["file_path"])
        example_index = ex_info["example_index"]
        room_id = ex_info["room_id"]
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
            selected_midi,  # MIDI data, which will be None if MIDI is not used
            selected_pred_pedal, # None if no pred pedal is used
            low_res_label,
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask,
        ) = self.fetch_segment(file_path, example_index, start_frame, end_frame, room_id=room_id)

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

        # Update return logic to handle audio+pedal mode without MIDI
        if self.predict_pedal_key is not None:
            if self.midi:
                # Audio + MIDI + Predicted Pedal (existing mode)
                return (
                    selected_feature,
                    selected_midi,
                    selected_pred_pedal,
                    low_res_label,
                    quantized_pedal_value_masked,
                    soft_pedal_onset_masked,
                    soft_pedal_offset_masked,
                    loss_mask
                )
            else:
                # Audio + Predicted Pedal (new mode)
                return (
                    selected_feature,
                    selected_pred_pedal,
                    low_res_label,
                    quantized_pedal_value_masked,
                    soft_pedal_onset_masked,
                    soft_pedal_offset_masked,
                    loss_mask
                )
        elif self.midi:
            # Audio + MIDI (existing mode)
            return (
                selected_feature,
                selected_midi,
                low_res_label,
                quantized_pedal_value_masked,
                soft_pedal_onset_masked,
                soft_pedal_offset_masked,
                loss_mask
            )
        else:
            # Audio only (existing mode)
            return (
                selected_feature,
                low_res_label,
                quantized_pedal_value_masked,
                soft_pedal_onset_masked,
                soft_pedal_offset_masked,
                loss_mask
            )
    
    def pad_data(self, selected_feature, selected_midi, selected_pred_pedal, quantized_pedal_value_masked,
                    soft_pedal_onset_masked, soft_pedal_offset_masked, loss_mask):
        """
        Updated to handle MIDI padding as well
        """
        pad_length = self.max_frame - selected_feature.shape[0]
        if pad_length == self.max_frame:
            print("[Warning] pad_length", pad_length)
            
        # Pad audio features
        selected_feature = F.pad(selected_feature, (0, 0, 0, pad_length), "constant", 0)
        
        # Pad MIDI features if they exist
        if selected_midi is not None:
            selected_midi = F.pad(selected_midi, (0, 0, 0, pad_length), "constant", 0)
        
        if selected_pred_pedal is not None:
            selected_pred_pedal = F.pad(selected_pred_pedal, (0, 0, 0, pad_length), "constant", 0)
            
        
        # Pad labels and masks
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
            selected_midi,
            selected_pred_pedal, 
            quantized_pedal_value_masked,
            soft_pedal_onset_masked,
            soft_pedal_offset_masked,
            loss_mask
        )