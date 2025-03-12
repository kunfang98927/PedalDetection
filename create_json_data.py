import os
import json
import h5py
from tqdm import tqdm
from src.dirs import DATA_PATH_ROOM1, DATA_PATH_ROOM2, DATA_PATH_ROOM3, DATA_PATH_REAL

h5_files = [DATA_PATH_ROOM1, DATA_PATH_ROOM2, DATA_PATH_ROOM3, DATA_PATH_REAL]

train_data = []
val_data = []
test_data = []

for path in h5_files:
    with h5py.File(path, "r") as h5f:
        # Determine total_examples: either from an attribute or by checking the features dataset/group.
        if "total_examples" in h5f.attrs:
            total_examples = h5f.attrs["total_examples"]
        else:
            if isinstance(h5f["features"], h5py.Group):
                total_examples = len(h5f["features"])
            else:
                total_examples = h5f["features"].shape[0]

        for i in tqdm(range(total_examples), desc=f"Processing {path}"):
            # Get feature and instant_values
            if isinstance(h5f["features"], h5py.Group):
                str_idx = str(i)
                features_ds = h5f["features"][str_idx]
            else:
                features_ds = h5f["features"][i]

            if isinstance(h5f["instant_values"], h5py.Group):
                str_idx = str(i)
                # We assume the second row of the dataset is the pedal label.
                inst_values_ds = h5f["instant_values"][str_idx][1]
            else:
                inst_values_ds = h5f["instant_values"][i][1]

            # Get metadata. Here we assume metadata is stored as a dataset,
            # and that each metadata is a numeric array.
            metadata_arr = h5f["metadata"][i][:]
            # Convert metadata to list for JSON (and assume it has four elements):
            metadata_list = metadata_arr.tolist()

            room_id, midi_id, pedal_factor, split = metadata_list

            if "real" in path:
                room_id = 0

            # Create an entry with the necessary information.
            example = {
                "file_path": os.path.abspath(path),
                "example_index": i,
                "num_frames": features_ds.shape[1],
                "room_id": int(room_id),  # 0 for real data, 1, 2, 3 for room data
                "midi_id": int(midi_id),
                "pedal_factor": int(pedal_factor),  # 0 for no pedal, 1 for pedal
                # "split": split,  # 0 for train, 1 for val, 2 for test
            }

            if split == 0:
                train_data.append(example)
            elif split == 1:
                val_data.append(example)
            elif split == 2:
                test_data.append(example)
            else:
                print(f"Warning: Unknown split value {split} for file {path} index {i}")

# Save the JSON files.
output_dir = "sample_data"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_data, f, indent=2)
with open(os.path.join(output_dir, "val.json"), "w") as f:
    json.dump(val_data, f, indent=2)
with open(os.path.join(output_dir, "test.json"), "w") as f:
    json.dump(test_data, f, indent=2)

print("JSON files have been created: train.json, val.json, test.json")
