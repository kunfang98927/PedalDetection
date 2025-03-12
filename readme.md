## Pedal Detection

### Create the dataset .json file

Run the following command will create a .json file which contains the path of all .h5 dataset files with num_frames, midi_id, room_id, pedal_factor for each audio file. Easy to load the dataset in the training stage.
```
python create_json_data.py
```

The output .json file will be have the following format:
```
[
  {
    "file_path": "/scratch/kunfang/pedal_data/data/kong2552room1synth0303.h5",
    "example_index": 12,
    "num_frames": 16776,
    "room_id": 1,
    "midi_id": 7,
    "pedal_factor": 1
  },
  {
    "file_path": "/scratch/kunfang/pedal_data/data/kong2552room1synth0303.h5",
    "example_index": 13,
    "num_frames": 16776,
    "room_id": 1,
    "midi_id": 7,
    "pedal_factor": 0
  },
  ...
]
```

### Train the model

```
python train_h5.py
```

### Inference

Run the following command will inference the model and save the results in .npy files.
```
python inference_h5.py
```

### View the result

This will read the .npy files saved in inference stage and calculate all metrics (MSE, MAE, F1) and plot the results.
```
python calculate_metric.py
```