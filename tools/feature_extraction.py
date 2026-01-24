import numpy as np
import librosa
import pretty_midi
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import gc  # For garbage collection
import re
import h5py  # For HDF5 file format
import psutil  # For memory monitoring


# System utilities
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    return f"{mem:.1f} MB"

def get_current_time():
    return datetime.now().strftime("%m%d")

TIME = str(get_current_time())

# Constants for audio processing (must be edited here)
N_FFT = 2048
SR = 16000
N_MELS = 229
N_MFCC = 20
HOP_SECONDS = 1.
FRAMES_PER_SECOND = 100
POINTS_PER_HOP = 10
HOP_LENGTH = SR // FRAMES_PER_SECOND
IF_NORMALIZE = True
SUBSET = 1
REAL_AUDIO = True
ROOM = 2
BATCH_SIZE = 50  # Process this many tracks before adding to HDF5
MAX_BATCHES = 0  # Added parameter to limit the number of batches for testing


### Included in the dataset but not used in the model
######################################################
def discretize_pedal_value(value, no_pedal_threshold=10, full_pedal_threshold=95):
    """
    Convert CC64 value to discrete class with customizable thresholds:
    0: No pedal (0 to no_pedal_threshold)
    1: Partial pedal (no_pedal_threshold+1 to full_pedal_threshold-1)
    2: Full pedal (full_pedal_threshold to 127)
    """
    if value <= no_pedal_threshold:
        return 0
    elif value >= full_pedal_threshold:
        return 2
    else:
        return 1

def get_pedal_state_distribution(values, no_pedal_threshold=10, full_pedal_threshold=95):
    """
    First discretize values into classes 0,1,2 then compute distribution
    """
    discrete_classes = np.array([
        discretize_pedal_value(v, no_pedal_threshold, full_pedal_threshold) 
        for v in values
    ])
    class_counts = np.bincount(discrete_classes, minlength=3)
    return class_counts / len(values)

######################################################

def get_instant_values(frame_points):
    """
    Get raw pedal values for start, middle, and end points of a frame
    """
    n_points = len(frame_points)
    start_idx = 0
    mid_idx = (n_points - 1) // 2
    end_idx = n_points - 1
    
    return (frame_points[start_idx], frame_points[mid_idx], frame_points[end_idx])

def process_audio_midi(wav_path, midi_path, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT, 
                      n_mels=N_MELS, n_mfcc=N_MFCC, points_per_hop=POINTS_PER_HOP,
                      no_pedal_threshold=10, full_pedal_threshold=95,
                      normalize=True, mfcc_drop=False):
    """Process WAV and MIDI files using log-mel spectrograms, MFCCs, and pedal data"""
    # Load and process audio
    y, _ = librosa.load(wav_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=False
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if mfcc_drop:
        mfccs_zero = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc+1, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels,
            center=False
        )
        mfccs = mfccs_zero[1:n_mfcc+1,:]
    else:
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels,
            center=False
        )
        
    if normalize: # Per feature by track

        log_mel_mean = np.mean(log_mel_spec, axis=1, keepdims=True)
        log_mel_std = np.std(log_mel_spec, axis=1, keepdims=True)
        log_mel_spec = (log_mel_spec - log_mel_mean) / (log_mel_std + 1e-6)

        mfcc_mean = np.mean(mfccs, axis=1, keepdims=True)
        mfcc_std = np.std(mfccs, axis=1, keepdims=True)
        mfccs = (mfccs - mfcc_mean) / (mfcc_std + 1e-6)

    n_frames = log_mel_spec.shape[1]
    
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        # Collect CC64 events
        pedal_events = []
        for instrument in pm.instruments:
            for control in instrument.control_changes:
                if control.number == 64:
                    pedal_events.append((control.time, control.value))
        pedal_events.sort(key=lambda x: x[0])
           
        if not pedal_events:
            raise ValueError(f"No pedal events found in MIDI file: {midi_path}")
            
        # Initialize labels
        average_labels = np.zeros((3, n_frames))
        instant_values = np.zeros((3, n_frames))
        
        # Calculate total duration and points
        total_duration = (hop_length * (n_frames - 1) + n_fft) / sr
        points_per_frame = points_per_hop
        remaining_points = (n_frames - 1) * points_per_hop
        total_points = points_per_frame + remaining_points
     
        # Create interpolation points all at once
        interp_times = np.linspace(0, total_duration, num=total_points)
        interp_values = np.zeros_like(interp_times)
        
        # Fill interpolated values
        current_event_idx = 0
        current_value = pedal_events[current_event_idx][1]
        next_event_idx = 1
        
        for i, time in enumerate(interp_times):
            # Move to next event if we've passed it
            while next_event_idx < len(pedal_events) and pedal_events[next_event_idx][0] <= time:
                current_event_idx = next_event_idx
                current_value = pedal_events[current_event_idx][1]
                next_event_idx += 1
            interp_values[i] = current_value
        
        # Process frames
        for frame_idx in range(n_frames):
            start_idx = frame_idx * points_per_hop
            frame_points = interp_values[start_idx:start_idx + points_per_frame]     
            
            average_labels[:, frame_idx] = get_pedal_state_distribution(
                frame_points, no_pedal_threshold, full_pedal_threshold)
            
            start_val, mid_val, end_val = get_instant_values(frame_points)
            instant_values[0, frame_idx] = start_val
            instant_values[1, frame_idx] = mid_val
            instant_values[2, frame_idx] = end_val
                
    except Exception as e:
        raise ValueError(f"Error processing MIDI file {midi_path}: {e}")
    
    features = np.concatenate([log_mel_spec, mfccs], axis=0)
    
    # Clean up to free memory
    del y, mel_spec, mfccs, interp_times, interp_values
    
    return features, average_labels, instant_values

def filter_and_sample_data(df, subset_ratio=SUBSET, real_audio=REAL_AUDIO):
    """Filter and sample the dataset based on configuration"""
    if real_audio:
        # Select all for real audio
        subset_ratio = 1.0
        # Filter by pedal_multi_factor == 1.0
        df = df[df['pedal_multi_factor'] == 1.0].copy()
        print(f'Filtered DataFrame length: {len(df)}')
    
    # Filter to exclude train data without cc64
    # mask = ((df['cc64_status'] == 'has_cc64') | (~(df['orig_split'] == 'train')))
    # mask = ((df['cc64_status'] != 'has_cc64') & (df['orig_split'] == 'train'))
    # df_filtered = df[mask].copy()  # Create explicit copy to avoid SettingWithCopyWarning
    df_filtered = df.copy()
    print('Filtered length after excluding :', len(df_filtered))
    
    # Map splits to numeric values
    split_mapping = {'train': 0, 'validation': 1, 'test': 2}
    df_filtered.loc[:, 'split_encoded'] = df_filtered['orig_split'].map(split_mapping)
    
    # Sample subset of tracks if needed by train/test split
    if subset_ratio < 1.0:

        track_groups = df_filtered.groupby("split_encoded")["track_index"].unique()

        # Sample subset_ratio of track groups per split
        sampled_tracks = {
            split: np.random.choice(
                tracks, 
                size=int(len(tracks) * subset_ratio), 
                replace=False
            ) for split, tracks in track_groups.items()
        }

        subset_df = df_filtered[
            df_filtered.apply(
                lambda row: row["track_index"] in sampled_tracks[row["split_encoded"]], 
                axis=1
            )
        ]
        
        rows_per_split = subset_df.groupby('split_encoded').size()
        print("Split sizes:", rows_per_split)
        
        return subset_df.reset_index(drop=True)
    
    return df_filtered.reset_index(drop=True)

def process_dataset_h5(df, batch_size=BATCH_SIZE, real_audio=REAL_AUDIO, room=ROOM, max_batches=None):
    """Process data in batches and save to a single HDF5 file"""
    total_rows = len(df)
    print(f"Processing {total_rows} rows in batches of {batch_size}")
    df['num_frames'] = None

    if real_audio:
        real_str = "real"
    else:
        real_str = "synth"
        
    if room and not real_audio:
        room_str = f"room{str(room)}"
    else:
        room_str = ""
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    h5_file_path = f"kong{total_rows}{room_str}{real_str}{timestamp}.h5"
    print(f"Will save all data to: {h5_file_path}")
    
    num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling
    
   
    if max_batches is not None and max_batches > 0:
        num_batches = min(num_batches, max_batches)
        print(f"Limiting processing to {num_batches} batches for testing")
        # Adjust total_rows to match the limited number of batches
        total_rows_to_process = min(total_rows, num_batches * batch_size)
    else:
        total_rows_to_process = total_rows
    
   
    with h5py.File(h5_file_path, 'w') as h5f:
       
        features_group = h5f.create_group('features')
        avg_labels_group = h5f.create_group('average_labels')
        instant_values_group = h5f.create_group('instant_values')
      
        metadata_dataset = h5f.create_dataset(
            'metadata', 
            shape=(total_rows_to_process, 4),  # 4 columns for metadata
            dtype=np.float32, 
            chunks=True,
            compression='gzip'
        )
        
        processed_count = 0
        successful_indices = []  # Track successfully processed indices
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_size_actual = end_idx - start_idx
            
            print(f"\nProcessing batch {batch_idx+1}/{num_batches} (rows {start_idx}-{end_idx-1})")
            batch_df = df.iloc[start_idx:end_idx]
            
            
            features_list = []
            average_labels_list = []
            instant_values_list = []
            batch_indices = []  # To keep track of which rows were successful
            
            # Process each file in the batch
            for idx, row in batch_df.iterrows():
                try:
                    midi_path = row['midi_data_path']
                    if row['synth_setting'] == 0:
                        wav_path = os.path.join('../maestro-v2.0.0', row['orig_audio_filename'])
                    else:
                        wav_path = row['synthesis_path']
                    
                    
                    features, average_labels, instant_values = process_audio_midi(
                        wav_path,
                        midi_path,
                        sr=SR,
                        hop_length=HOP_LENGTH,
                        n_fft=N_FFT,
                        n_mels=N_MELS,
                        n_mfcc=N_MFCC,
                        normalize=IF_NORMALIZE
                    )
                    
                    
                    features_list.append(features)
                    average_labels_list.append(average_labels)
                    instant_values_list.append(instant_values)
                    batch_indices.append(idx)
                    successful_indices.append(idx)  
                    print(idx, 'num_frames', features.shape[1])
                    df.at[idx, 'num_frames'] = features.shape[1]
                    
                    # Show progress
                    processed_in_batch = len(batch_indices)
                    if processed_in_batch % 5 == 0 or processed_in_batch == batch_size_actual:
                        print(f"  Processed {processed_in_batch}/{batch_size_actual} in current batch")
                        
                except Exception as e:
                    print(f"Error processing entry {idx} (batch position {idx - start_idx}): {e}")
                    continue
            
            # Store metadata for the batch
            for i, global_idx in enumerate(batch_indices):
                # Calculate relative position in metadata_dataset
                metadata_position = len(successful_indices) - len(batch_indices) + i
                
                # Get the row from the full DataFrame
                metadata_row = df.loc[global_idx][['synth_setting', 'track_index', 'pedal_multi_factor', 'split_encoded']].values.astype(np.float32)
                
                # Store at the correct position
                metadata_dataset[metadata_position] = metadata_row
                
            print(f"  Stored metadata for {len(batch_indices)} sequences")
            
            # Store features, labels, and values
            for i, idx in enumerate(batch_indices):
                # Use sequential numbering instead of DataFrame index
                str_idx = str(len(successful_indices) - len(batch_indices) + i)
                
            
                features_dataset = features_group.create_dataset(
                    str_idx, 
                    data=features_list[i],
                    compression='gzip',
                    chunks=True
                )
                
                
                avg_labels_dataset = avg_labels_group.create_dataset(
                    str_idx, 
                    data=average_labels_list[i],
                    compression='gzip',
                    chunks=True
                )
                
                
                instant_values_dataset = instant_values_group.create_dataset(
                    str_idx, 
                    data=instant_values_list[i],
                    compression='gzip',
                    chunks=True
                )
            
            processed_count += len(batch_indices)
            print(f"Batch {batch_idx+1} added to HDF5. Total processed so far: {processed_count}/{total_rows_to_process}")
            print(f"Memory usage: {get_memory_usage()}")
            
            # Print some statistics for the first batch
            if batch_idx == 0:
                print("\nSequence lengths (first 3 examples):")
                for i in range(min(3, len(features_list))):
                    print(f"  Example {i}: features {features_list[i].shape}, " +
                          f"average_labels {average_labels_list[i].shape}, " +
                          f"instant_values {instant_values_list[i].shape}")
                
            # Clean up memory after saving
            del features_list, average_labels_list, instant_values_list
            gc.collect()
        
        # Resize the metadata dataset to match the actual number of processed examples
        if processed_count < total_rows_to_process:
            metadata_dataset.resize((processed_count, 4))
            
        
        h5f.attrs['total_examples'] = processed_count
        h5f.attrs['n_mels'] = N_MELS
        h5f.attrs['n_mfcc'] = N_MFCC
        h5f.attrs['sr'] = SR
        h5f.attrs['hop_length'] = HOP_LENGTH
        h5f.attrs['creation_time'] = datetime.now().isoformat()
    
    print(f"\nAll processing complete! Data saved to {h5_file_path}")
    print(f"Successfully processed {processed_count} out of {total_rows_to_process} entries")
    return h5_file_path

def main():
   
    csv_path = 'synthesis_metadata.csv'
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV file")
    
    filtered_df = filter_and_sample_data(df, SUBSET, True)
    print(f"Total filtered length: {len(filtered_df)}")

    
    
    if MAX_BATCHES > 0:
        print(f"Running in test mode: processing only {MAX_BATCHES} batches")
    
    
    h5_file = process_dataset_h5(filtered_df, BATCH_SIZE, REAL_AUDIO, ROOM, MAX_BATCHES)

    filtered_df.to_csv('filtered_processed_metadata.csv', index=False)
    
    print(f"\nProcessing complete! Data saved to {h5_file}")

if __name__ == "__main__":
    main()