import numpy as np
import pretty_midi
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import gc
import h5py
import psutil
import json
import librosa

# Import your existing functions
from feature_extraction import (
    get_memory_usage, discretize_pedal_value, get_pedal_state_distribution,
    get_instant_values, SR, HOP_LENGTH, N_FFT, POINTS_PER_HOP,
    MIN_MIDI_PITCH, MAX_MIDI_PITCH
)

CSV_PATH = 'metadata.csv'
DATA_PATH = '../maestro-v3.0.0'
BATCH_SIZE = 50
MAX_BATCHES = 0  # 0 means process all

# JSON file paths for different splits
TRAIN_JSON = 'sample_data/train.json'
VALID_JSON = 'sample_data/val.json' 
TEST_JSON = 'sample_data/test.json'
MIDI_DIR = 'transkun_midi'  # Same MIDI files, just extracting pedal data instead


def process_pedal_only(midi_path, target_num_frames, sr=SR, hop_length=HOP_LENGTH, 
                      n_fft=N_FFT, points_per_hop=POINTS_PER_HOP):
    """
    Process only MIDI file to extract pedal (CC64) data with exact frame count.
    Only extracts sustain pedal (CC64) values.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        n_frames = target_num_frames
        
        # Calculate duration based on target frames
        total_duration = (hop_length * (n_frames - 1) + n_fft) / sr
        
        # Initialize output arrays - only pedal values (single dimension)
        pedal_instant_values = np.zeros(n_frames)  # Just pedal values per frame
        
        # Create interpolation grid
        points_per_frame = points_per_hop
        remaining_points = (n_frames - 1) * points_per_hop
        total_points = points_per_frame + remaining_points
        interp_times = np.linspace(0, total_duration, num=total_points)
        
        # Initialize pedal interpolated values
        pedal_interp_values = np.zeros(total_points)
        
        # Process pedal data (CC64) from all instruments
        for instrument in pm.instruments:
            # Look for control changes (CC64 = sustain pedal)
            for cc in instrument.control_changes:
                if cc.number == 64:  # CC64 is sustain pedal
                    # Find the time index for this control change
                    time_idx = np.searchsorted(interp_times, cc.time, side='left')
                    if time_idx < total_points:
                        # Set pedal value from this point forward until next change
                        # Binarize: 0-63 = off (0), 64-127 = on (1)
                        pedal_value = 1 if cc.value >= 64 else 0
                        pedal_interp_values[time_idx:] = pedal_value

        # Extract frame-level pedal features - one value per frame
        for frame_idx in range(n_frames):
            start_idx = frame_idx * points_per_hop
            
            # Take the pedal value from the middle of each frame
            mid_idx = start_idx + points_per_hop // 2
            if mid_idx < len(pedal_interp_values):
                pedal_instant_values[frame_idx] = pedal_interp_values[mid_idx]
            elif start_idx < len(pedal_interp_values):
                # Fallback to start if middle is out of bounds
                pedal_instant_values[frame_idx] = pedal_interp_values[start_idx]

        # Clean up
        del interp_times, pedal_interp_values
        
        return pedal_instant_values

    except Exception as e:
        raise ValueError(f"Error processing MIDI file {midi_path}: {e}")


def verify_audio_duration(audio_path, expected_frames, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT, tolerance_frames=5):
    """
    Verify that the audio duration matches the expected number of frames.
    """
    try:
        
        # Load audio to get actual duration
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Calculate actual frames that would be generated
        actual_frames = int(np.ceil((len(y) - n_fft) / hop_length)) + 1
        
        # Check if frame counts are close enough
        frame_difference = abs(actual_frames - expected_frames)
        is_valid = frame_difference <= tolerance_frames
        duration_seconds = len(y) / sr
        
        return is_valid, actual_frames, duration_seconds, frame_difference
        
    except Exception as e:
        print(f"    Warning: Could not verify audio duration for {audio_path}: {e}")
        return True, expected_frames, None, 0


def load_all_json_metadata(train_json=TRAIN_JSON, valid_json=VALID_JSON, test_json=TEST_JSON):
    """
    Load and concatenate metadata from train/valid/test JSON files.
    
    Returns:
        list: Combined metadata from all splits
    """
    all_metadata = []
    
    json_files = [
        (train_json, 'train'),
        (valid_json, 'validation'), 
        (test_json, 'test')
    ]
    
    for json_path, split_name in json_files:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                split_data = json.load(f)
            print(f"Loaded {len(split_data)} examples from {split_name}: {json_path}")
            all_metadata.extend(split_data)
        else:
            print(f"Warning: {split_name} JSON file not found: {json_path}")
    
    print(f"Total metadata entries: {len(all_metadata)}")
    return all_metadata


def create_pedal_h5_from_json(metadata_list, output_h5_path, csv_path=CSV_PATH, 
                             data_path=DATA_PATH, batch_size=BATCH_SIZE, 
                             max_batches=MAX_BATCHES, verify_audio=True):
    """
    Create pedal HDF5 file directly from JSON metadata.
    
    Args:
        metadata_list: List of JSON metadata items
        output_h5_path: Output HDF5 file path
        csv_path: Path to metadata CSV file
        data_path: Path to MAESTRO data directory
        batch_size: Number of examples to process per batch
        max_batches: Maximum batches to process (0 = all)
        verify_audio: Whether to verify audio duration matches num_frames
    """
    print(f"Creating Pedal HDF5: {output_h5_path}")
    print(f"Processing {len(metadata_list)} examples")
    
    # Load CSV for file path lookup
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} entries")
    
    # Apply batch limit if specified
    if max_batches > 0:
        total_to_process = min(len(metadata_list), max_batches * batch_size)
        print(f"Limited to {total_to_process} examples ({max_batches} batches)")
    else:
        total_to_process = len(metadata_list)
    
    # Statistics
    processed_count = 0
    failed_count = 0
    duration_checks = 0
    duration_mismatches = 0
    
    # Create HDF5 file
    with h5py.File(output_h5_path, 'w') as h5f:
        # Create only pedal group - no MIDI groups
        pedal_instant_group = h5f.create_group('pedal_values')
        
        # Create metadata dataset
        # Convert JSON metadata to array format [room_id, midi_id, pedal_factor, split]
        metadata_array = np.array([
            [item['room_id'], item['midi_id'], item['pedal_factor'], item['split']]
            for item in metadata_list[:total_to_process]
        ], dtype=np.float32)
        
        h5f.create_dataset(
            'metadata',
            data=metadata_array,
            compression='gzip',
            chunks=True
        )
        
        # Process in batches
        num_batches = (total_to_process + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_to_process)
            
            print(f"\nBatch {batch_idx+1}/{num_batches} (examples {start_idx} to {end_idx-1})")
            
            batch_processed = 0
            batch_failed = 0
            
            for i in range(start_idx, end_idx):
                json_item = metadata_list[i]
                
                try:
                    # Get info directly from JSON
                    midi_id = json_item['midi_id']
                    room_id = json_item['room_id'] 
                    num_frames = json_item['num_frames']
                    example_index = json_item['example_index']
                    
                    # Look up file paths in CSV using midi_id
                    matching_rows = df[df['midi_id'] == midi_id]
                    if len(matching_rows) == 0:
                        raise ValueError(f"No CSV entry found for midi_id {midi_id}")
                    
                    csv_row = matching_rows.iloc[0]
                    
                    # Get MIDI file path (same files, extracting pedal data)
                    midi_filename = str(midi_id) + '.mid' 
                    midi_path = os.path.join(MIDI_DIR, midi_filename)
                    
                    # Get audio file path (for verification)
                    if room_id == 0:
                        audio_filename = csv_row['orig_audio_filename']
                        audio_path = os.path.join(data_path, audio_filename)
                    else:
                        audio_path = csv_row['synthesis_path']
                    
                    # Verify files exist
                    if not os.path.exists(midi_path):
                        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
                    
                    # Verify audio duration if requested
                    if verify_audio:
                        if not os.path.exists(audio_path):
                            raise FileNotFoundError(f"Audio file not found: {audio_path}")
                        
                        duration_checks += 1
                        is_valid, actual_frames, duration, frame_diff = verify_audio_duration(
                            audio_path, num_frames
                        )
                        
                        if not is_valid:
                            duration_mismatches += 1
                            print(f"    Warning: Example {example_index} frame mismatch - Expected: {num_frames}, Actual: {actual_frames} (diff: {frame_diff})")
                    
                    # Process MIDI with exact frame count - only pedal data
                    pedal_instant = process_pedal_only(
                        midi_path, target_num_frames=num_frames
                    )
                    
                    # Store only pedal data using example_index as key
                    str_idx = str(example_index)
                    
                    pedal_instant_group.create_dataset(
                        str_idx,
                        data=pedal_instant,
                        compression='gzip',
                        chunks=True
                    )
                    
                    batch_processed += 1
                    processed_count += 1
                    
                except Exception as e:
                    print(f"    Error processing example {example_index} (midi_id {midi_id}): {e}")
                    batch_failed += 1
                    failed_count += 1
                    
                    # Create zero-filled pedal data to maintain indexing
                    try:
                        str_idx = str(json_item['example_index'])
                        num_frames = json_item['num_frames']
                        
                        pedal_instant_group.create_dataset(
                            str_idx,
                            data=np.zeros(num_frames),  # Shape: (frames,) for pedal
                            compression='gzip',
                            chunks=True
                        )
                    except Exception as fill_error:
                        print(f"    Could not create zero-filled data for {example_index}: {fill_error}")
            
            print(f"    Batch {batch_idx+1}: Processed {batch_processed}, Failed {batch_failed}")
            print(f"    Total so far: {processed_count} processed, {failed_count} failed")
            print(f"    Memory usage: {get_memory_usage()}")
            
            gc.collect()
        
        # Add file attributes
        h5f.attrs['total_examples'] = total_to_process
        h5f.attrs['processed_examples'] = processed_count
        h5f.attrs['failed_examples'] = failed_count
        h5f.attrs['duration_checks'] = duration_checks
        h5f.attrs['duration_mismatches'] = duration_mismatches
        h5f.attrs['sr'] = SR
        h5f.attrs['hop_length'] = HOP_LENGTH
        h5f.attrs['n_fft'] = N_FFT
        h5f.attrs['points_per_hop'] = POINTS_PER_HOP
        h5f.attrs['creation_time'] = datetime.now().isoformat()
        h5f.attrs['csv_source'] = csv_path
        h5f.attrs['data_path'] = data_path
    
    print(f"\n{'='*60}")
    print(f"Pedal HDF5 creation completed!")
    print(f"Output file: {output_h5_path}")
    print(f"Successfully processed: {processed_count}/{total_to_process}")
    print(f"Failed: {failed_count}/{total_to_process}")
    if verify_audio and duration_checks > 0:
        print(f"Duration verification: {duration_checks} checked, {duration_mismatches} mismatches")
        mismatch_rate = (duration_mismatches / duration_checks) * 100
        print(f"Mismatch rate: {mismatch_rate:.1f}%")
    print(f"{'='*60}")
    
    return output_h5_path


def verify_pedal_h5_structure(pedal_h5_path, metadata_list, num_checks=10):
    """
    Verify the created pedal HDF5 structure against the JSON metadata.
    """
    print(f"\n Verifying Pedal HDF5 structure: {pedal_h5_path}")
    
    with h5py.File(pedal_h5_path, 'r') as h5f:
        # Check groups exist - only pedal and metadata
        required_groups = ['pedal_values', 'metadata']
        for group in required_groups:
            if group in h5f:
                print(f"Group '{group}' exists")
            else:
                print(f"Group '{group}' missing")
                return False
        
        # Check random examples
        check_indices = np.random.choice(len(metadata_list), size=min(num_checks, len(metadata_list)), replace=False)
        
        print(f"\nChecking {len(check_indices)} random examples:")
        
        for i, meta_idx in enumerate(check_indices):
            json_item = metadata_list[meta_idx]
            example_index = json_item['example_index']
            expected_frames = json_item['num_frames']
            midi_id = json_item['midi_id']
            
            str_idx = str(example_index)
            
            # Check if pedal data exists
            has_pedal_instant = str_idx in h5f['pedal_values']
            
            if has_pedal_instant:
                # Check frame counts and shape
                pedal_data = h5f['pedal_values'][str_idx]
                pedal_shape = pedal_data.shape
                expected_shape = (expected_frames,)  # Shape for pedal: n_frames
                
                if pedal_shape == expected_shape:
                    print(f"Example {example_index}: midi_id={midi_id}, shape={pedal_shape}")
                else:
                    print(f"Example {example_index}: Shape mismatch - Expected:{expected_shape}, Got:{pedal_shape}")
            else:
                print(f"Example {example_index}: Missing pedal data")
    
    print("Structure verification complete")
    return True


def main():
    """Main function"""
    print("Creating Pedal HDF5 from JSON metadata")
    print("="*60)
    
    # Load all JSON metadata files
    all_metadata = load_all_json_metadata(TRAIN_JSON, VALID_JSON, TEST_JSON)
    
    if len(all_metadata) == 0:
        print("\n No metadata loaded! Please check your JSON file paths:")
        print(f"  Train: {TRAIN_JSON}")
        print(f"  Valid: {VALID_JSON}")
        print(f"  Test: {TEST_JSON}")
        return
    
    # Create output filename
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_h5_path = f"pedal{len(all_metadata)}_{timestamp}.h5"
    
    # Create the pedal HDF5
    pedal_h5_path = create_pedal_h5_from_json(
        metadata_list=all_metadata,
        output_h5_path=output_h5_path,
        csv_path=CSV_PATH,
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        max_batches=MAX_BATCHES,
        verify_audio=True
    )
    
    # Verify structure
    verify_pedal_h5_structure(pedal_h5_path, all_metadata)
    
    print(f"\n Pedal HDF5 created: {pedal_h5_path}")
    print(f"   Uses same example_index as your existing JSON structure")
    print(f"   Contains {len(all_metadata)} examples from all splits")


if __name__ == "__main__":
    main()