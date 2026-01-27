import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm

# Import your existing modules
from gesture import SimplePedalGestureSegmenter
from action_analysis import deferred_label_context


def count_actions_per_type(states: np.ndarray) -> Dict[str, int]:
    """
    Count the number of frames for each action type from deferred_label_context output.
    
    Args:
        states: Array of states (-1, 0, 1) from deferred_label_context
    
    Returns:
        Dictionary with counts for each action type
    """
    unique, counts = np.unique(states, return_counts=True)
    action_counts = dict(zip(unique, counts))
    
    # Ensure all action types are present
    result = {
        'release_frames': action_counts.get(-1, 0),
        'sustained_frames': action_counts.get(0, 0),
        'press_frames': action_counts.get(1, 0)
    }
    
    return result


def count_gestures_per_type(segments: list) -> tuple:
    gesture_counts = {}
    frame_counts = {}  
    
    for segment in segments:
        classification = segment['classification']
        duration = segment['duration']
        
        if classification not in gesture_counts:
            gesture_counts[classification] = 0
            frame_counts[classification] = 0


        gesture_counts[classification] += 1
        frame_counts[classification] += duration    
    
    return gesture_counts, frame_counts 


def compute_frame_indices(num_frames: int, start_index: int = 0) -> tuple:
    """
    Compute start and end indices for extracting frames from NPY array.
    
    Args:
        num_frames: Number of frames needed
        start_index: Starting index (cumulative from previous entries)
    
    Returns:
        Tuple of (start_idx, end_idx) for array slicing
    """
    return start_index, start_index + num_frames


def main():
    # Configuration
    metadata_csv_path = "../metadata.csv"
    json_file_path = "../sample_data/test.json"

    # Grount truth
    npy_file_path = "../inf-data/p_v_labels_test_set_r0-pf1.npy"
    output_csv_path = "../csv_dir/test_gt_gesture.csv"

    # AUDIO+MIDI
    # npy_file_path = "../inf-data/a+m-p_v_preds_test_set_r0-pf1.npy"
    # output_csv_path = "../csv_dir/test_a+m_gesture.csv"

    # AUDIO
    # npy_file_path = "../inf-data/a-p_v_preds_test_set_r0-pf1.npy"
    # output_csv_path = "../csv_dir/test_a_gesture.csv"

    # BINARY
    # npy_file_path = "../inf-data/binary-p_v_preds_test_set_r0-pf1.npy"
    # output_csv_path = "../csv_dir/test_bin+a_gesture.csv"
    
    # Analysis parameters
    gesture_threshold = 0.01 # 0.01 for gt; 0.05 for pred
    action_window_size = 19
    action_min_r_squared = 0.5
    
    # === Step 1: Load and filter metadata.csv ===
    
    metadata_df = pd.read_csv(metadata_csv_path)
    print(f"Loaded metadata.csv with {len(metadata_df)} rows")
    
    test_df = metadata_df[(metadata_df['orig_split'] == 'test') & (metadata_df['room_id'] == 0)].copy()
    print(f"After filtering by split='test': {len(test_df)} rows")
    
    if len(test_df) == 0:
        print("ERROR: No test split entries found!")
        return


    # === Step 2: Load JSON metadata ===

    with open(json_file_path, 'r') as f:
        json_metadata = json.load(f)
    
    print(f"Loaded JSON metadata with {len(json_metadata)} entries")
    
    # Create a lookup dictionary: midi_id -> json_entry
    json_lookup = {}
    if isinstance(json_metadata, list):
        for entry in json_metadata:
            midi_id = entry.get('midi_id')
            if midi_id is not None:
                json_lookup[midi_id] = entry
    else:
        # If it's a dictionary, handle accordingly
        for key, entry in json_metadata.items():
            midi_id = entry.get('midi_id')
            if midi_id is not None:
                json_lookup[midi_id] = entry
    
    print(f"Created JSON lookup with {len(json_lookup)} midi_id entries")


    # === Step 3: Match CSV with JSON and get num_frames ===

    # Match midi_ids and get num_frames from JSON
    matched_rows = []
    unmatched_count = 0
    
    for idx, row in test_df.iterrows():
        midi_id = row['midi_id']
        
        if midi_id in json_lookup:
            json_entry = json_lookup[midi_id]
            num_frames = (json_entry.get('json_num_frames') or 
                         json_entry.get('num_frames') or 
                         json_entry.get('frames'))
            
            if num_frames is not None:
                row_dict = row.to_dict()
                row_dict['json_num_frames'] = num_frames
                row_dict['original_idx'] = idx  # Keep track of original index
                matched_rows.append(row_dict)
            else:
                print(f"Warning: No num_frames found in JSON for midi_id {midi_id}")
                unmatched_count += 1
        else:
            print(f"Warning: midi_id {midi_id} not found in JSON")
            unmatched_count += 1
    
    print(f"Successfully matched: {len(matched_rows)} rows")
    print(f"Unmatched: {unmatched_count} rows")
    
    if len(matched_rows) == 0:
        print("ERROR: No matching entries found between CSV and JSON!")
        return

    # === Step 4: Load NPY file ===

    npy_data = np.load(npy_file_path)
    print(f"NPY file loaded. Shape: {npy_data.shape}")
    
    # === Step 5: Process each matched row ===
    
    for row in matched_rows:
        row['release_frames'] = 0
        row['sustained_frames'] = 0
        row['press_frames'] = 0
    
    # We'll add gesture columns dynamically as we discover gesture types
    all_gesture_types = set()
    gesture_data = {}  # Store gesture counts per row index
    frame_data = {}
    
    # Initialize segmenter
    segmenter = SimplePedalGestureSegmenter(threshold=gesture_threshold, min_cycle_duration=3, print_out=False)
    
    # Track totals for summary
    total_action_counts = {'release_frames': 0, 'sustained_frames': 0, 'press_frames': 0}
    total_gesture_counts = {}
    total_frame_counts = {}
    
    successful_count = 0
    current_index = 0  # Running index for NPY array access
    
    for row_idx, row in tqdm(enumerate(matched_rows), total=len(matched_rows), desc="Processing rows"):
        try:
            midi_id = row['midi_id']
            json_num_frames = row['json_num_frames']
            
            # Compute indices for this entry
            start_idx, end_idx = compute_frame_indices(json_num_frames, current_index)
            
            # Check if we have enough data in NPY array
            if end_idx > len(npy_data):
                print(f"Error: Not enough data in NPY for row {row_idx} (midi_id {midi_id}). "
                      f"Needed: {end_idx}, Available: {len(npy_data)}")
                break
            
            # Extract signal from NPY array
            signal = npy_data[start_idx:end_idx]
            
            # Update current index for next iteration
            current_index = end_idx
            
            # Verify signal length matches JSON
            if len(signal) != json_num_frames:
                print(f"Length mismatch for midi_id {midi_id}: "
                      f"NPY={len(signal)}, JSON={json_num_frames}")
                continue
            
            # === ACTION ANALYSIS ===
            
            # Get action states using deferred_label_context
            states = deferred_label_context(
                    signal,
                    slope_threshold=0.005, 
                    window_size=action_window_size,
                    min_r_squared=action_min_r_squared,
                    plot=False, 
                    print_out=False
            )
            
            # Count action frames
            action_counts = count_actions_per_type(states)
            
            # Update row
            row['release_frames'] = action_counts['release_frames']
            row['sustained_frames'] = action_counts['sustained_frames']
            row['press_frames'] = action_counts['press_frames']
            
            # Update totals
            for key, count in action_counts.items():
                total_action_counts[key] += count
            
            # === GESTURE ANALYSIS ===
            # Get gesture segments using SimplePedalGestureSegmenter
            segments = segmenter.segment_signal(signal)
            
            # Count gestures per type
            gesture_counts, frame_counts = count_gestures_per_type(segments)
            
            # Store gesture data for this row
            gesture_data[row_idx] = gesture_counts
            frame_data[row_idx] = frame_counts

            # Track all gesture types found
            all_gesture_types.update(gesture_counts.keys())
            
            # Update gesture totals
            for gesture_type, count in gesture_counts.items():
                if gesture_type not in total_gesture_counts:
                    total_gesture_counts[gesture_type] = 0
                total_gesture_counts[gesture_type] += count

            for gesture_type, frames in frame_counts.items():
                if gesture_type not in total_frame_counts:
                    total_frame_counts[gesture_type] = 0
                total_frame_counts[gesture_type] += frames
            
            successful_count += 1
            
        except Exception as e:
            print(f"Error processing row {row_idx} (midi_id {midi_id}): {e}")
            continue
    
    # === Step 6: Add gesture columns to dataframe ===
    
    # Add gesture count columns dynamically
    for gesture_type in sorted(all_gesture_types):
        col_name = f'gesture_{gesture_type}'
        
        for row_idx, row in enumerate(matched_rows):
            if row_idx in gesture_data and gesture_type in gesture_data[row_idx]:
                count = gesture_data[row_idx][gesture_type]
            else:
                count = 0
            row[col_name] = count

    # Add gesture frame count columns dynamically
    for gesture_type in sorted(all_gesture_types):
        col_name = f'gesture_{gesture_type}_frames'
        
        for row_idx, row in enumerate(matched_rows):
            if row_idx in frame_data and gesture_type in frame_data[row_idx]:
                frames = frame_data[row_idx][gesture_type]
            else:
                frames = 0
            row[col_name] = frames
    
    # === Step 7: Create output dataframe and save results ===
    
    result_df = pd.DataFrame(matched_rows)
    
    # Remove the temporary columns
    if 'original_idx' in result_df.columns:
        result_df = result_df.drop('original_idx', axis=1)
    
    result_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Total test split entries: {len(test_df)}")
    print(f"Successfully matched with JSON: {len(matched_rows)}")
    print(f"Successfully processed: {successful_count}/{len(matched_rows)} rows")


if __name__ == "__main__":
    main()