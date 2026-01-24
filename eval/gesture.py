import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import matplotlib.patches as patches
from scipy.signal import find_peaks

np.set_printoptions(threshold=np.inf)  


class SimplePedalGestureSegmenter:
    """
    Simple Piano Pedal Gesture Segmentation System with Clean Decision Tree.
    
    Step 1: Find gesture boundaries (press to release)
    Step 2: Classify using max occupation + duration + oscillations
    
    Decision Logic:
    - Max occupation = % of frames at 90% of max value
    - Low occupation + Short → Pinnacle  
    - Low occupation + Long → Hill
    - High occupation + Smooth → Highland
    - High occupation + Oscillating → Mountain
    """
    
    def __init__(self, threshold: float = 0.01, min_cycle_duration: int = 3, print_out=True):
        self.threshold = threshold
        self.min_cycle_duration = min_cycle_duration
        self.print_out = print_out
    
    def find_gesture_boundaries(self, signal: np.ndarray) -> List[Dict]:
        """
        Step 1: Find all gesture boundaries ensuring COMPLETE coverage.
        SIMPLIFIED approach: Accept ALL above-threshold regions as gestures.
        
        Args:
            signal: 1D numpy array of pedal values
            
        Returns:
            List of segments covering the entire signal with no gaps
        """
        segments = []
        current_pos = 0

        if self.print_out:
            print(f"Finding boundaries with threshold={self.threshold}, min_duration={self.min_cycle_duration}")
            print(f"Signal length: {len(signal)}, above threshold: {np.sum(signal > self.threshold)} frames")
        
        while current_pos < len(signal):
            if signal[current_pos] > self.threshold:
                # Found start of above-threshold region
                gesture_start = max(current_pos - 1, 0) # make sure it start from the real zero!
                gesture_end = current_pos
                
                # Find the end of this above-threshold region
                while gesture_end < len(signal) and signal[gesture_end] > self.threshold:
                    gesture_end += 1
                
                # Include trailing zero
                if gesture_end + 1 < len(signal) and signal[gesture_end+1] <= self.threshold: 
                    gesture_end += 1

                duration = gesture_end - gesture_start
                if segments and gesture_start <= segments[-1]['end']:
                    print()
                    print("="*60)
                    print(f"WARNING: Overlap detected! Current gesture starts at {gesture_start}, "
                        f"but previous segment ends at {segments[-1]['end']}")
                    print(f"  Previous: {segments[-1]['start']}-{segments[-1]['end']} ({segments[-1]['classification']})")
                    print(f"  Current:  {gesture_start}-{gesture_end-1} (gesture)")
                
                # SIMPLIFIED: Accept ALL above-threshold regions as gestures (no minimum duration filter)
                segments.append({
                    'start': gesture_start,
                    'end': gesture_end - 1,
                    'duration': duration,
                    'signal': signal[gesture_start:gesture_end].copy(),
                    'classification': 'gesture'
                })
                # print(f"  Gesture: {gesture_start}-{gesture_end-1} ({duration} frames)")
                
                # Move to the end of this region
                current_pos = gesture_end
                
            else:
                # We're in a below-threshold region
                plain_start = current_pos
                
                # Find the end of the below-threshold region
                while current_pos < len(signal) and signal[current_pos] <= self.threshold:
                    current_pos += 1
                
                plain_end = current_pos - 1

                # Add plain segment
                # Only keep plain if it still has length
                if plain_end > plain_start:
                    # If this plain is directly before a gesture, trim off the last frame
                    if plain_end + 1 < len(signal) and signal[plain_end + 1] > self.threshold:
                        plain_end -= 1
                    duration = plain_end - plain_start + 1
                    if segments and plain_start <= segments[-1]['end']:
                        print()
                        print("="*60)
                        print(f"WARNING: Overlap detected! Current plain starts at {plain_start}, "
                            f"but previous segment ends at {segments[-1]['end']}")
                        print("="*60)
                    segments.append({
                        'start': plain_start,
                        'end': plain_end,
                        'duration': duration,
                        'signal': signal[plain_start:plain_end + 1].copy(),
                        'classification': 'plain'
                    })
                    # print(f"  Plain: {plain_start}-{plain_end} ({duration} frames)")

        
        # Verify complete coverage
        if self.print_out:
            total_frames_covered = sum(seg['duration'] for seg in segments)
            if total_frames_covered != len(signal):
                print(f"WARNING: Coverage mismatch! Signal: {len(signal)}, Covered: {total_frames_covered}")
            else:
                print(f"✓ Complete coverage: {total_frames_covered} frames")
            
        return segments

    def analyze_gesture_signal(
        self,
        gesture_signal: np.ndarray,
        max_occupation_threshold: float = 0.9
    ) -> Dict:
        """
        Analyze gesture characteristics using simplified logic.

        Key measurements:
        1. Max occupation: How much of the gesture stays at/near maximum value
        2. Duration: Length of the gesture 
        3. Oscillations: Up-down patterns indicating mountain technique

        Args:
            gesture_signal: 1D array of the gesture signal
            peak_threshold: fraction of max_value to consider a peak
            valley_floor: minimum absolute value allowed for valleys
            max_occupation_threshold: fraction of max_value for "near max" frames

        Returns:
            Dictionary with gesture characteristics
        """
        if len(gesture_signal) == 0:
            return {
                'max_value': 0,
                'max_occupation': 0,
                'duration': 0,
                'oscillation_count': 0,
                'near_max_frames': 0,
                'max_streak_at_peak': 0
            }

        max_value = np.max(gesture_signal)
        duration = len(gesture_signal)

        # Calculate max occupation
        near_max_frames = np.sum(gesture_signal >= max_value * max_occupation_threshold)
        max_occupation = near_max_frames / duration

        # Adaptive parameters based on signal properties
        signal_range = np.max(gesture_signal) - np.min(gesture_signal)
        adaptive_prominence = signal_range * 0.1  # 10% of signal range
        adaptive_distance = max(3, len(gesture_signal) // 100)  # 1% of signal length, min 3

        valleys, _ = find_peaks(-gesture_signal, 
                            distance=adaptive_distance, 
                            prominence=adaptive_prominence)

        valleys = [v for v in valleys if gesture_signal[v] >= self.threshold]

        # print(f"adaptive_distance: {adaptive_distance}, adaptive_prominence: {adaptive_prominence}, valleys: {valleys}")

        # Count oscillations as number of valleys (local minima)
        oscillation_count = len(valleys)

        # Longest streak at/near peak
        max_streak_at_peak = 0
        current_streak = 0
        for val in gesture_signal:
            if val >= max_value * max_occupation_threshold:
                current_streak += 1
                max_streak_at_peak = max(max_streak_at_peak, current_streak)
            else:
                current_streak = 0

        return {
            'max_value': max_value,
            'duration': duration,
            'max_occupation': max_occupation,
            'near_max_frames': near_max_frames,
            'oscillation_count': oscillation_count,
            'max_streak_at_peak': max_streak_at_peak
        }


    def classify_gesture(
        self,
        characteristics: Dict,
        max_occupation_threshold: float = 0.65,   # % of frames near max
        duration_threshold: int = 100,            # min frames for long gesture previously 100
        oscillation_threshold: int = 1            # oscillation cutoff
    ) -> str:
        self.max_occ_thres = max_occupation_threshold
        self.dur_thres = duration_threshold
        self.osc_thres = oscillation_threshold
        """
        Duration stats: mean=147.8, median=70.0, std=402.9
        Max value stats: mean=113.6, median=127.0, std=23.6
        Max occupation stats: mean=0.645, median=0.688, std=0.220
        Oscillation stats: mean=1.3, median=0.0, std=6.4


        Classify gesture using a simplified decision tree.

        Logic:
        1. Check oscillations (mountain vs smooth patterns)
        2. Check max occupation (sustained vs brief contact with maximum)
        3. Check duration (short vs long gestures)

        Args:
            characteristics: Dictionary from analyze_gesture_signal
            max_occupation_threshold: minimum fraction of frames at "near max"
            duration_threshold: minimum frames for a gesture to be considered long
            oscillation_threshold: minimum oscillations for "mountain" type

        Returns:
            Classification string:
            - 'mountain': oscillating, long, sustained gestures
            - 'highland': smooth, long, sustained gestures
            - 'pinnacle': brief peaks (short or rare sustained contact)
            - 'hill': brief peaks but long duration
            - 'plain': fallback for very weak gestures
        """
        max_occupation = characteristics['max_occupation']
        duration = characteristics['duration']
        oscillation_count = characteristics['oscillation_count']

        # # Case 1: oscillatory gestures → "mountain"
        # if oscillation_count > oscillation_threshold:
        #     return "mountain"

        # Case 2: brief contact with max value
        # if max_occupation < self.max_occ_thres:
        #     if duration < self.dur_thres:
        #         if oscillation_count < self.osc_thres:
        #             return "pinnacle"
        #         else:
        #             return "hill"
        #     else:
        #         if oscillation_count >= self.osc_thres:
        #             return "mountain" 
        #         else:
        #             return "highland"      # long but brief
        # else:
        #     # Case 3: sustained contact with max value
        #     if duration < self.dur_thres:
        #         return "pinnacle"  # rare: short but strong
        #     else:
        #         return "highland"  # long & smooth sustain
            

        if duration < self.dur_thres:
            if max_occupation < self.max_occ_thres:
                if oscillation_count < self.osc_thres:
                    return "pinnacle"
                else:
                    return "hill"
            else:
                return "pinnacle"
        else:
            if max_occupation < self.max_occ_thres:
                return "mountain"
            else:
                return "highland"

        # if duration <= duration_threshold:
        #     if oscillation_count < oscillation_threshold:
        #         return "pinnacle"
        #     else:
        #         return "hill"
        # else:
        #     if max_occupation >= max_occupation_threshold and oscillation_count < oscillation_threshold:
        #         return "highland"
        #     else:
        #         return "mountain"

    
    def segment_signal(self, signal: np.ndarray) -> List[Dict]:
        """
        Complete segmentation: find boundaries then classify each segment.
        
        Args:
            signal: 1D numpy array of pedal signal
            
        Returns:
            List of classified gesture segments covering the entire signal
        """
        if self.print_out:
            print(f"\n=== SEGMENTING SIGNAL ===")
            print(f"\n=== CLASSIFYING GESTURES ===")
        
        # Step 1: Find gesture boundaries
        segments = self.find_gesture_boundaries(signal)
                
        
        # Step 2: Classify each gesture segment
        for i, segment in enumerate(segments):
            if segment['classification'] == 'gesture':
                if self.print_out:
                    print(f"\nClassifying segment {i+1}: frames {segment['start']}-{segment['end']}")
                
                # Analyze and classify this gesture
                characteristics = self.analyze_gesture_signal(segment['signal'])
                classification = self.classify_gesture(characteristics)
                if self.print_out:
                    # Detailed logging of the decision process
                    print(f"  Max value: {characteristics['max_value']:.3f}")
                    print(f"  Duration: {characteristics['duration']} frames")
                    print(f"  Max occupation: {characteristics['max_occupation']:.1%} ({characteristics['near_max_frames']}/{characteristics['duration']} frames ≥90% max)")
                    print(f"  Oscillations: {characteristics['oscillation_count']}")
                    print(f"  Peak streak: {characteristics['max_streak_at_peak']} frames")
                
                # Show decision logic
                max_occ = characteristics['max_occupation']
                duration = characteristics['duration']
                oscillations = characteristics['oscillation_count']
                
                if self.print_out:
                    if duration < self.dur_thres:
                        # Short duration
                        if max_occ < self.max_occ_thres:
                            if oscillations < self.osc_thres:
                                print(f"  → Short duration + Brief contact + Smooth = PINNACLE")
                            else:
                                print(f"  → Short duration + Brief contact + Oscillations = HILL")
                        else:
                            print(f"  → Short duration + Sustained contact = PINNACLE")
                    else:
                        # Long duration
                        if max_occ < self.max_occ_thres:
                            print(f"  → Long duration + Brief contact = MOUNTAIN")
                        else:
                            print(f"  → Long duration + Sustained contact = HIGHLAND")
                    
                    print(f"  → Final classification: {classification.upper()}")
                
                segment['classification'] = classification
                segment['characteristics'] = characteristics
            else:
                # Already classified as plain
                segment['characteristics'] = {
                    'max_value': np.max(segment['signal']) if len(segment['signal']) > 0 else 0,
                    'duration': segment['duration'],
                    'max_occupation': 0,
                    'oscillation_count': 0
                }
        
        # Add IDs
        for i, segment in enumerate(segments):
            segment['id'] = i
        
        # Verify coverage
        if self.print_out:
            total_frames = sum(s['duration'] for s in segments)
            print(f"\n=== COVERAGE CHECK ===")
            print(f"Signal length: {len(signal)}")
            print(f"Total coverage: {total_frames}")
            print(f"Complete: {'✓' if total_frames == len(signal) else '✗'}")
            
        return segments
    
    def print_summary(self, segments: List[Dict]):
        """Print a summary of the segmentation results."""
        print(f"\n=== SEGMENTATION SUMMARY ===")
        print(f"Total segments: {len(segments)}")
        
        # Count by type
        type_counts = {}
        for segment in segments:
            t = segment['classification']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"Segment types: {type_counts}")
        
        print(f"\nDetailed segments:")
        for segment in segments:
            chars = segment.get('characteristics', {})
            max_val = chars.get('max_value', 0)
            max_occ = chars.get('max_occupation', 0)
            oscillations = chars.get('oscillation_count', 0)
            
            print(f"  {segment['id']:2d}. {segment['classification']:8s}: "
                  f"frames {segment['start']:3d}-{segment['end']:3d} "
                  f"({segment['duration']:3d} frames, max: {max_val:.3f}, "
                  f"occ: {max_occ:.1%}, osc: {oscillations})")


def plot_simple_segmentation(signal: np.ndarray, segments: List[Dict], 
                           start_frame: int = 0, end_frame: int = None,
                           title: str = "Simple Pedal Gesture Segmentation",
                           threshold: float = 0.):
    """
    Plot signal with gesture segmentation.
    """
    if end_frame is None:
        end_frame = len(signal)
    
    frames = np.arange(start_frame, min(end_frame, len(signal)))
    signal_portion = signal[start_frame:end_frame]

    # print("signals to be plotted", signal_portion)
    
    # Filter segments to the selected range
    segments_in_range = [s for s in segments 
                        if s['start'] < end_frame and s['end'] >= start_frame]
    
    print("segment to be plotted", segments_in_range)
    # Colors for different gesture types
    colors = {
        'highland': '#E74C3C',   # Red
        'mountain': '#3498DB',   # Blue  
        'pinnacle': '#2ECC71',   # Green
        'hill': '#F39C12',       # Orange
        'plain': '#ECF0F1'       # Light gray
    }
    
    plt.figure(figsize=(15, 6))
    
    # Plot signal
    plt.plot(frames, signal_portion, 'k-', linewidth=2, alpha=0.8)
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold {threshold}')
    plt.ylabel('Pedal Value', fontsize=12)
    plt.xlabel('Frame', fontsize=12)
    plt.title(f'{title} - Frames {start_frame} to {end_frame}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.1)
    
    # Add segment markings
    for segment in segments_in_range:
        s_start = max(segment['start'], start_frame)
        s_end = min(segment['end'], end_frame - 1)
        
        if s_start <= s_end:
            color = colors.get(segment['classification'], '#CCCCCC')
            
            # Add colored background
            plt.axvspan(s_start, s_end, alpha=0.4, color=color, edgecolor='black', linewidth=0.5)
            
            # Add gesture type label
            mid_point = (s_start + s_end) / 2
            y_pos = np.max(signal[s_start:s_end+1]) + 0.05
            
            plt.text(mid_point, y_pos, segment['classification'], 
                    ha='center', va='bottom', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='black'))
    
    # Add legend
    legend_elements = [patches.Patch(color=color, label=gesture_type) 
                      for gesture_type, color in colors.items()]
    plt.legend(handles=legend_elements + [plt.gca().lines[1]], 
              loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()

def smooth_signal(signal: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply a simple moving average smoothing to reduce minor zigzags.
    
    Args:
        signal: 1D array of signal values
        window_size: number of points for the moving average (odd number preferred)
    
    Returns:
        Smoothed signal
    """
    if window_size < 2:
        return signal  # No smoothing
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same')
    return smoothed

# Test function for your data
def test_with_your_data(gt_path: str, pred_path: str, signal_idx: int = 0, start_idx: int = 0, 
                        load_length: int = 500, smooth: bool = False,
                        gt_threshold: float = 0., pred_threshold: float = 0.05):
    """
    Test the simple segmenter with your actual data files.
    
    Args:
        gt_path: Path to ground truth .npy file
        pred_path: Path to prediction .npy file
        signal_idx: Which signal to process if multiple
        start_idx: Start frame idx to load
        load_length: Number of frames to load
        smooth: If we want to smooth predict data to remove noise/artifact
    """
    try:
        # Load data
        gt_data = np.load(gt_path)
        pred_data = np.load(pred_path)
        
        print(f"Loaded data shapes: GT {gt_data.shape}, Pred {pred_data.shape}")
        
        # Extract signals
        if gt_data.ndim > 1:
            gt_signal = gt_data[signal_idx]
            pred_signal = pred_data[signal_idx]
        else:
            gt_signal = gt_data
            pred_signal = pred_data

        if smooth:
            pred_signal = smooth_signal(pred_signal)
        
        print(f"Processing signal {signal_idx}, length: {len(gt_signal)}")
        print(f"GT range: [{np.min(gt_signal):.3f}, {np.max(gt_signal):.3f}]")
        print(f"Pred range: [{np.min(pred_signal):.3f}, {np.max(pred_signal):.3f}]")
        
        # Create segmenter - you can adjust these parameters
        segmenter_gt = SimplePedalGestureSegmenter(
            threshold=gt_threshold,  # Adjust based on your data
            min_cycle_duration=10  # Adjust based on your frame rate
        )

        segmenter_pred = SimplePedalGestureSegmenter(
            threshold=pred_threshold,  # Adjust based on your data
            min_cycle_duration=10  # Adjust based on your frame rate
        )
        
        # Segment both signals
        print(f"\n{'='*60}")
        print(f"SEGMENTING GROUND TRUTH")
        print(f"{'='*60}")
        gt_segments = segmenter_gt.segment_signal(gt_signal[start_idx:start_idx+load_length])
        segmenter_gt.print_summary(gt_segments)
        
        print(f"\n{'='*60}")
        print(f"SEGMENTING PREDICTIONS")
        print(f"{'='*60}")
        pred_segments = segmenter_pred.segment_signal(pred_signal[start_idx:start_idx+load_length])
        segmenter_pred.print_summary(pred_segments)
        
        # Plot results
        plot_simple_segmentation(gt_signal[start_idx:start_idx+load_length], gt_segments, 
                                start_frame=0, end_frame=load_length,
                                title=f"Ground Truth - Signal {signal_idx}",
                                threshold=gt_threshold)
        
        plot_simple_segmentation(pred_signal[start_idx:start_idx+load_length], pred_segments,
                                start_frame=0, end_frame=load_length, 
                                title=f"Predictions - Signal {signal_idx}",
                                threshold=pred_threshold)
        

        return gt_signal, pred_signal, gt_segments, pred_segments
        
    except Exception as e:
        print(f"Error: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # # Test with synthetic data first
    # print("=== TESTING WITH SYNTHETIC DATA ===")
    
    # # Create test signal with all gesture types
    # signal = np.zeros(1000)
    
    # # Highland pattern: frames 100-250 (long + high occupation)
    # signal[100:130] = np.linspace(0, 1, 30)      # ramp up
    # signal[130:220] = 1.0                        # sustain
    # signal[220:250] = np.linspace(1, 0, 30)      # ramp down
    
    # # Pinnacle pattern: frames 350-400 (short + low occupation)
    # t = np.arange(50)
    # signal[350:400] = np.exp(-((t - 25) / 10)**2)
    
    # # Mountain pattern: frames 500-600 (long + oscillating)
    # signal[500:530] = np.linspace(0, 0.9, 30)    # ramp up
    # for i in range(530, 570):                    # oscillate
    #     t = (i - 530) / 40 * 4 * np.pi
    #     signal[i] = 0.8 + 0.2 * np.sin(t)
    # signal[570:600] = np.linspace(0.8, 0, 30)    # ramp down
    
    # # Hill pattern: frames 700-800 (long + brief peak)
    # signal[700:750] = np.linspace(0, 1, 50)
    # signal[750:760] = np.linspace(1, 0.7, 10)
    # signal[760:800] = np.linspace(0.7, 0, 40)
    
    # segmenter = SimplePedalGestureSegmenter(threshold=0.01, min_cycle_duration=10)
    # segments = segmenter.segment_signal(signal)
    # segmenter.print_summary(segments)
    
    # plot_simple_segmentation(signal, segments, title="Test Signal with New Logic")
    
    # print("\n" + "="*60)
    # print("TO USE WITH YOUR DATA:")
    # print("="*60)
    # print("gt_path = 'path/to/your/gt_file.npy'")
    # print("pred_path = 'path/to/your/pred_file.npy'")
    # print("test_with_your_data(gt_path, pred_path, signal_idx=0)")



    gt_path = "../inf-data/p_v_labels_test_set_r0-pf1.npy"
    pred_path = "../inf-data/binary-p_v_preds_test_set_r0-pf1.npy"
    test_with_your_data(gt_path, pred_path, start_idx=16462, load_length=2000)

