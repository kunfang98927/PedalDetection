import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

# Import your segmenter (assuming the file is named 'gesture2.py')
from gesture import SimplePedalGestureSegmenter, plot_simple_segmentation

class FFTGestureComparisonAnalyzer:
    """
    Compares ground truth and prediction signals using gesture boundaries from ground truth.
    
    For each GT segment boundary, computes:
    - FFT-based MSE using reconstructed signals from limited frequency coefficients
    - FFT coefficient magnitudes for both GT and prediction signals
    - All scores weighted by segment length relative to total relevant signal length
    """
    
    def __init__(self, gt_threshold: float = 0.01, pred_threshold: float = 0.05):
        self.gt_threshold = gt_threshold
        self.pred_threshold = pred_threshold
        
    def compute_segment_metrics(self, gt_signal: np.ndarray, pred_signal: np.ndarray, 
                              start_idx: int, end_idx: int, n_coefficients: int = 11) -> Dict:
        """
        Compute MSE using FFT reconstruction with specified number of coefficients.
        
        Process:
        1. Extract segment portions from both signals
        2. Apply FFT to each segment
        3. Keep only the first n_coefficients (low-frequency components)
        4. Reconstruct signals using inverse FFT
        5. Compute MSE between reconstructed signals
        
        Args:
            gt_signal: Ground truth signal portion
            pred_signal: Prediction signal portion
            start_idx: Start index of segment
            end_idx: End index of segment (inclusive)
            n_coefficients: Number of FFT coefficients to keep for reconstruction (default: 11)
            
        Returns:
            Dictionary with FFT-based MSE and coefficient magnitudes
        """
        # Extract segment portions (same indices = same length)
        gt_segment = gt_signal[start_idx:end_idx+1]
        pred_segment = pred_signal[start_idx:end_idx+1]
        
        segment_len = len(gt_segment)
        
        if segment_len == 0:
            return {
                'fft_mse': 0.0,
                'segment_length': 0,
                'gt_values': [],
                'pred_values': []
            }
        
        # Handle edge case where segment is shorter than requested coefficients
        effective_n_coeffs = min(n_coefficients, segment_len)
        
        # Apply FFT to both segments
        gt_fft = np.fft.fft(gt_segment)
        pred_fft = np.fft.fft(pred_segment)
        
        # Keep only the first n_coefficients (low-frequency components)
        # Zero out higher frequency components
        gt_fft_filtered = np.zeros_like(gt_fft)
        pred_fft_filtered = np.zeros_like(pred_fft)
        
        # Keep DC component and first (n_coefficients-1)/2 positive and negative frequencies
        # For n_coefficients=11: keep DC + 5 positive + 5 negative frequency components
        if effective_n_coeffs == 1:
            # Only DC component
            gt_fft_filtered[0] = gt_fft[0]
            pred_fft_filtered[0] = pred_fft[0]
        else:
            # DC component
            gt_fft_filtered[0] = gt_fft[0]
            pred_fft_filtered[0] = pred_fft[0]
            
            # Positive frequencies
            n_pos_freq = (effective_n_coeffs - 1) // 2
            gt_fft_filtered[1:n_pos_freq+1] = gt_fft[1:n_pos_freq+1]
            pred_fft_filtered[1:n_pos_freq+1] = pred_fft[1:n_pos_freq+1]
            
            # Corresponding negative frequencies (for real signal symmetry)
            if n_pos_freq > 0:
                gt_fft_filtered[-n_pos_freq:] = gt_fft[-n_pos_freq:]
                pred_fft_filtered[-n_pos_freq:] = pred_fft[-n_pos_freq:]
        
        # Reconstruct signals using inverse FFT
        gt_reconstructed = np.real(np.fft.ifft(gt_fft_filtered))
        pred_reconstructed = np.real(np.fft.ifft(pred_fft_filtered))
        
        # Compute MSE between reconstructed signals
        fft_mse = np.mean((gt_reconstructed - pred_reconstructed) ** 2)
        
        # Extract the coefficients that were kept (for analysis/debugging)
        kept_coefficients_gt = []
        kept_coefficients_pred = []
        
        # Store DC component
        kept_coefficients_gt.append(gt_fft[0])
        kept_coefficients_pred.append(pred_fft[0])
        
        # Store positive frequency components that were kept
        if effective_n_coeffs > 1:
            n_pos_freq = (effective_n_coeffs - 1) // 2
            kept_coefficients_gt.extend(gt_fft[1:n_pos_freq+1])
            kept_coefficients_pred.extend(pred_fft[1:n_pos_freq+1])
        
        return {
            'fft_mse': fft_mse,
            'segment_length': segment_len,
            'gt_values': [abs(c) for c in kept_coefficients_gt],
            'pred_values': [abs(c) for c in kept_coefficients_pred]
        }
        
    
    def analyze_signals(self, gt_signal: np.ndarray, pred_signal: np.ndarray, 
                       segmenter_gt, segmenter_pred=None, n_coefficients: int = 11) -> Dict:
        """
        Complete analysis comparing GT and predictions using GT segment boundaries.
        
        Args:
            gt_signal: Ground truth signal
            pred_signal: Prediction signal
            segmenter_gt: Configured segmenter for ground truth
            segmenter_pred: Optional segmenter for predictions (for visualization)
            n_coefficients: Number of FFT coefficients to use for reconstruction
            
        Returns:
            Dictionary with analysis results per gesture type and overall
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING SIGNAL COMPARISON (FFT-based with {n_coefficients} coefficients)")
        print(f"{'='*60}")
        
        # Get GT segments (these define our boundaries)
        gt_segments = segmenter_gt.segment_signal(gt_signal)
        
        # Optional: get pred segments for visualization
        pred_segments = None
        if segmenter_pred is not None:
            pred_segments = segmenter_pred.segment_signal(pred_signal)
        
        # Initialize results structure
        results = {
            'overall': defaultdict(list),
            'by_type': defaultdict(lambda: defaultdict(list)),
            'segment_details': []
        }
        
        # Filter to only gesture segments (not plain)
        # gesture_segments = [s for s in gt_segments if s['classification'] != 'plain']
        gesture_segments = gt_segments
        total_gesture_frames = sum(s['duration'] for s in gesture_segments)
        
        print(f"Total gesture segments: {len(gesture_segments)}")
        print(f"Total gesture frames: {total_gesture_frames}")

        # Calculate total frames per gesture type for proper weighting
        frames_by_type = defaultdict(int)
        for segment in gesture_segments:
            frames_by_type[segment['classification']] += segment['duration']
        
        # Analyze each gesture segment
        for i, segment in enumerate(gesture_segments):
            start_idx = segment['start']
            end_idx = segment['end']
            gesture_type = segment['classification']
            
            print(f"\nAnalyzing segment {i+1}/{len(gesture_segments)}: {gesture_type} "
                  f"(frames {start_idx}-{end_idx}, duration: {segment['duration']})")
            
            # Compute metrics for this segment
            metrics = self.compute_segment_metrics(gt_signal, pred_signal, start_idx, end_idx, n_coefficients)
            
            # Weight by segment length for overall results
            overall_weight = segment['duration'] / total_gesture_frames
            overall_weighted_mse = metrics['fft_mse'] * overall_weight
            
            # Weight by segment length within gesture type for by-type results
            type_weight = segment['duration'] / frames_by_type[gesture_type]
            type_weighted_mse = metrics['fft_mse'] * type_weight

            
            # Store detailed results
            segment_result = {
                'segment_id': i,
                'type': gesture_type,
                'start': start_idx,
                'end': end_idx,
                'duration': segment['duration'],
                'overall_weight': overall_weight,
                'type_weight': type_weight,
                'fft_mse': metrics['fft_mse'],
                'overall_weighted_mse': overall_weighted_mse,
                'type_weighted_mse': type_weighted_mse,
                'gt_values': metrics['gt_values'],
                'pred_values': metrics['pred_values']
            }
            results['segment_details'].append(segment_result)
            
            # Add to overall and by-type results
            results['overall']['weighted_mse'].append(overall_weighted_mse)
            results['overall']['raw_mse'].append(metrics['fft_mse'])
            results['overall']['duration'].append(segment['duration'])
            
            results['by_type'][gesture_type]['weighted_mse'].append(type_weighted_mse)
            results['by_type'][gesture_type]['raw_mse'].append(metrics['fft_mse'])
            results['by_type'][gesture_type]['duration'].append(segment['duration'])
            
            # Print segment summary
            print(f"  GT FFT magnitudes:   {[f'{v:.4f}' for v in metrics['gt_values']]}")
            print(f"  Pred FFT magnitudes: {[f'{v:.4f}' for v in metrics['pred_values']]}")
            print(f"  FFT MSE: {metrics['fft_mse']:.6f}")
            print(f"  Overall Weighted MSE: {overall_weighted_mse:.6f} (weight: {overall_weight:.4f})")
            print(f"  Type Weighted MSE: {type_weighted_mse:.6f} (weight: {type_weight:.4f})")
        
        # Compute summary statistics
        results['summary'] = self._compute_summary_stats(results)
        
        return results, gt_segments, pred_segments
    
    def _compute_summary_stats(self, results: Dict) -> Dict:
        """Compute summary statistics from weighted results."""
        summary = {
            'overall': {},
            'by_type': {}
        }
        
        # Overall summary
        if results['overall']['weighted_mse']:
            summary['overall']['weighted_total_mse'] = np.sum(results['overall']['weighted_mse'])
            summary['overall']['average_raw_mse'] = np.mean(results['overall']['raw_mse'])
            summary['overall']['total_segments'] = len(results['overall']['weighted_mse'])
            summary['overall']['total_frames'] = np.sum(results['overall']['duration'])
        
        # By-type summary
        for gesture_type, type_data in results['by_type'].items():
            if type_data['weighted_mse']:
                summary['by_type'][gesture_type] = {
                    'weighted_total_mse': np.sum(type_data['weighted_mse']),
                    'average_raw_mse': np.mean(type_data['raw_mse']),
                    'total_segments': len(type_data['weighted_mse']),
                    'total_frames': np.sum(type_data['duration'])
                }
        
        return summary
    
    def print_summary(self, results: Dict):
        """Print a comprehensive summary of the analysis results."""
        print(f"\n{'='*80}")
        print(f"FFT-BASED MSE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        summary = results['summary']
        
        # Overall results
        print(f"\nOVERALL RESULTS (FFT MSE weighted by segment duration):")
        print(f"-" * 60)
        overall = summary['overall']
        
        if 'weighted_total_mse' in overall:
            print(f"Weighted Total MSE: {overall['weighted_total_mse']:.6f}")
            print(f"Average Raw MSE:    {overall['average_raw_mse']:.6f}")
            print(f"Total Segments:     {overall['total_segments']}")
            print(f"Total Frames:       {overall['total_frames']}")
        
        # By gesture type
        print(f"\nRESULTS BY GESTURE TYPE:")
        print(f"-" * 60)
        
        for gesture_type, type_data in summary['by_type'].items():
            print(f"\n{gesture_type.upper()}:")
            print(f"  Weighted Total MSE: {type_data['weighted_total_mse']:.6f}")
            print(f"  Average Raw MSE:    {type_data['average_raw_mse']:.6f}")
            print(f"  Segments:           {type_data['total_segments']}")
            print(f"  Frames:             {type_data['total_frames']}")
        
        # Segment distribution summary
        print(f"\nSEGMENT DISTRIBUTION:")
        print(f"-" * 30)
        type_counts = defaultdict(int)
        total_frames_by_type = defaultdict(int)
        
        for segment in results['segment_details']:
            gesture_type = segment['type']
            type_counts[gesture_type] += 1
            total_frames_by_type[gesture_type] += segment['duration']
        
        total_segments = sum(type_counts.values())
        total_frames = sum(total_frames_by_type.values())
        
        for gesture_type in sorted(type_counts.keys()):
            count = type_counts[gesture_type]
            frames = total_frames_by_type[gesture_type]
            pct_segments = count / total_segments * 100
            pct_frames = frames / total_frames * 100
            print(f"{gesture_type:10s}: {count:3d} segments ({pct_segments:5.1f}%), "
                  f"{frames:4d} frames ({pct_frames:5.1f}%)")
        
        print(f"{'Total':10s}: {total_segments:3d} segments, {total_frames:4d} frames")
        
        print(f"\nNOTE: MSE computed from reconstructed signals using FFT with limited coefficients")
        print(f"Weighted Total MSE is the sum of (segment_MSE × segment_duration/total_duration)")

    def plot_comparison(self, gt_signal: np.ndarray, pred_signal: np.ndarray, 
                       gt_segments: List[Dict], pred_segments: List[Dict] = None,
                       start_frame: int = 0, end_frame: int = None,
                       title_prefix: str = "Signal Comparison"):
        """
        Plot GT and prediction signals with their respective segment boundaries.
        GT plot shows GT segmentation, Pred plot shows Pred segmentation.
        """
        if end_frame is None:
            end_frame = max(len(gt_signal),2000)
            
        frames = np.arange(start_frame, min(end_frame, len(gt_signal)))
        gt_portion = gt_signal[start_frame:end_frame]
        pred_portion = pred_signal[start_frame:end_frame]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Colors for different gesture types
        colors = {
            'highland': '#F9EBC5',   # Light cream
            'mountain': '#CCDD5E',   # Light green  
            'pinnacle': '#798A58',   # Dark green
            'hill': '#F0BB78',       # Light orange
            'plain': '#ECF0F1'       # Light gray
        }
        
        # Plot ground truth with GT segmentation
        ax1.plot(frames, gt_portion, 'k-', linewidth=2, alpha=0.8, label='Ground Truth')
        ax1.axhline(y=self.gt_threshold, color='red', linestyle='--', alpha=0.5, 
                   label=f'GT Threshold {self.gt_threshold}')
        ax1.set_ylabel('Pedal Value', fontsize=12)
        ax1.set_title(f'{title_prefix} - Ground Truth (GT Segmentation)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.1)
        
        # Plot predictions with Pred segmentation
        ax2.plot(frames, pred_portion, 'b-', linewidth=2, alpha=0.8, label='Predictions')
        ax2.axhline(y=self.pred_threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'Pred Threshold {self.pred_threshold}')
        ax2.set_ylabel('Pedal Value', fontsize=12)
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_title(f'{title_prefix} - Predictions (Pred Segmentation)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.1)
        
        # Add GT segment markings to GT plot
        gt_segments_in_range = [s for s in gt_segments 
                               if s['start'] < end_frame and s['end'] >= start_frame]
        
        for segment in gt_segments_in_range:
            s_start = max(segment['start'], start_frame)
            s_end = min(segment['end'], end_frame - 1)
            
            if s_start <= s_end:
                color = colors.get(segment['classification'], '#CCCCCC')
                
                # Add colored background to GT plot
                ax1.axvspan(s_start, s_end, alpha=0.3, color=color, 
                           edgecolor='black', linewidth=0.5)
                
                # Add label to GT plot
                mid_point = (s_start + s_end) / 2
                y_pos = np.max(gt_signal[s_start:s_end+1]) + 0.05
                
                ax1.text(mid_point, y_pos, segment['classification'], 
                        ha='center', va='bottom', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                alpha=0.8, edgecolor='black'))
        
        # Add Pred segment markings to Pred plot (if available)
        if pred_segments is not None:
            pred_segments_in_range = [s for s in pred_segments 
                                     if s['start'] < end_frame and s['end'] >= start_frame]
            
            for segment in pred_segments_in_range:
                s_start = max(segment['start'], start_frame)
                s_end = min(segment['end'], end_frame - 1)
                
                if s_start <= s_end:
                    color = colors.get(segment['classification'], '#CCCCCC')
                    
                    # Add colored background to Pred plot
                    ax2.axvspan(s_start, s_end, alpha=0.3, color=color, 
                               edgecolor='black', linewidth=0.5)
                    
                    # Add label to Pred plot
                    mid_point = (s_start + s_end) / 2
                    y_pos = np.max(pred_signal[s_start:s_end+1]) + 0.05
                    
                    ax2.text(mid_point, y_pos, segment['classification'], 
                            ha='center', va='bottom', fontsize=10, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                    alpha=0.8, edgecolor='black'))
        
        # Add legends
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


def compare_gt_pred_signals_fft(gt_path: str, pred_path: str, 
                               start_idx: int = 0, 
                               load_length: int = 500,
                               gt_threshold: float = 0.01, pred_threshold: float = 0.05,
                               n_coefficients: int = 11,
                               if_plot: bool = True):
    """
    Main function to compare ground truth and prediction signals using FFT-based analysis.
    
    Args:
        gt_path: Path to ground truth .npy file
        pred_path: Path to prediction .npy file  
        start_idx: Start frame to analyze
        load_length: Number of frames to analyze
        gt_threshold: Threshold for GT segmentation
        pred_threshold: Threshold for prediction segmentation
        n_coefficients: Number of FFT coefficients to use for reconstruction
        if_plot: Whether to generate plots
    """ 
    # Load data
    gt_data = np.load(gt_path)
    pred_data = np.load(pred_path)
    
    print(f"Loaded data shapes: GT {gt_data.shape}, Pred {pred_data.shape}")
    
    gt_signal = gt_data
    pred_signal = pred_data
        
    if load_length < 0 and start_idx == 0:
        load_length = len(gt_signal)
    # Extract the portion to analyze
    gt_signal = gt_signal[start_idx:start_idx+load_length]
    pred_signal = pred_signal[start_idx:start_idx+load_length]
    
    print(f"Processing frames {start_idx} to {start_idx+load_length}")
    print(f"GT range: [{np.min(gt_signal):.3f}, {np.max(gt_signal):.3f}]")
    print(f"Pred range: [{np.min(pred_signal):.3f}, {np.max(pred_signal):.3f}]")
    
    # Create segmenters
    segmenter_gt = SimplePedalGestureSegmenter(
        threshold=gt_threshold,
        min_cycle_duration=3,
        print_out=False
    )
    
    segmenter_pred = SimplePedalGestureSegmenter(
        threshold=pred_threshold,
        min_cycle_duration=3,
        print_out=False
    )
    
    # Create comparison analyzer
    analyzer = FFTGestureComparisonAnalyzer(gt_threshold, pred_threshold)
    
    # Perform analysis
    results, gt_segments, pred_segments = analyzer.analyze_signals(
        gt_signal, pred_signal, segmenter_gt, segmenter_pred, n_coefficients
    )
    
    # Print summary
    analyzer.print_summary(results)
    
    if if_plot:
        # Plot comparison
        analyzer.plot_comparison(gt_signal, pred_signal, gt_segments, pred_segments,
                                title_prefix=f"Signal frames {start_idx} to {start_idx+load_length} Comparison")
    
    return results, gt_segments, pred_segments, analyzer
        

if __name__ == "__main__":
    
    # With real data:
    gt_path = "../inf-data/p_v_labels_test_set_r0-pf1.npy" # ground truth
    pred_path = "../inf-data/binary-p_v_preds_test_set_r0-pf1.npy" # binary model result
    # pred_path = "../inf-data/a-p_v_preds_test_set_r0-pf1.npy" # audio model result
    # pred_path = "../inf-data/a+m-p_v_preds_test_set_r0-pf1.npy" # audio+midi model result
    
    results = compare_gt_pred_signals_fft(
        gt_path=gt_path,
        pred_path=pred_path,
        start_idx=00,
        load_length=-500,
        gt_threshold=0.01,
        pred_threshold=0.05,
        n_coefficients=11,
        if_plot=False
    )
    
    if results is not None:
        results_data, gt_segments, pred_segments, analyzer = results
        print("\nFFT-based analysis complete! Check the printed summary and plots above.")