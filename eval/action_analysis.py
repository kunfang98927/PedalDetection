import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


def deferred_label_context(x, 
                        slope_threshold=0.01, 
                        window_size=5,
                        min_r_squared=0.5,
                        plot=True, 
                        print_out=False):
    """
    Simple sliding window linear regression approach for state classification.
    
    Parameters:
    - slope_threshold: Threshold for classifying slopes as significant
    - window_size: Size of sliding window (should be odd)
    - min_r_squared: Minimum R² to trust the regression slope
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    states = np.zeros(n, dtype=int)
    
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2

    # Window size shrinks near signal boundaries (e.g., only 10 points at endpoints with window_size=19)
    # This creates asymmetric windows that may reduce regression reliability at edges.
    
    for t in tqdm(range(n)):
        # Define window bounds centered on current point
        start_idx = max(0, t - half_window)
        end_idx = min(n, t + half_window + 1)
        
        # Get window data
        window_indices = np.arange(start_idx, end_idx)
        window_values = x[start_idx:end_idx]
        
        if len(window_indices) < 3:  # Need at least 3 points for regression
            states[t] = 0
            continue
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(window_indices, window_values)
        r_squared = r_value**2
        
        if print_out:
            print(f"t={t}, window=[{start_idx}:{end_idx}], slope={slope:.4f}, R²={r_squared:.3f}")
        
        # Classify based on slope if regression is reliable
        if r_squared >= min_r_squared:
            if abs(slope) < slope_threshold:
                states[t] = 0  # Flat/stable
            elif slope > 0:
                states[t] = 1  # Rising
            else:
                states[t] = -1  # Falling
        else:
            # If regression is unreliable, default to stable
            states[t] = 0
    
    if print_out:
        print('\nSignal:', x)
        print('States:', states)
    
    if plot:
        create_state_plot(x, states)
    
    return states


def create_state_plot(x, states):
    """Create visualization of signal with state-colored background."""
    colors = {1: 'green', 0: 'lightgray', -1: 'red'}
    n = len(x)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Main signal plot with state-colored background
    ax1.plot(range(n), x, color='black', linewidth=2, zorder=5)
    
    # Add state-colored background
    for i in range(n):
        if i < n-1:
            ax1.axvspan(i-0.4, i+0.4, color=colors[states[i]], alpha=0.3)
        else:
            ax1.axvspan(i-0.4, i, color=colors[states[i]], alpha=0.3)
    
    ax1.scatter(range(n), x, color='black', zorder=6, s=30)
    ax1.set_title("Signal with Sliding Window Linear Regression States")
    ax1.set_ylabel("Signal Value")
    ax1.grid(True, alpha=0.3)
    
    # States plot
    state_colors = [colors[state] for state in states]
    ax2.scatter(range(n), states, c=state_colors, s=50, zorder=5)
    ax2.plot(range(n), states, color='blue', alpha=0.5, linewidth=1)
    ax2.set_ylabel('State')
    ax2.set_xlabel('Time')
    ax2.set_title('State Sequence (-1: Fall, 0: Flat, 1: Rise)')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_metrics(ground_truth, predictions, states=[1, 0, -1]):
    recalls = {}
    precisions = {}
    f1s = {}
    counts_gt = {}
    counts_pred = {}
    
    for state in states:
        # True positives for this state
        tp = np.sum((ground_truth == state) & (predictions == state))
        
        # Ground-truth count (denominator for recall)
        gt_count = np.sum(ground_truth == state)
        counts_gt[state] = gt_count
        
        # Predicted count (denominator for precision)
        pred_count = np.sum(predictions == state)
        counts_pred[state] = pred_count
        
        # Recall
        recall = tp / gt_count if gt_count > 0 else 0
        recalls[state] = recall
        
        # Precision
        precision = tp / pred_count if pred_count > 0 else 0
        precisions[state] = precision
        
        # F1
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        f1s[state] = f1

    # Weighted averages
    total_gt = sum(counts_gt.values())
    total_pred = sum(counts_pred.values())
    
    weighted_recall = (
        sum(recalls[s] * counts_gt[s] for s in states) / total_gt
        if total_gt > 0 else 0
    )
    weighted_precision = (
        sum(precisions[s] * counts_gt[s] for s in states) / total_gt
        if total_gt > 0 else 0
    )
    weighted_f1 = (
        sum(f1s[s] * counts_gt[s] for s in states) / total_gt
        if total_gt > 0 else 0
    )
    
    # Macro averages (simple average across classes)
    macro_recall = np.mean(list(recalls.values()))
    macro_precision = np.mean(list(precisions.values()))
    macro_f1 = np.mean(list(f1s.values()))
    
    # Overall metrics (micro-averaged)
    # Total true positives across all classes
    total_tp = np.sum(ground_truth == predictions)
    

    # Note: For complete point-by-point classification, 
    # overall precision/recall/F1 will be identical since len(predictions) == len(ground_truth). 

    # Overall precision = total correct predictions / total predictions
    overall_precision = total_tp / len(predictions) if len(predictions) > 0 else 0
    
    # Overall recall = total correct predictions / total ground truth
    overall_recall = total_tp / len(ground_truth) if len(ground_truth) > 0 else 0
    
    # Overall F1 (same as overall precision/recall since they're equal in multiclass)
    overall_f1 = overall_precision  # This equals overall_recall in complete multiclass classification
    
    
    return {
        "recall_per_class": recalls,
        "precision_per_class": precisions,
        "f1_per_class": f1s,
        "weighted_recall": weighted_recall,
        "weighted_precision": weighted_precision,
        "weighted_f1": weighted_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
    }

def compare_gt_vs_pred(gt_path, pred_path, start_idx=0, duration=1000, 
                      gt_slope_threshold=0.01, gt_window_size=5,
                      pred_slope_threshold=0.01, pred_window_size=5,
                      gt_min_r_square=0.5, pred_min_r_square=0.5,
                      plot=True, print_out=False):
    """
    Load ground truth and predictions, apply sliding window labeling, and plot comparison.
    
    Args:
        gt_path: Path to ground truth .npy file
        pred_path: Path to predictions .npy file  
        start_idx: Starting index for extraction
        duration: Number of frames to extract
        gt_slope_threshold: Slope threshold for ground truth
        gt_window_size: Window size for ground truth
        pred_slope_threshold: Slope threshold for predictions
        pred_window_size: Window size for predictions
    """
    
    # Load the data
    gt_data = np.load(gt_path)
    pred_data = np.load(pred_path)
    
    print(f"Ground truth shape: {gt_data.shape}")
    print(f"Predictions shape: {pred_data.shape}")
    
    if duration > 0:
        gt_signal = gt_data[start_idx:start_idx+duration]
        pred_signal = pred_data[start_idx:start_idx+duration]
        print(f"Frame {start_idx}-{start_idx+duration} - GT range: [{gt_signal.min():.3f}, {gt_signal.max():.3f}]")
        print(f"Frame {start_idx}-{start_idx+duration} - Pred range: [{pred_signal.min():.3f}, {pred_signal.max():.3f}]")
    else:
        gt_signal = gt_data
        pred_signal = pred_data
    
    # Apply sliding window labeling to both signals
    gt_states = deferred_label_context(gt_signal, gt_slope_threshold, gt_window_size, gt_min_r_square,\
                                       plot=False, print_out=print_out)
    pred_states = deferred_label_context(pred_signal, pred_slope_threshold, pred_window_size, pred_min_r_square,\
                                         plot=False, print_out=print_out)

    if plot:
        # Create the comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Colors for states
        colors = {1:'green', 0:'lightgray', -1:'red'}
        
        # Plot ground truth
        plot_signal_with_states(ax1, gt_signal, gt_states, colors, "Ground Truth")
        
        # Plot predictions  
        plot_signal_with_states(ax2, pred_signal, pred_states, colors, "Predictions")
        
        # Set common x-axis
        ax2.set_xlabel('Time Steps')
        plt.tight_layout()
        plt.show()
    
    return gt_signal, pred_signal, gt_states, pred_states


def plot_signal_with_states(ax, signal, states, colors, title):
    """
    Plot signal with background colored by state.
    """
    n = len(signal)
    
    # Plot background colors
    start = 0
    current_state = states[0]
    for i in range(1, n):
        if states[i] != current_state:
            ax.axvspan(start-1, i-1, color=colors[current_state], alpha=0.3)
            start = i
            current_state = states[i]

    # Last segment
    ax.axvspan(start-1, n-1, color=colors[current_state], alpha=0.3)
    
    # Plot signal curve
    ax.plot(range(n), signal, color='black', linewidth=2, zorder=5)
    ax.set_title(f"{title} (1:green, 0:lightgray, -1:red)")
    ax.set_ylabel('Signal Value')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, n, 50))
    ax.set_ylim(-0.05, 1.05)

    return True

def calculate_signal_metrics(gt_signal, pred_signal):
    """
    Calculate MSE and MAE between ground truth and predicted signals.
    
    Args:
        gt_signal: Ground truth signal (continuous values)
        pred_signal: Predicted signal (continuous values)
    
    Returns:
        dict: Dictionary containing MSE and MAE values
    """
    # Ensure signals are numpy arrays
    gt_signal = np.array(gt_signal)
    pred_signal = np.array(pred_signal)
    
    # Check if signals have the same length
    if len(gt_signal) != len(pred_signal):
        raise ValueError(f"Signal lengths don't match: gt={len(gt_signal)}, pred={len(pred_signal)}")
    
    # Calculate differences
    errors = gt_signal - pred_signal
    
    # Mean Squared Error
    mse = np.mean(errors ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(errors))
    
    # Root Mean Squared Error (bonus metric)
    rmse = np.sqrt(mse)
    
    # Additional statistics
    max_error = np.max(np.abs(errors))
    min_error = np.min(np.abs(errors))
    std_error = np.std(errors)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "max_absolute_error": max_error,
        "min_absolute_error": min_error,
        "error_std": std_error,
        "signal_length": len(gt_signal)
    }

if __name__ == "__main__":
    
    tricky_signals = {
        # 1. Oscillating around threshold - should it commit to a state?
        "threshold_oscillation": [0, 0.009, 0.008, 0.011, 0.009, 0.012, 0.008],
        
        # 2. Slow drift that eventually hits cumulative threshold
        "slow_drift": [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1],
        
        # 3. Fake-out: looks like trend reversal but continues original direction
        "fake_reversal": [0, 0.05, 0.1, 0.15, 0.2, 0.18, 0.19, 0.25, 0.3, 0.35],
        
        # 4. Multiple state changes in quick succession
        "rapid_changes": [0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15],
        
        # 5. Long plateau after trend - when should it become neutral?
        "trend_then_long_plateau": [0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        
        # 6. Staircase pattern - each step just below individual threshold
        "subtle_staircase": [0, 0.009, 0.018, 0.027, 0.036, 0.045, 0.054, 0.063, 0.072, 0.081, 0.09, 0.099, 0.108],
        
        # 7. V-shaped pattern - down then up
        "v_shape": [0.5, 0.4, 0.3, 0.2, 0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
        
        # 8. Noise around a trend
        "noisy_trend": [0, 0.02, 0.01, 0.04, 0.03, 0.06, 0.05, 0.08, 0.07, 0.1, 0.09, 0.12],
        
        # 9. Big jump followed by exact reversal
        "jump_and_reverse": [0, 0.2, 0],
        
        # 10. Cumulative change that crosses zero
        "crossing_zero": [0.1, 0.05, 0, -0.05, -0.1, -0.05, 0, 0.05, 0.1],
        
        # 11. Barely-threshold changes
        "edge_cases": [0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # exactly 0.1 total
        
        # 12. Mixed signals
        "conflicting_signals": [0, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10],  # big jump up, then slow decay
    }

    # Test with a few signals
    # for name, sig in list(tricky_signals.items())[:3]:
    #     print(f"\n{name}:")
    #     states = deferred_label_context(sig, slope_threshold=0.01, window_size=3, plot=True, print_out=True)
    #     print(f"Signal: {sig}")
    #     print(f"States: {states}")

    # With real data:
    gt_path = "../inf-data/p_v_labels_test_set_r0-pf1.npy" # ground truth
    pred_path = "../inf-data/binary-p_v_preds_test_set_r0-pf1.npy" # binary model result
    # pred_path = "../inf-data/a-p_v_preds_test_set_r0-pf1.npy" # audio model result
    # pred_path = "../inf-data/a+m-p_v_preds_test_set_r0-pf1.npy" # audio+midi model result

    gt_signal, pred_signal, gt_states, pred_states = compare_gt_vs_pred(
        gt_path, pred_path, 
        start_idx=0000,  
        duration=-500, # Use negative or 0 for full length
        ###### Parameters below are used in our paper ######
        gt_slope_threshold=0.005,
        gt_window_size=19,
        pred_slope_threshold=0.005, 
        pred_window_size=19,
        gt_min_r_square=0.5,
        pred_min_r_square=0.5,
        plot=False,
        print_out=False
    )

    signal_metrics = calculate_signal_metrics(gt_signal, pred_signal)
    metrics = calculate_metrics(gt_states, pred_states, states=[1, 0, -1])

    # Print signal-level metrics
    print("="*50)
    print("SIGNAL-LEVEL METRICS")
    print("="*50)
    print(f"MSE: {signal_metrics['mse']:.6f}")
    print(f"MAE: {signal_metrics['mae']:.6f}")
    print(f"Signal Length: {signal_metrics['signal_length']}")
    # Print state-level metrics
    print("\n" + "="*50)
    print("STATE-LEVEL METRICS")
    print("="*50)
    print("Recall per class:")
    for class_id, recall in metrics["recall_per_class"].items():
        print(f"  Class {class_id}: {recall:.4f}")

    print("\nPrecision per class:")
    for class_id, precision in metrics["precision_per_class"].items():
        print(f"  Class {class_id}: {precision:.4f}")

    print("\nF1 per class:")
    for class_id, f1 in metrics["f1_per_class"].items():
        print(f"  Class {class_id}: {f1:.4f}")

    print(f"\nWeighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    print(f"\nMacro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    