"""
Full evaluation pipeline for MPHY0043 Coursework.

Evaluates all trained models and generates comparison statistics.

Usage:
    python -m mphy0043_cw.evaluation.evaluate --config mphy0043_cw/config.yaml --task A
    python -m mphy0043_cw.evaluation.evaluate --config mphy0043_cw/config.yaml --task B
    python -m mphy0043_cw.evaluation.evaluate --config mphy0043_cw/config.yaml --task all
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mphy0043_cw.evaluation.metrics import (
    compute_time_prediction_metrics,
    compute_tool_detection_metrics,
    compute_improvement,
    compute_per_tool_improvement,
    paired_t_test,
    format_metrics_table,
    TOOL_NAMES
)
from mphy0043_cw.data.dataset import (
    create_dataset, load_video_data,
    TEST_VIDEO_IDS, VAL_VIDEO_IDS,
    NUM_PHASES, NUM_TOOLS
)


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        content = f.read()

    # Handle potential merge conflict markers
    if '<<<<<<< HEAD' in content:
        # Take the HEAD version (everything before =======)
        content = content.split('=======')[0]
        content = content.replace('<<<<<<< HEAD\n', '')

    return yaml.safe_load(content)


# ============================================================================
# TASK A: TIME PREDICTION EVALUATION
# ============================================================================

def evaluate_time_predictor(
    model,
    data_dir: str,
    video_ids: List[int],
    timing_labels: Dict,
    batch_size: int = 8
) -> Dict:
    """
    Evaluate time prediction model on a set of videos.

    Args:
        model: Trained time predictor model
        data_dir: Path to Cholec80 data directory
        video_ids: List of video IDs to evaluate
        timing_labels: Pre-computed timing labels
        batch_size: Batch size for inference

    Returns:
        Dictionary with all evaluation metrics
    """
    all_y_true = []
    all_y_pred = []
    all_phases = []
    all_video_ids = []

    print(f"Evaluating on {len(video_ids)} videos...")

    for video_id in video_ids:
        try:
            # Load video data
            video_data = load_video_data(data_dir, video_id)

            if video_id not in timing_labels:
                print(f"  Warning: No timing labels for video {video_id}")
                continue

            labels = timing_labels[video_id]
            n_frames = len(video_data['frame_ids'])

            # Create dataset for this video
            ds = create_dataset(
                data_dir=data_dir,
                video_ids=[video_id],
                batch_size=batch_size,
                shuffle=False,
                include_timing=True,
                timing_labels=timing_labels
            )

            # Collect predictions
            video_y_true = []
            video_y_pred = []
            video_phases = []

            for batch in ds:
                # Get inputs
                frames = batch['frame']
                phases = batch['phase']
                elapsed = batch.get('elapsed_phase', tf.zeros_like(phases, dtype=tf.float32))

                # For time predictor, we need sequence format
                # Add sequence dimension for single-frame evaluation
                frames_seq = tf.expand_dims(frames, axis=1)  # (batch, 1, H, W, C)
                elapsed_seq = tf.expand_dims(tf.expand_dims(elapsed, axis=-1), axis=1)  # (batch, 1, 1)
                phases_seq = tf.expand_dims(phases, axis=1)  # (batch, 1)

                # Make prediction
                inputs = {
                    'frames': frames_seq,
                    'elapsed_phase': elapsed_seq,
                    'phase': phases_seq
                }
                predictions = model(inputs, training=False)
                # Extract remaining_phase prediction (first output)
                if isinstance(predictions, dict):
                    pred_remaining = predictions['remaining_phase']
                elif isinstance(predictions, (list, tuple)):
                    pred_remaining = predictions[0]
                else:
                    pred_remaining = predictions

                # Get ground truth
                true_remaining = batch['remaining_phase']

                video_y_true.extend(true_remaining.numpy().flatten())
                video_y_pred.extend(pred_remaining.numpy().flatten())
                video_phases.extend(phases.numpy().flatten())

            all_y_true.extend(video_y_true)
            all_y_pred.extend(video_y_pred)
            all_phases.extend(video_phases)
            all_video_ids.extend([video_id] * len(video_y_true))

            print(f"  Video {video_id:02d}: {len(video_y_true)} frames")

        except Exception as e:
            print(f"  Error evaluating video {video_id}: {e}")
            continue

    # Convert to arrays
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    phases = np.array(all_phases)

    print(f"\nTotal samples: {len(y_true)}")

    # Compute metrics
    metrics = compute_time_prediction_metrics(y_true, y_pred, phases)

    # Add per-video metrics for statistical testing
    video_maes = []
    for vid in sorted(set(all_video_ids)):
        mask = np.array(all_video_ids) == vid
        if np.sum(mask) > 0:
            video_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            video_maes.append(video_mae)
    metrics['per_video_mae'] = video_maes

    return metrics


# ============================================================================
# TASK B: TOOL DETECTION EVALUATION
# ============================================================================

def evaluate_tool_detector(
    model,
    data_dir: str,
    video_ids: List[int],
    batch_size: int = 8,
    timing_labels: Optional[Dict] = None,
    use_timing: bool = False,
    time_predictor_model=None
) -> Dict:
    """
    Evaluate tool detection model on a set of videos.

    Args:
        model: Trained tool detector model
        data_dir: Path to Cholec80 data directory
        video_ids: List of video IDs to evaluate
        batch_size: Batch size for inference
        timing_labels: Pre-computed timing labels (for oracle mode)
        use_timing: Whether to use timing inputs
        time_predictor_model: Time predictor for predicted timing mode

    Returns:
        Dictionary with all evaluation metrics
    """
    all_y_true = []
    all_y_pred = []
    all_video_ids = []

    print(f"Evaluating on {len(video_ids)} videos...")

    for video_id in video_ids:
        try:
            # Create dataset
            ds = create_dataset(
                data_dir=data_dir,
                video_ids=[video_id],
                batch_size=batch_size,
                shuffle=False,
                include_timing=use_timing,
                timing_labels=timing_labels if use_timing else None
            )

            video_y_true = []
            video_y_pred = []

            for batch in ds:
                frames = batch['frame']
                tools_true = batch['instruments']

                if use_timing:
                    # Prepare timing inputs
                    if time_predictor_model is not None:
                        # Use predicted timing
                        phases = batch['phase']
                        elapsed = batch.get('elapsed_phase', tf.zeros_like(phases, dtype=tf.float32))

                        time_pred = time_predictor_model([frames, phases, elapsed], training=False)
                        if isinstance(time_pred, dict):
                            remaining_time = time_pred['remaining_phase']
                        elif isinstance(time_pred, (list, tuple)):
                            remaining_time = time_pred[0]
                        else:
                            remaining_time = time_pred

                        # Compute phase progress from predicted remaining time
                        # (This is an approximation - actual depends on total phase duration)
                        elapsed_float = tf.cast(elapsed, tf.float32)
                        total_estimated = elapsed_float + tf.maximum(remaining_time, 0.0)
                        phase_progress = elapsed_float / (total_estimated + 1.0)
                    else:
                        # Use oracle (ground truth) timing
                        remaining_time = tf.cast(batch['remaining_phase'], tf.float32)
                        phase_progress = batch['phase_progress']

                    phases = batch['phase']

                    # Make prediction with timing (use dict inputs)
                    predictions = model({
                        'frame': frames,
                        'remaining_phase': tf.expand_dims(remaining_time, axis=-1),
                        'remaining_surgery': tf.expand_dims(remaining_time, axis=-1),  # Approximate
                        'phase_progress': tf.expand_dims(phase_progress, axis=-1),
                        'phase': phases
                    }, training=False)
                else:
                    # Visual-only baseline
                    predictions = model(frames, training=False)

                video_y_true.extend(tools_true.numpy())
                video_y_pred.extend(predictions.numpy())

            all_y_true.extend(video_y_true)
            all_y_pred.extend(video_y_pred)
            all_video_ids.extend([video_id] * len(video_y_true))

            print(f"  Video {video_id:02d}: {len(video_y_true)} frames")

        except Exception as e:
            print(f"  Error evaluating video {video_id}: {e}")
            continue

    # Convert to arrays
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    print(f"\nTotal samples: {len(y_true)}")

    # Compute metrics
    metrics = compute_tool_detection_metrics(y_true, y_pred)

    # Add per-video mAP for statistical testing
    video_maps = []
    for vid in sorted(set(all_video_ids)):
        mask = np.array(all_video_ids) == vid
        if np.sum(mask) > 0:
            vid_metrics = compute_tool_detection_metrics(y_true[mask], y_pred[mask])
            video_maps.append(vid_metrics['mAP'])
    metrics['per_video_mAP'] = video_maps

    # Store raw predictions for further analysis
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred

    return metrics


# ============================================================================
# COMPARISON EVALUATION
# ============================================================================

def compare_tool_detectors(
    baseline_metrics: Dict,
    timed_oracle_metrics: Dict,
    timed_predicted_metrics: Dict
) -> Dict:
    """
    Compare baseline vs timed tool detectors.

    Args:
        baseline_metrics: Metrics from visual-only baseline
        timed_oracle_metrics: Metrics using ground-truth timing
        timed_predicted_metrics: Metrics using predicted timing

    Returns:
        Comparison results
    """
    results = {}

    # Overall mAP comparison
    results['mAP'] = {
        'baseline': baseline_metrics['mAP'],
        'timed_oracle': timed_oracle_metrics['mAP'],
        'timed_predicted': timed_predicted_metrics['mAP'],
        'improvement_oracle': timed_oracle_metrics['mAP'] - baseline_metrics['mAP'],
        'improvement_predicted': timed_predicted_metrics['mAP'] - baseline_metrics['mAP']
    }

    # Per-tool improvement
    results['per_tool'] = {}
    for tool in TOOL_NAMES:
        baseline_ap = baseline_metrics['per_tool'][tool]['ap']
        oracle_ap = timed_oracle_metrics['per_tool'][tool]['ap']
        predicted_ap = timed_predicted_metrics['per_tool'][tool]['ap']

        results['per_tool'][tool] = {
            'baseline_ap': baseline_ap,
            'oracle_ap': oracle_ap,
            'predicted_ap': predicted_ap,
            'improvement_oracle': oracle_ap - baseline_ap,
            'improvement_predicted': predicted_ap - baseline_ap
        }

    # Statistical significance
    if 'per_video_mAP' in baseline_metrics and 'per_video_mAP' in timed_predicted_metrics:
        baseline_vids = baseline_metrics['per_video_mAP']
        predicted_vids = timed_predicted_metrics['per_video_mAP']

        if len(baseline_vids) == len(predicted_vids) and len(baseline_vids) > 1:
            t_stat, p_value = paired_t_test(baseline_vids, predicted_vids)
            results['statistical_test'] = {
                'test': 'paired_t_test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_005': p_value < 0.05
            }

    return results


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def run_task_a_evaluation(config: dict, data_dir: str, checkpoint_path: str) -> Dict:
    """Run full Task A (time prediction) evaluation."""
    from mphy0043_cw.models.time_predictor import create_time_predictor

    print("\n" + "=" * 60)
    print("TASK A: TIME PREDICTION EVALUATION")
    print("=" * 60)

    # Load timing labels
    timing_labels_path = config['paths']['timing_labels_path']
    if not os.path.exists(timing_labels_path):
        print(f"Error: Timing labels not found at {timing_labels_path}")
        print("Run preprocessing first: python -m mphy0043_cw.data.preprocessing")
        return {}

    print(f"Loading timing labels from {timing_labels_path}")
    from mphy0043_cw.data.preprocessing import load_timing_labels
    timing_labels = load_timing_labels(timing_labels_path)

    # Create and load model
    print(f"Loading model from {checkpoint_path}")
    model = create_time_predictor(
        d_model=config['model']['time_predictor']['d_model'],
        d_state=config['model']['time_predictor']['d_state'],
        n_ssm_blocks=config['model']['time_predictor']['n_ssm_blocks'],
        phase_embedding_dim=config['model']['time_predictor']['phase_embedding_dim'],
        backbone_trainable_layers=config['model']['backbone_trainable_layers']
    )
    model.load_weights(checkpoint_path)

    # Evaluate on test set
    # Config already uses 1-indexed video IDs (1-80)
    test_video_ids = config['data']['test_videos']

    metrics = evaluate_time_predictor(
        model=model,
        data_dir=data_dir,
        video_ids=test_video_ids,
        timing_labels=timing_labels,
        batch_size=config['training']['batch_size']
    )

    print("\n" + format_metrics_table(metrics, task='A'))

    return metrics


def run_task_b_evaluation(config: dict, data_dir: str, checkpoint_dir: str, time_predictor_path: str = None) -> Dict:
    """Run full Task B (tool detection) evaluation."""
    from mphy0043_cw.models.tool_detector import create_tool_detector
    from mphy0043_cw.models.timed_tool_detector import create_timed_tool_detector

    print("\n" + "=" * 60)
    print("TASK B: TOOL DETECTION EVALUATION")
    print("=" * 60)

    # Load timing labels for oracle mode
    timing_labels_path = config['paths']['timing_labels_path']
    timing_labels = None
    if os.path.exists(timing_labels_path):
        from mphy0043_cw.data.preprocessing import load_timing_labels
        timing_labels = load_timing_labels(timing_labels_path)

    # Config already uses 1-indexed video IDs (1-80)
    test_video_ids = config['data']['test_videos']
    batch_size = config['training']['batch_size']

    results = {}

    # 1. Evaluate baseline (visual-only)
    print("\n--- Baseline (Visual Only) ---")
    baseline_path = os.path.join(checkpoint_dir, 'tool_detector.keras')
    if os.path.exists(baseline_path):
        baseline_model = create_tool_detector(
            backbone_trainable_layers=config['model']['backbone_trainable_layers']
        )
        baseline_model.load_weights(baseline_path)

        baseline_metrics = evaluate_tool_detector(
            model=baseline_model,
            data_dir=data_dir,
            video_ids=test_video_ids,
            batch_size=batch_size,
            use_timing=False
        )
        results['baseline'] = baseline_metrics
        print(format_metrics_table(baseline_metrics, task='B'))
    else:
        print(f"Baseline model not found at {baseline_path}")

    # 2. Evaluate timed model with oracle timing
    print("\n--- Timed Model (Oracle Timing) ---")
    timed_path = os.path.join(checkpoint_dir, 'timed_tool_detector.keras')
    if os.path.exists(timed_path) and timing_labels is not None:
        timed_model = create_timed_tool_detector(
            backbone_trainable_layers=config['model']['backbone_trainable_layers']
        )
        timed_model.load_weights(timed_path)

        oracle_metrics = evaluate_tool_detector(
            model=timed_model,
            data_dir=data_dir,
            video_ids=test_video_ids,
            batch_size=batch_size,
            timing_labels=timing_labels,
            use_timing=True,
            time_predictor_model=None  # Oracle mode
        )
        results['timed_oracle'] = oracle_metrics
        print(format_metrics_table(oracle_metrics, task='B'))
    else:
        print(f"Timed model not found or timing labels unavailable")

    # 3. Evaluate timed model with predicted timing
    print("\n--- Timed Model (Predicted Timing) ---")
    if 'timed_oracle' in results and time_predictor_path and os.path.exists(time_predictor_path):
        from mphy0043_cw.models.time_predictor import create_time_predictor

        time_model = create_time_predictor(
            d_model=config['model']['time_predictor']['d_model'],
            d_state=config['model']['time_predictor']['d_state'],
            n_ssm_blocks=config['model']['time_predictor']['n_ssm_blocks'],
            phase_embedding_dim=config['model']['time_predictor']['phase_embedding_dim'],
            backbone_trainable_layers=config['model']['backbone_trainable_layers']
        )
        time_model.load_weights(time_predictor_path)

        predicted_metrics = evaluate_tool_detector(
            model=timed_model,
            data_dir=data_dir,
            video_ids=test_video_ids,
            batch_size=batch_size,
            timing_labels=timing_labels,
            use_timing=True,
            time_predictor_model=time_model  # Predicted mode
        )
        results['timed_predicted'] = predicted_metrics
        print(format_metrics_table(predicted_metrics, task='B'))
    else:
        print(f"Time predictor not found or oracle results unavailable")

    # 4. Compare results
    if 'baseline' in results and 'timed_oracle' in results:
        print("\n--- Comparison ---")

        timed_predicted = results.get('timed_predicted', results['timed_oracle'])
        comparison = compare_tool_detectors(
            results['baseline'],
            results['timed_oracle'],
            timed_predicted
        )
        results['comparison'] = comparison

        print(f"\nmAP Improvement (Oracle):    {comparison['mAP']['improvement_oracle']:.4f}")
        print(f"mAP Improvement (Predicted): {comparison['mAP']['improvement_predicted']:.4f}")

        if 'statistical_test' in comparison:
            print(f"\nStatistical Test (Paired T-Test):")
            print(f"  t-statistic: {comparison['statistical_test']['t_statistic']:.4f}")
            print(f"  p-value:     {comparison['statistical_test']['p_value']:.4f}")
            print(f"  Significant: {comparison['statistical_test']['significant_at_005']}")

        print("\nPer-Tool AP Improvement (Predicted):")
        for tool in TOOL_NAMES:
            imp = comparison['per_tool'][tool]['improvement_predicted']
            print(f"  {tool:<15}: {imp:+.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate MPHY0043 models')
    parser.add_argument('--config', type=str, default='mphy0043_cw/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Cholec80 data directory')
    parser.add_argument('--task', type=str, choices=['A', 'B', 'all'], default='all',
                        help='Which task to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    checkpoint_dir = config['paths']['checkpoint_dir']
    time_predictor_path = os.path.join(checkpoint_dir, 'time_predictor.keras')

    all_results = {}

    if args.task in ['A', 'all']:
        if os.path.exists(time_predictor_path):
            task_a_results = run_task_a_evaluation(
                config=config,
                data_dir=args.data_dir,
                checkpoint_path=time_predictor_path
            )
            all_results['task_a'] = task_a_results
        else:
            print(f"\nTime predictor checkpoint not found at {time_predictor_path}")
            print("Train the model first: python -m mphy0043_cw.training.train_time")

    if args.task in ['B', 'all']:
        task_b_results = run_task_b_evaluation(
            config=config,
            data_dir=args.data_dir,
            checkpoint_dir=checkpoint_dir,
            time_predictor_path=time_predictor_path
        )
        all_results['task_b'] = task_b_results

    # Save results
    if args.output:
        # Remove non-serializable items
        serializable_results = {}
        for task_key, task_results in all_results.items():
            serializable_results[task_key] = {}
            for key, value in task_results.items():
                if isinstance(value, np.ndarray):
                    continue  # Skip raw arrays
                elif isinstance(value, dict):
                    # Recursively clean
                    clean_dict = {}
                    for k, v in value.items():
                        if not isinstance(v, np.ndarray):
                            clean_dict[k] = v
                    serializable_results[task_key][key] = clean_dict
                else:
                    serializable_results[task_key][key] = value

        with open(args.output, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
