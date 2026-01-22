"""
Evaluation metrics for MPHY0043 Coursework.

Provides comprehensive metrics for both tasks:
- Task A: Time prediction metrics (MAE, wMAE, within-X-min accuracy)
- Task B: Tool detection metrics (mAP, per-tool AP, precision, recall, F1)
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


# ============================================================================
# TASK A: TIME PREDICTION METRICS
# ============================================================================

def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error in frames.

    Args:
        y_true: Ground truth remaining time (frames)
        y_pred: Predicted remaining time (frames)

    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_mae_minutes(y_true, y_pred, fps=1):
    """
    Compute Mean Absolute Error in minutes.

    Args:
        y_true: Ground truth remaining time (frames)
        y_pred: Predicted remaining time (frames)
        fps: Frames per second (Cholec80 is 1 fps)

    Returns:
        MAE in minutes
    """
    mae_frames = compute_mae(y_true, y_pred)
    return mae_frames / (60 * fps)


def compute_weighted_mae(y_true, y_pred):
    """
    Compute Weighted Mean Absolute Error.

    Weights errors by inverse of remaining time - prioritizes
    accuracy for near-term predictions (more clinically relevant).

    Args:
        y_true: Ground truth remaining time
        y_pred: Predicted remaining time

    Returns:
        Weighted MAE value
    """
    errors = np.abs(y_true - y_pred)
    weights = 1.0 / (np.abs(y_true) + 1.0)
    weights = weights / np.mean(weights)  # Normalize
    return np.mean(weights * errors)


def compute_within_threshold(y_true, y_pred, threshold_minutes, fps=1):
    """
    Compute percentage of predictions within a time threshold.

    Args:
        y_true: Ground truth remaining time (frames)
        y_pred: Predicted remaining time (frames)
        threshold_minutes: Threshold in minutes
        fps: Frames per second

    Returns:
        Percentage of predictions within threshold
    """
    threshold_frames = threshold_minutes * 60 * fps
    errors = np.abs(y_true - y_pred)
    within = np.sum(errors <= threshold_frames) / len(errors)
    return within * 100  # Return as percentage


def compute_per_phase_mae(y_true, y_pred, phases):
    """
    Compute MAE per surgical phase.

    Args:
        y_true: Ground truth remaining time
        y_pred: Predicted remaining time
        phases: Phase labels for each sample

    Returns:
        Dictionary mapping phase_id to MAE
    """
    phase_maes = {}
    for phase_id in range(7):
        mask = phases == phase_id
        if np.sum(mask) > 0:
            phase_maes[phase_id] = compute_mae(y_true[mask], y_pred[mask])
        else:
            phase_maes[phase_id] = None
    return phase_maes


def compute_horizon_mae(y_true, y_pred, horizons_minutes=[(0, 5), (5, 10), (10, 30), (30, float('inf'))]):
    """
    Compute MAE at different prediction horizons.

    This answers: How does prediction accuracy degrade as we try to
    predict further into the future?

    Args:
        y_true: Ground truth remaining time (frames)
        y_pred: Predicted remaining time (frames)
        horizons_minutes: List of (min, max) tuples defining horizons

    Returns:
        Dictionary mapping horizon to MAE
    """
    horizon_maes = {}
    fps = 1  # Cholec80 fps

    for min_mins, max_mins in horizons_minutes:
        min_frames = min_mins * 60 * fps
        max_frames = max_mins * 60 * fps if max_mins != float('inf') else float('inf')

        mask = (y_true >= min_frames) & (y_true < max_frames)

        if np.sum(mask) > 0:
            mae = compute_mae(y_true[mask], y_pred[mask])
            mae_mins = mae / 60
            horizon_maes[f'{min_mins}-{max_mins}min'] = {
                'mae_frames': mae,
                'mae_minutes': mae_mins,
                'n_samples': int(np.sum(mask))
            }
        else:
            horizon_maes[f'{min_mins}-{max_mins}min'] = {
                'mae_frames': None,
                'mae_minutes': None,
                'n_samples': 0
            }

    return horizon_maes


def compute_time_prediction_metrics(y_true, y_pred, phases=None):
    """
    Compute all time prediction metrics.

    Args:
        y_true: Ground truth remaining time
        y_pred: Predicted remaining time
        phases: Optional phase labels for per-phase analysis

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'mae_frames': compute_mae(y_true, y_pred),
        'mae_minutes': compute_mae_minutes(y_true, y_pred),
        'weighted_mae': compute_weighted_mae(y_true, y_pred),
        'within_2_min': compute_within_threshold(y_true, y_pred, 2),
        'within_5_min': compute_within_threshold(y_true, y_pred, 5),
        'within_10_min': compute_within_threshold(y_true, y_pred, 10),
        'horizon_mae': compute_horizon_mae(y_true, y_pred)
    }

    if phases is not None:
        metrics['per_phase_mae'] = compute_per_phase_mae(y_true, y_pred, phases)

    return metrics


# ============================================================================
# TASK B: TOOL DETECTION METRICS
# ============================================================================

TOOL_NAMES = [
    'Grasper',
    'Bipolar',
    'Hook',
    'Scissors',
    'Clipper',
    'Irrigator',
    'SpecimenBag'
]


def compute_average_precision(y_true, y_pred):
    """
    Compute Average Precision (AP) for binary classification.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities

    Returns:
        AP value
    """
    return average_precision_score(y_true, y_pred)


def compute_map(y_true, y_pred):
    """
    Compute Mean Average Precision (mAP) across all tools.

    Args:
        y_true: Ground truth labels (n_samples, n_tools)
        y_pred: Predicted probabilities (n_samples, n_tools)

    Returns:
        mAP value and per-tool APs
    """
    n_tools = y_true.shape[1]
    aps = []

    for i in range(n_tools):
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        aps.append(ap)

    return np.mean(aps), aps


def compute_precision_recall_f1(y_true, y_pred, threshold=0.5):
    """
    Compute precision, recall, and F1 at a given threshold.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        (precision, recall, f1)
    """
    y_pred_binary = (y_pred >= threshold).astype(float)

    tp = np.sum(y_true * y_pred_binary)
    fp = np.sum((1 - y_true) * y_pred_binary)
    fn = np.sum(y_true * (1 - y_pred_binary))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def compute_per_tool_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute metrics for each tool separately.

    Args:
        y_true: Ground truth labels (n_samples, n_tools)
        y_pred: Predicted probabilities (n_samples, n_tools)
        threshold: Classification threshold

    Returns:
        Dictionary with per-tool metrics
    """
    n_tools = y_true.shape[1]
    tool_metrics = {}

    for i in range(n_tools):
        tool_name = TOOL_NAMES[i]
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # AP
        ap = average_precision_score(y_t, y_p)

        # Precision, Recall, F1
        precision, recall, f1 = compute_precision_recall_f1(y_t, y_p, threshold)

        # Class distribution
        n_positive = np.sum(y_t)
        n_negative = len(y_t) - n_positive
        prevalence = n_positive / len(y_t)

        tool_metrics[tool_name] = {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'prevalence': prevalence
        }

    return tool_metrics


def compute_tool_detection_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute all tool detection metrics.

    Args:
        y_true: Ground truth labels (n_samples, n_tools)
        y_pred: Predicted probabilities (n_samples, n_tools)
        threshold: Classification threshold

    Returns:
        Dictionary with all metrics
    """
    # mAP
    mAP, per_tool_ap = compute_map(y_true, y_pred)

    # Overall precision, recall, F1
    overall_p, overall_r, overall_f1 = compute_precision_recall_f1(
        y_true.flatten(), y_pred.flatten(), threshold
    )

    # Per-tool metrics
    tool_metrics = compute_per_tool_metrics(y_true, y_pred, threshold)

    metrics = {
        'mAP': mAP,
        'overall_precision': overall_p,
        'overall_recall': overall_r,
        'overall_f1': overall_f1,
        'per_tool': tool_metrics
    }

    return metrics


# ============================================================================
# COMPARISON METRICS
# ============================================================================

def compute_improvement(baseline_metrics, timed_metrics, metric_name='mAP'):
    """
    Compute the improvement of timed model over baseline.

    Args:
        baseline_metrics: Metrics from baseline model
        timed_metrics: Metrics from timed model
        metric_name: Name of metric to compare

    Returns:
        Absolute and relative improvement
    """
    baseline_val = baseline_metrics[metric_name]
    timed_val = timed_metrics[metric_name]

    absolute = timed_val - baseline_val
    relative = (absolute / baseline_val) * 100 if baseline_val > 0 else 0

    return {
        'baseline': baseline_val,
        'timed': timed_val,
        'absolute_improvement': absolute,
        'relative_improvement_pct': relative
    }


def compute_per_tool_improvement(baseline_metrics, timed_metrics):
    """
    Compute improvement for each tool.

    Args:
        baseline_metrics: Per-tool metrics from baseline
        timed_metrics: Per-tool metrics from timed model

    Returns:
        Dictionary with per-tool improvements
    """
    improvements = {}

    for tool in TOOL_NAMES:
        baseline_ap = baseline_metrics['per_tool'][tool]['ap']
        timed_ap = timed_metrics['per_tool'][tool]['ap']

        improvements[tool] = {
            'baseline_ap': baseline_ap,
            'timed_ap': timed_ap,
            'improvement': timed_ap - baseline_ap,
            'relative_improvement_pct': ((timed_ap - baseline_ap) / baseline_ap * 100)
            if baseline_ap > 0 else 0
        }

    return improvements


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def paired_t_test(scores1, scores2):
    """
    Perform paired t-test to check if difference is significant.

    Args:
        scores1: Scores from first model (e.g., per-video mAP)
        scores2: Scores from second model

    Returns:
        (t_statistic, p_value)
    """
    from scipy import stats
    return stats.ttest_rel(scores1, scores2)


def mcnemar_test(y_true, pred1, pred2, threshold=0.5):
    """
    Perform McNemar's test for comparing two classifiers.

    Args:
        y_true: Ground truth labels
        pred1: Predictions from first model
        pred2: Predictions from second model
        threshold: Classification threshold

    Returns:
        (statistic, p_value)
    """
    from scipy import stats

    pred1_binary = (pred1 >= threshold).astype(int)
    pred2_binary = (pred2 >= threshold).astype(int)

    # Contingency table
    # b: model1 correct, model2 wrong
    # c: model1 wrong, model2 correct
    correct1 = (pred1_binary == y_true)
    correct2 = (pred2_binary == y_true)

    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    # McNemar's test with continuity correction
    if b + c == 0:
        return 0, 1.0

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_metrics_table(metrics, task='A'):
    """
    Format metrics as a readable table string.

    Args:
        metrics: Dictionary of metrics
        task: 'A' for time prediction, 'B' for tool detection

    Returns:
        Formatted string
    """
    lines = []

    if task == 'A':
        lines.append("=" * 50)
        lines.append("TIME PREDICTION METRICS (Task A)")
        lines.append("=" * 50)
        lines.append(f"MAE (frames):     {metrics['mae_frames']:.2f}")
        lines.append(f"MAE (minutes):    {metrics['mae_minutes']:.2f}")
        lines.append(f"Weighted MAE:     {metrics['weighted_mae']:.2f}")
        lines.append(f"Within 2 min:     {metrics['within_2_min']:.1f}%")
        lines.append(f"Within 5 min:     {metrics['within_5_min']:.1f}%")
        lines.append(f"Within 10 min:    {metrics['within_10_min']:.1f}%")

        lines.append("\nMAE by Prediction Horizon:")
        for horizon, data in metrics['horizon_mae'].items():
            if data['mae_minutes'] is not None:
                lines.append(f"  {horizon:>12}: {data['mae_minutes']:.2f} min "
                             f"(n={data['n_samples']})")

    elif task == 'B':
        lines.append("=" * 50)
        lines.append("TOOL DETECTION METRICS (Task B)")
        lines.append("=" * 50)
        lines.append(f"mAP:              {metrics['mAP']:.4f}")
        lines.append(f"Overall Prec:     {metrics['overall_precision']:.4f}")
        lines.append(f"Overall Recall:   {metrics['overall_recall']:.4f}")
        lines.append(f"Overall F1:       {metrics['overall_f1']:.4f}")

        lines.append("\nPer-Tool Metrics:")
        lines.append(f"{'Tool':<15} {'AP':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        lines.append("-" * 50)
        for tool in TOOL_NAMES:
            m = metrics['per_tool'][tool]
            lines.append(f"{tool:<15} {m['ap']:>8.3f} {m['precision']:>8.3f} "
                         f"{m['recall']:>8.3f} {m['f1']:>8.3f}")

    return "\n".join(lines)


if __name__ == '__main__':
    # Test metrics with dummy data
    print("Testing evaluation metrics...")

    # Task A metrics
    print("\n" + "=" * 60)
    print("Testing Task A (Time Prediction) Metrics")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 1000

    # Simulate predictions (with some error)
    y_true_time = np.random.uniform(0, 30000, n_samples)  # 0-500 min
    noise = np.random.normal(0, 300, n_samples)  # ~5 min std error
    y_pred_time = np.maximum(0, y_true_time + noise)
    phases = np.random.randint(0, 7, n_samples)

    time_metrics = compute_time_prediction_metrics(y_true_time, y_pred_time, phases)
    print(format_metrics_table(time_metrics, task='A'))

    # Task B metrics
    print("\n" + "=" * 60)
    print("Testing Task B (Tool Detection) Metrics")
    print("=" * 60)

    y_true_tools = np.random.randint(0, 2, (n_samples, 7)).astype(float)
    # Simulate predictions with correlation to ground truth
    y_pred_tools = np.clip(
        y_true_tools * 0.7 + np.random.uniform(0, 0.5, (n_samples, 7)),
        0, 1
    )

    tool_metrics = compute_tool_detection_metrics(y_true_tools, y_pred_tools)
    print(format_metrics_table(tool_metrics, task='B'))

    print("\nAll tests passed!")
