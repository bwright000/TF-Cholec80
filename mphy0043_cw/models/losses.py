"""
Loss functions for MPHY0043 Coursework time prediction models.
"""

import tensorflow as tf


# =============================================================================
# NORMALIZATION CONSTANTS
# =============================================================================

MAX_REMAINING_PHASE = 5000.0    # Max frames in a single phase (~83 min at 1fps)
MAX_REMAINING_SURGERY = 30000.0  # Max frames in entire surgery (~500 min at 1fps)
MAX_FUTURE_PHASE_START = 30000.0  # Max frames until future phase starts


def normalize_targets(remaining_phase, future_phase_starts):
    """
    Normalize targets to [0, 1] range for stable training.

    Args:
        remaining_phase: Raw remaining phase time in frames
        future_phase_starts: Raw future phase start times in frames

    Returns:
        Tuple of (normalized_remaining, normalized_future)
    """
    remaining_norm = tf.cast(remaining_phase, tf.float32) / MAX_REMAINING_PHASE
    remaining_norm = tf.clip_by_value(remaining_norm, 0.0, 1.0)

    future_norm = tf.cast(future_phase_starts, tf.float32) / MAX_FUTURE_PHASE_START
    mask = tf.cast(future_phase_starts >= 0, tf.float32)
    future_norm = future_norm * mask + (1.0 - mask) * (-1.0)

    return remaining_norm, future_norm


def denormalize_predictions(remaining_pred, future_pred):
    """
    Convert normalized predictions back to frame counts.

    Args:
        remaining_pred: Normalized remaining phase prediction [0, 1]
        future_pred: Normalized future phase predictions [0, 1] or -1

    Returns:
        Tuple of (remaining_frames, future_frames)
    """
    remaining_frames = remaining_pred * MAX_REMAINING_PHASE
    future_frames = future_pred * MAX_FUTURE_PHASE_START
    return remaining_frames, future_frames


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def huber_loss(y_true, y_pred, delta=100.0):
    """
    Huber loss for time prediction.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        delta: Threshold for quadratic vs linear loss.
               For raw frames: use delta=100.0 (default, ~1.7 min at 1fps)
               For normalized [0,1]: use delta=0.1

    Returns:
        Mean Huber loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    return tf.reduce_mean(huber)


def weighted_huber_loss(y_true, y_pred, delta=0.1, use_weighting=False):
    """
    Huber loss with optional soft weighting for near-term predictions.

    Args:
        y_true: Ground truth (normalized to [0, 1])
        y_pred: Predictions (normalized to [0, 1])
        delta: Huber loss delta (default 0.1 for normalized targets)
        use_weighting: Whether to apply near-term weighting (default False)

    Returns:
        (Weighted) mean Huber loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    if use_weighting:
        weights = tf.exp(-y_true * 2.0)
        weights = weights / tf.reduce_mean(weights)
        return tf.reduce_mean(weights * huber)
    else:
        return tf.reduce_mean(huber)


def future_phase_loss(y_true, y_pred, delta=100.0):
    """
    Loss for future phase start time predictions.

    Handles -1 values (phase already passed) by masking them out.

    Args:
        y_true: Ground truth future phase starts, -1 if passed
        y_pred: Predicted future phase starts
        delta: Huber loss delta (default 100.0 for raw frames)

    Returns:
        Masked Huber loss
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Mask for valid predictions (not -1)
    mask = tf.cast(y_true >= 0, tf.float32)

    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    masked_loss = huber * mask
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(masked_loss) / n_valid


def combined_time_loss(y_true_remaining, y_pred_remaining,
                       y_true_future, y_pred_future,
                       remaining_weight=1.0, future_weight=0.5,
                       delta=100.0):
    """
    Combined loss for both prediction tasks.

    Args:
        y_true_remaining: Ground truth remaining phase time
        y_pred_remaining: Predicted remaining phase time
        y_true_future: Ground truth future phase starts
        y_pred_future: Predicted future phase starts
        remaining_weight: Weight for remaining phase loss
        future_weight: Weight for future phase loss
        delta: Huber loss delta (default 100.0 for raw frames)

    Returns:
        Combined weighted loss
    """
    loss_remaining = huber_loss(y_true_remaining, y_pred_remaining, delta=delta)
    loss_future = future_phase_loss(y_true_future, y_pred_future, delta=delta)

    return remaining_weight * loss_remaining + future_weight * loss_future


# =============================================================================
# BACKWARD-COMPATIBLE ALIASES
# =============================================================================

huber_loss_raw = huber_loss
future_phase_loss_raw = future_phase_loss
combined_time_loss_raw = combined_time_loss
