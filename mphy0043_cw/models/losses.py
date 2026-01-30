"""
Loss functions for MPHY0043 Coursework time prediction models.

Contains loss functions used by:
- time_predictor.py (SSM-based model)
- time_predictor_lstm.py (LSTM baseline)
- train_time.py (training script)
"""

import tensorflow as tf


def weighted_huber_loss(y_true, y_pred, delta=1.0):
    """
    Weighted Huber loss that gives higher weight to near-term predictions.

    Near-term predictions are more clinically relevant (OR scheduling),
    so we weight them more heavily.

    Args:
        y_true: Ground truth remaining times
        y_pred: Predicted remaining times
        delta: Huber loss delta parameter

    Returns:
        Weighted loss value
    """
    # Compute Huber loss
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    # Weight by inverse of remaining time (more weight for near-term)
    # Add small constant to avoid division by zero
    weights = 1.0 / (tf.abs(y_true) + 1.0)

    # Normalize weights
    weights = weights / tf.reduce_mean(weights)

    return tf.reduce_mean(weights * huber)


def future_phase_loss(y_true, y_pred):
    """
    Loss for future phase start time predictions.

    Handles -1 values (phase already passed) by masking them out.

    Args:
        y_true: Ground truth future phase starts, -1 if passed
        y_pred: Predicted future phase starts

    Returns:
        Masked Huber loss
    """
    # Create mask for valid predictions (not -1)
    mask = tf.cast(y_true >= 0, tf.float32)

    # Huber loss on valid predictions only
    error = y_true - y_pred
    abs_error = tf.abs(error)
    delta = 1.0
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    # Apply mask
    masked_loss = huber * mask

    # Average over valid predictions only
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(masked_loss) / n_valid


def combined_time_loss(y_true_remaining, y_pred_remaining,
                       y_true_future, y_pred_future,
                       remaining_weight=1.0, future_weight=0.5):
    """
    Combined loss for both prediction tasks.

    Args:
        y_true_remaining: Ground truth remaining phase time
        y_pred_remaining: Predicted remaining phase time
        y_true_future: Ground truth future phase starts
        y_pred_future: Predicted future phase starts
        remaining_weight: Weight for remaining phase loss
        future_weight: Weight for future phase loss

    Returns:
        Combined weighted loss
    """
    loss_remaining = weighted_huber_loss(y_true_remaining, y_pred_remaining)
    loss_future = future_phase_loss(y_true_future, y_pred_future)

    return remaining_weight * loss_remaining + future_weight * loss_future
