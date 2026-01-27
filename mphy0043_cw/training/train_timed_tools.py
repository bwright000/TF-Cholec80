"""
Training script for Timed Tool Detector (Task B - With Timing).

Trains the tool detection model that uses BOTH visual input AND timing information.
This model will be compared against the baseline to measure the improvement.

Usage:
    python -m mphy0043_cw.training.train_timed_tools --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80
"""

# ============================================================================
# ENVIRONMENT SETUP (must be before TensorFlow import)
# Disable cuDNN to avoid library loading issues on cluster
# ============================================================================
import os
os.environ['TF_USE_CUDNN'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import sys
import argparse
import yaml
import json
from datetime import datetime

import tensorflow as tf
import numpy as np

# ============================================================================
# GPU CONFIGURATION (must happen before any GPU operations)
# Reference: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Use MirroredStrategy for multi-GPU training (after GPU config)
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices for Task B: {strategy.num_replicas_in_sync}")

from mphy0043_cw.data.dataloader import get_train_dataset, get_val_dataset
from mphy0043_cw.models.timed_tool_detector import create_timed_tool_detector
from mphy0043_cw.models.tool_detector import FocalLoss, NUM_TOOLS, TOOL_NAMES


# ============================================================================
# METRICS
# ============================================================================

class MeanAveragePrecision(tf.keras.metrics.Metric):
    """Mean Average Precision for multi-label classification."""

    def __init__(self, num_classes=7, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.auc_metrics = [
            tf.keras.metrics.AUC(curve='PR', name=f'ap_{i}')
            for i in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        for i in range(self.num_classes):
            self.auc_metrics[i].update_state(
                y_true[:, i], y_pred[:, i], sample_weight
            )

    def result(self):
        aps = [m.result() for m in self.auc_metrics]
        return tf.reduce_mean(aps)

    def reset_state(self):
        for m in self.auc_metrics:
            m.reset_state()


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_batch_for_timed_tool_model(batch):
    """
    Prepare a batch from the dataloader for the timed tool detection model.

    Args:
        batch: Batch from dataloader

    Returns:
        (inputs, outputs) tuple for model training
    """
    # Keep frame as uint8 - model handles preprocessing internally
    frame = batch['frame']

    # Model expects timing inputs with shape (batch, 1), so expand dims
    remaining_phase = tf.cast(batch['remaining_phase'], tf.float32)
    remaining_surgery = tf.cast(batch['remaining_surgery'], tf.float32)
    phase_progress = tf.cast(batch['phase_progress'], tf.float32)

    # Expand dims if needed (dataloader provides scalars)
    if len(remaining_phase.shape) == 1:
        remaining_phase = tf.expand_dims(remaining_phase, axis=-1)
    if len(remaining_surgery.shape) == 1:
        remaining_surgery = tf.expand_dims(remaining_surgery, axis=-1)
    if len(phase_progress.shape) == 1:
        phase_progress = tf.expand_dims(phase_progress, axis=-1)

    inputs = {
        'frame': frame,
        'remaining_phase': remaining_phase,
        'remaining_surgery': remaining_surgery,
        'phase_progress': phase_progress,
        'phase': tf.cast(batch['phase'], tf.int32)
    }

    # Tools/Instruments are the targets
    outputs = tf.cast(batch['instruments'], tf.float32)

    return inputs, outputs


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_callbacks(config, checkpoint_dir):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'timed_tool_best.keras'),
            save_best_only=True,
            monitor='val_mean_average_precision',
            mode='max'
        ),
        # Reduce LR by half if mAP doesn't improve for 3 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mean_average_precision',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mean_average_precision',
            patience=8,
            restore_best_weights=True
        )
    ]
    return callbacks


def train_timed_tool_detector(config, data_dir):
    """Main training function for timed tool detector."""
    print("=" * 60)
    print("Training Timed Tool Detector (Task B - With Timing)")
    print("=" * 60)

    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    timing_labels_path = config['paths']['timing_labels_path']
    batch_size = config['training']['batch_size']

    # ========== DATA ==========
    print("\n1. Loading datasets...")

    # Use video IDs from config (1-indexed)
    train_video_ids = config['data'].get('train_videos', None)
    val_video_ids = config['data'].get('val_videos', None)

    train_ds = get_train_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        timing_labels_path=timing_labels_path,
        shuffle=True,
        video_ids=train_video_ids
    )

    val_ds = get_val_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        timing_labels_path=timing_labels_path,
        video_ids=val_video_ids
    )

    # Prepare batches for timed tool model
    train_ds = train_ds.map(prepare_batch_for_timed_tool_model, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prepare_batch_for_timed_tool_model, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("   Datasets loaded successfully")

    # ========== MODEL ==========
    print("\n2. Creating model...")

    model_config = config['model'].get('timed_tool_detector', {})

    # Get input shape from config for memory efficiency
    input_shape = (
        config['data'].get('frame_height', 480),
        config['data'].get('frame_width', 854),
        config['data'].get('frame_channels', 3)
    )

    with strategy.scope():
        # Load weights from Task A if you want to use the ResNet features
        # already learned there, or start fresh for the baseline comparison.
        model = create_timed_tool_detector(
            visual_hidden_dim=model_config.get('visual_hidden_dim', 512),
            timing_hidden_dim=model_config.get('timing_hidden_dim', 64),
            combined_hidden_dim=model_config.get('combined_hidden_dim', 512),
            dropout_rate=config['training'].get('dropout_rate', 0.3),
            backbone_trainable_layers=model_config.get('backbone_trainable_layers', 1),
            input_shape=input_shape,
            max_remaining_phase=model_config.get('max_remaining_phase', 5000.0),
            max_remaining_surgery=model_config.get('max_remaining_surgery', 30000.0)
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['training']['learning_rate']
        )
        
        # Mixed precision optimizer wrapper
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        loss = FocalLoss(
            gamma=config['training'].get('focal_gamma', 2.0),
            alpha=config['training'].get('focal_alpha', 0.25)
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                MeanAveragePrecision(num_classes=NUM_TOOLS, name='mean_average_precision')
            ]
        )

    print("   Model created successfully")

    # ========== TRAINING ==========
    print("\n3. Starting training...")

    callbacks = create_callbacks(config, checkpoint_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # ========== SAVE RESULTS ==========
    print("\n4. Saving results...")

    # Save final model
    final_path = os.path.join(checkpoint_dir, 'timed_tool_detector.keras')
    model.save(final_path)
    print(f"   Model saved to: {final_path}")

    # Save training history
    history_path = os.path.join(checkpoint_dir, 'timed_tool_detector_history.json')
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    if 'val_mean_average_precision' in history.history:
        print(f"Best val mAP: {max(history.history['val_mean_average_precision']):.4f}")
    print("=" * 60)

    return model, history


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Timed Tool Detector (Task B with Timing)'
    )
    parser.add_argument('--config', type=str, default='mphy0043_cw/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Cholec80 data directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Train
    train_timed_tool_detector(config, args.data_dir)


if __name__ == '__main__':
    main()
