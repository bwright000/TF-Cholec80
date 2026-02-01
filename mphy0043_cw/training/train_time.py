"""
Training script for Time Predictor (Task A) using TensorFlow with multi-GPU support.
Trains a model to predict remaining surgery time and future phase start times
based on video sequences and timing information.
Usage:
    python -m mphy0043_cw.training.train_time --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80
"""

# ============================================================================
# ENVIRONMENT SETUP (must be before TensorFlow import)
# Disable cuDNN to avoid library loading issues on cluster
# ============================================================================
import os
# os.environ['TF_USE_CUDNN'] = '0'
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import sys
import argparse
import yaml
import json
import tensorflow as tf
import numpy as np
from mphy0043_cw.data.dataloader import get_train_sequence_dataset, get_val_sequence_dataset
from mphy0043_cw.models.time_predictor import create_time_predictor
from mphy0043_cw.models.losses import (
    huber_loss_raw, future_phase_loss_raw
)

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

# Mixed precision for speed and memory efficiency
# Reference: https://www.tensorflow.org/guide/mixed_precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Use MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_batch_for_time_model(batch):
    """
    Prepare batch for time model with raw frame count targets.

    Targets are raw frame counts (no normalization):
    - remaining_phase: frames until current phase ends
    - future_phase_starts: frames until each future phase starts (-1 if passed)
    """
    inputs = {
        'frames': batch['frames'],
        'phase': batch['phase'],
        'elapsed_phase': batch['elapsed_phase']
    }

    # Raw frame targets (no normalization)
    outputs = {
        'remaining_phase': tf.cast(batch['remaining_phase'], tf.float32),
        'future_phase_starts': tf.cast(batch['future_phase_starts'], tf.float32)
    }
    return inputs, outputs

# ============================================================================
# TRAINER
# ============================================================================

class TimePredictorTrainer(tf.keras.Model):
    """
    Custom training wrapper for time predictor model.

    Uses raw frame count targets and Huber loss with delta=100 frames.
    MAE is computed directly in frames (no normalization needed).
    Per-phase MAE analysis is done in evaluate.py after training.
    """

    def __init__(self, base_model, remaining_weight=0.8, future_weight=0.6,
                 huber_delta=100.0, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.remaining_weight = remaining_weight
        self.future_weight = future_weight
        self.huber_delta = huber_delta
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        # MAE in frames (used for early stopping via 'val_mae_frames')
        self.mae_frames_metric = tf.keras.metrics.MeanAbsoluteError(name='mae_frames')

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def _compute_loss(self, outputs, predictions):
        """Compute combined loss for remaining phase and future phase predictions."""
        loss_rem = huber_loss_raw(
            outputs['remaining_phase'],
            predictions['remaining_phase'],
            delta=self.huber_delta
        )
        loss_fut = future_phase_loss_raw(
            outputs['future_phase_starts'],
            predictions['future_phase_starts'],
            delta=self.huber_delta
        )
        return self.remaining_weight * loss_rem + self.future_weight * loss_fut

    def _update_metrics(self, outputs, predictions, total_loss):
        """Update tracking metrics."""
        y_true_f32 = tf.cast(outputs['remaining_phase'], tf.float32)
        y_pred_f32 = tf.cast(predictions['remaining_phase'], tf.float32)
        self.loss_tracker.update_state(total_loss)
        self.mae_frames_metric.update_state(y_true_f32, y_pred_f32)

    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            total_loss = self._compute_loss(outputs, predictions)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._update_metrics(outputs, predictions, total_loss)
        return {"loss": self.loss_tracker.result(), "mae_frames": self.mae_frames_metric.result()}

    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        predictions = self(inputs, training=False)
        total_loss = self._compute_loss(outputs, predictions)

        self._update_metrics(outputs, predictions, total_loss)
        return {"loss": self.loss_tracker.result(), "mae_frames": self.mae_frames_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_frames_metric]


# ============================================================================
# LEARNING RATE SCHEDULE WITH WARMUP
# ============================================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by cosine decay.

    Warmup helps stabilize early training by starting with a smaller learning rate
    and gradually increasing to the target rate.
    """

    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        # Cache as TF constants to avoid repeated casting every step
        self.base_lr = tf.constant(base_lr, dtype=tf.float32)
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        self.total_steps = tf.constant(total_steps, dtype=tf.float32)
        self.decay_steps = self.total_steps - self.warmup_steps
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup
        warmup_lr = self.base_lr * (step / self.warmup_steps)

        # Cosine decay after warmup
        decay_step = step - self.warmup_steps
        cosine_decay = 0.5 * (1.0 + tf.cos(self.pi * decay_step / self.decay_steps))
        decay_lr = self.base_lr * cosine_decay

        # Use warmup_lr if step < warmup_steps, else decay_lr
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'base_lr': float(self.base_lr.numpy()),
            'warmup_steps': int(self.warmup_steps.numpy()),
            'total_steps': int(self.total_steps.numpy())
        }


class BaseModelCheckpoint(tf.keras.callbacks.Callback):
    """Custom callback to save base_model (not the wrapper) after every epoch."""

    def __init__(self, base_model, checkpoint_dir, monitor='val_mae_frames'):
        super().__init__()
        self.base_model = base_model
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.best = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        # Save this epoch's model
        epoch_path = os.path.join(self.checkpoint_dir, f'time_predictor_epoch_{epoch+1}.keras')
        self.base_model.save(epoch_path)
        print(f"\nSaved epoch {epoch+1} model to {epoch_path}")

        # Track best and save as main checkpoint
        if current is not None and current < self.best:
            self.best = current
            self.best_epoch = epoch + 1
            best_path = os.path.join(self.checkpoint_dir, 'time_predictor.keras')
            self.base_model.save(best_path)
            print(f"New best model ({self.monitor}={current:.2f} frames) saved to {best_path}")

    def on_train_end(self, logs=None):
        print(f"\nTraining complete. Best model was epoch {self.best_epoch} ({self.monitor}={self.best:.2f} frames)")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_time_predictor(config, data_dir):
    # Use sequence_batch_size (not batch_size) for sequence data
    # Memory: batch × seq_len × H × W × C × 4 bytes = batch × 64 × 480 × 854 × 3 × 4
    batch_size = config['training'].get('sequence_batch_size', 4)
    model_config = config['model']['time_predictor']

    # Ensure checkpoint directory exists
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Use video IDs from config (1-indexed)
    train_video_ids = config['data'].get('train_videos', None)
    val_video_ids = config['data'].get('val_videos', None)

    with strategy.scope():
        # Load Datasets
        train_ds = get_train_sequence_dataset(
            data_dir=data_dir,
            sequence_length=model_config['sequence_length'],
            batch_size=batch_size,
            stride=model_config.get('sequence_stride', 16),
            timing_labels_path=config['paths']['timing_labels_path'],
            video_ids=train_video_ids
        ).map(prepare_batch_for_time_model, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        val_ds = get_val_sequence_dataset(
            data_dir=data_dir,
            sequence_length=model_config['sequence_length'],
            batch_size=batch_size,
            stride=model_config['sequence_length'],
            timing_labels_path=config['paths']['timing_labels_path'],
            video_ids=val_video_ids
        ).map(prepare_batch_for_time_model, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        # Create S4-style SSM Model
        base_model = create_time_predictor(
            sequence_length=model_config['sequence_length'],
            d_model=model_config['d_model'],
            d_state=model_config['d_state'],
            n_ssm_blocks=model_config['n_ssm_blocks'],
            backbone_trainable_layers=config['model'].get('backbone_trainable_layers', 0)
        )

        model = TimePredictorTrainer(
            base_model,
            remaining_weight=config['training'].get('remaining_weight', 0.8),
            future_weight=config['training'].get('future_weight', 0.6),
            huber_delta=config['training'].get('huber_delta', 100.0)
        )

        # Learning rate with warmup
        # Estimate steps: ~50 videos × ~2000 frames/video / batch_size / stride
        # With stride=16, batch=1, seq_len=64: ~6250 sequences per epoch
        epochs = config['training'].get('epochs', 20)
        steps_per_epoch = config['training'].get('steps_per_epoch', 6000)
        total_steps = epochs * steps_per_epoch
        warmup_steps = config['training'].get('warmup_steps', steps_per_epoch)  # 1 epoch warmup

        base_lr = config['training']['learning_rate']
        lr_schedule = WarmupCosineDecay(
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(optimizer=optimizer)

        print(f"\n{'='*60}")
        print("Training Configuration:")
        print(f"  - Base learning rate: {base_lr}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Huber delta: {config['training'].get('huber_delta', 100.0)} frames")
        print(f"  - Targets: RAW FRAME COUNTS (no normalization)")
        print(f"  - MAE displayed directly in frames")
        print(f"{'='*60}\n")

    # Callbacks and Fitting
    # Save every epoch + track best model
    # Monitor val_mae_frames for early stopping (more interpretable)
    callbacks = [
        BaseModelCheckpoint(base_model, checkpoint_dir, monitor='val_mae_frames'),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae_frames',
            patience=config['training'].get('early_stopping_patience', 5),
            mode='min',
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['training'].get('epochs', 20),
        callbacks=callbacks
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Time Predictor (Task A)')
    parser.add_argument('--config', type=str, default='mphy0043_cw/config.yaml',
                        help='Path to config file')
    # Made optional for local testing if paths are already in config
    parser.add_argument('--data_dir', type=str, 
                        help='Path to Cholec80 data directory')
    
    args = parser.parse_args()

    # 1. Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Determine Data Directory
    # Priority: Command line argument > Config file > Environment variable
    data_dir = args.data_dir or config['paths'].get('data_dir')
    
    if not data_dir or not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found. "
              "Please provide it via --data_dir or update config.yaml")
        sys.exit(1)

    # 3. Launch Training
    try:
        train_time_predictor(config, data_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
