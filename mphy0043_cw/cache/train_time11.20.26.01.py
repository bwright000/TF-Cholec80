"""
Training script for Time Predictor (Task A) using TensorFlow with multi-GPU support.
Trains a model to predict remaining surgery time and future phase start times
based on video sequences and timing information.
Usage:
    python -m mphy0043_cw.training.train_time --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80
"""


import os
import sys
import argparse
import yaml
import json
import tensorflow as tf
import numpy as np
from mphy0043_cw.data.dataloader import get_train_sequence_dataset, get_val_sequence_dataset
from mphy0043_cw.models.time_predictor import (
    create_time_predictor,
    weighted_huber_loss,
    future_phase_loss,
    total_duration_loss,
    delta_loss,
    combined_time_loss_v2
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
    """Prepare batch with new features for recommendations 1, 2, 3."""
    inputs = {
        'frames': batch['frames'],
        'phase': batch['phase'],
    }
    outputs = {
        'total_phase_duration': batch['total_phase_duration'],  # Recommendation 1
        'remaining_phase': batch['remaining_phase'],
        'future_phase_deltas': batch['future_phase_deltas'],  # Recommendation 2
        'future_phase_starts': batch['future_phase_starts']
    }
    return inputs, outputs

# ============================================================================
# TRAINER
# ============================================================================

class TimePredictorTrainer(tf.keras.Model):
    """
    Custom trainer for time predictor with new loss functions.

    Uses total_duration_loss and delta_loss per recommendations 1 & 2.
    """
    def __init__(self, base_model, duration_weight=1.0, delta_weight=0.6, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.duration_weight = duration_weight
        self.delta_weight = delta_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name='mae')

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)

            # NEW: Loss on total_phase_duration (Recommendation 1)
            loss_duration = total_duration_loss(
                outputs['total_phase_duration'],
                predictions['total_phase_duration']
            )

            # NEW: Loss on future_phase_deltas (Recommendation 2)
            loss_deltas = delta_loss(
                outputs['future_phase_deltas'],
                predictions['future_phase_deltas']
            )

            total_loss = (self.duration_weight * loss_duration +
                         self.delta_weight * loss_deltas)

        # Keras 3: LossScaleOptimizer handles scaling automatically
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        # MAE on derived remaining_phase for interpretability
        self.mae_metric.update_state(
            tf.cast(outputs['remaining_phase'], tf.float32),
            tf.cast(predictions['remaining_phase'], tf.float32)
        )
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        predictions = self(inputs, training=False)

        # NEW: Loss on total_phase_duration (Recommendation 1)
        loss_duration = total_duration_loss(
            outputs['total_phase_duration'],
            predictions['total_phase_duration']
        )

        # NEW: Loss on future_phase_deltas (Recommendation 2)
        loss_deltas = delta_loss(
            outputs['future_phase_deltas'],
            predictions['future_phase_deltas']
        )

        total_loss = (self.duration_weight * loss_duration +
                     self.delta_weight * loss_deltas)

        self.loss_tracker.update_state(total_loss)
        # MAE on derived remaining_phase for interpretability
        self.mae_metric.update_state(
            tf.cast(outputs['remaining_phase'], tf.float32),
            tf.cast(predictions['remaining_phase'], tf.float32)
        )
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_time_predictor(config, data_dir):
    batch_size = config['training'].get('batch_size', 64)
    model_config = config['model']['time_predictor']
    
    with strategy.scope():
        # Load Datasets
        train_ds = get_train_sequence_dataset(
            data_dir=data_dir,
            sequence_length=model_config['sequence_length'],
            batch_size=batch_size,
            stride=model_config.get('sequence_stride', 16),
            timing_labels_path=config['paths']['timing_labels_path']
        ).map(prepare_batch_for_time_model, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        val_ds = get_val_sequence_dataset(
            data_dir=data_dir,
            sequence_length=model_config['sequence_length'],
            batch_size=batch_size,
            stride=model_config['sequence_length'],
            timing_labels_path=config['paths']['timing_labels_path']
        ).map(prepare_batch_for_time_model, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        # Get input shape from config for memory efficiency
        input_shape = (
            config['data'].get('frame_height', 480),
            config['data'].get('frame_width', 854),
            config['data'].get('frame_channels', 3)
        )

        # Create Parallel SSM Model
        base_model = create_time_predictor(
            sequence_length=model_config['sequence_length'],
            d_model=model_config['d_model'],
            d_state=model_config['d_state'],
            n_ssm_blocks=model_config['n_ssm_blocks'],
            backbone_trainable_layers=config['model'].get('backbone_trainable_layers', 0),
            input_shape=input_shape
        )

        model = TimePredictorTrainer(
            base_model,
            remaining_weight=config['training'].get('remaining_weight', 0.8),
            future_weight=config['training'].get('future_weight', 0.6)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(optimizer=optimizer)

    # Callbacks and Fitting
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['paths']['checkpoint_dir'], 'time_predictor_best.keras'),
            save_best_only=True, monitor='val_mae'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=5)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
    
    # Save results
    base_model.save(os.path.join(config['paths']['checkpoint_dir'], 'time_predictor_final.keras'))

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
