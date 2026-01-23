"""
Training script for Time Predictor (Task A).

Trains the SSM-based time prediction model to predict:
1. Remaining time in current surgical phase
2. Start times of future phases

Usage:
    python -m mphy0043_cw.training.train_time --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80
"""

import os
import sys
import argparse
import yaml
import json
from datetime import datetime

import tensorflow as tf
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mphy0043_cw.data.dataloader import get_train_dataset, get_val_dataset
from mphy0043_cw.models.time_predictor import (
    create_time_predictor,
    weighted_huber_loss,
    future_phase_loss
)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_batch_for_time_model(batch):
    """
    Prepare a batch from the dataloader for the time prediction model.

    Args:
        batch: Batch from dataloader with keys:
            - frame: (batch, H, W, 3)
            - phase: (batch,)
            - elapsed_phase: (batch,)
            - remaining_phase: (batch,)
            - future_phase_starts: (batch, 7)

    Returns:
        (inputs, outputs) tuple for model training
    """
    # Normalize frame to [0, 1]
    frame = tf.cast(batch['frame'], tf.float32) / 255.0

    # Prepare inputs
    phase = batch['phase']
    elapsed = tf.cast(batch['elapsed_phase'], tf.float32)

    inputs = (frame, phase, elapsed)

    # Prepare outputs
    remaining_phase = tf.cast(batch['remaining_phase'], tf.float32)
    future_starts = tf.cast(batch['future_phase_starts'], tf.float32)

    outputs = {
        'remaining_phase': remaining_phase,
        'future_phase_starts': future_starts
    }

    return inputs, outputs


# ============================================================================
# CUSTOM MODEL WITH TRAINING STEP
# ============================================================================

class TimePredictorTrainer(tf.keras.Model):
    """
    Custom model wrapper for time prediction with custom training step.
    """

    def __init__(self, base_model, remaining_weight=1.0, future_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.remaining_weight = remaining_weight
        self.future_weight = future_weight

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.remaining_loss_tracker = tf.keras.metrics.Mean(name='remaining_loss')
        self.future_loss_tracker = tf.keras.metrics.Mean(name='future_loss')
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name='mae')

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        inputs, outputs = data

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)

            # Compute losses
            loss_remaining = weighted_huber_loss(
                outputs['remaining_phase'],
                predictions['remaining_phase']
            )
            loss_future = future_phase_loss(
                outputs['future_phase_starts'],
                predictions['future_phase_starts']
            )

            total_loss = (
                self.remaining_weight * loss_remaining +
                self.future_weight * loss_future
            )

        # Compute and clip gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.remaining_loss_tracker.update_state(loss_remaining)
        self.future_loss_tracker.update_state(loss_future)
        self.mae_metric.update_state(outputs['remaining_phase'], predictions['remaining_phase'])

        return {
            'loss': self.loss_tracker.result(),
            'remaining_loss': self.remaining_loss_tracker.result(),
            'future_loss': self.future_loss_tracker.result(),
            'mae': self.mae_metric.result()
        }

    def test_step(self, data):
        inputs, outputs = data
        predictions = self(inputs, training=False)

        loss_remaining = weighted_huber_loss(
            outputs['remaining_phase'],
            predictions['remaining_phase']
        )
        loss_future = future_phase_loss(
            outputs['future_phase_starts'],
            predictions['future_phase_starts']
        )
        total_loss = (
            self.remaining_weight * loss_remaining +
            self.future_weight * loss_future
        )

        self.loss_tracker.update_state(total_loss)
        self.remaining_loss_tracker.update_state(loss_remaining)
        self.future_loss_tracker.update_state(loss_future)
        self.mae_metric.update_state(outputs['remaining_phase'], predictions['remaining_phase'])

        return {
            'loss': self.loss_tracker.result(),
            'remaining_loss': self.remaining_loss_tracker.result(),
            'future_loss': self.future_loss_tracker.result(),
            'mae': self.mae_metric.result()
        }

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.remaining_loss_tracker,
            self.future_loss_tracker,
            self.mae_metric
        ]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_callbacks(config, checkpoint_dir):
    """Create training callbacks."""
    callbacks = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'time_predictor_best.keras'),
        save_best_only=True,
        monitor='val_mae',
        mode='min',
        verbose=1
    ))

    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=config['training'].get('early_stopping_patience', 10),
        mode='min',
        verbose=1,
        restore_best_weights=True
    ))

    # Learning rate reduction
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,
        patience=5,
        mode='min',
        verbose=1
    ))

    return callbacks


def train_time_predictor(config, data_dir):
    """Main training function."""
    print("=" * 60)
    print("Training Time Predictor (Task A)")
    print("=" * 60)

    checkpoint_dir = config['paths']['checkpoint_dir']
    timing_labels_path = config['paths']['timing_labels_path']
    batch_size = config['training']['batch_size']

    # ========== DATA ==========
    print("\n1. Loading datasets...")

    train_ds = get_train_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        timing_labels_path=timing_labels_path,
        shuffle=True
    )

    val_ds = get_val_dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        timing_labels_path=timing_labels_path
    )

    # Prepare batches for time model
    train_ds = train_ds.map(prepare_batch_for_time_model, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prepare_batch_for_time_model, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("   Datasets loaded successfully")

    # ========== MODEL ==========
    print("\n2. Creating model...")

    model_config = config['model']['time_predictor']

    base_model = create_time_predictor(
        d_model=model_config.get('d_model', 128),
        d_state=model_config.get('d_state', 32),
        n_ssm_blocks=model_config.get('n_ssm_blocks', 2),
        phase_embedding_dim=model_config.get('phase_embedding_dim', 16),
        dropout_rate=config['training'].get('dropout_rate', 0.3),
        backbone_trainable_layers=model_config.get('backbone_trainable_layers', 1)
    )

    # Wrap with custom training
    model = TimePredictorTrainer(
        base_model,
        remaining_weight=config['training'].get('remaining_weight', 1.0),
        future_weight=config['training'].get('future_weight', 0.5)
    )

    # Compile
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['training']['learning_rate']
    )
    model.compile(optimizer=optimizer)

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
    final_path = os.path.join(checkpoint_dir, 'time_predictor.keras')
    model.base_model.save(final_path)
    print(f"   Model saved to: {final_path}")

    # Save training history
    history_path = os.path.join(checkpoint_dir, 'time_predictor_history.json')
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    if 'val_mae' in history.history:
        print(f"Best val MAE: {min(history.history['val_mae']):.2f} frames")
    print("=" * 60)

    return model, history


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Time Predictor (Task A)')
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
    train_time_predictor(config, args.data_dir)


if __name__ == '__main__':
    main()
