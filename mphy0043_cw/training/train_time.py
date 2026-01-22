"""
Training script for Time Predictor (Task A).

Trains the SSM-based time prediction model to predict:
1. Remaining time in current surgical phase
2. Start times of future phases

Usage:
    python training/train_time.py --config config.yaml
    python training/train_time.py --config config.yaml --epochs 50 --batch_size 16
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.dataloader import get_train_dataset, get_val_dataset
from data.augmentation import get_augmentation_fn
from models.time_predictor import (
    create_time_predictor,
    weighted_huber_loss,
    future_phase_loss
)


# ============================================================================
# CUSTOM TRAINING STEP
# ============================================================================

class TimePredictor(tf.keras.Model):
    """
    Custom model class for time prediction with custom training step.

    This wraps the base time predictor model and adds:
    - Custom loss computation
    - Gradient clipping
    - Mixed precision support
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

        # Metrics
        self.mae_remaining = tf.keras.metrics.MeanAbsoluteError(name='mae_remaining')

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)

            # Compute losses
            loss_remaining = weighted_huber_loss(
                y['remaining_phase'],
                predictions['remaining_phase']
            )
            loss_future = future_phase_loss(
                y['future_phase_starts'],
                predictions['future_phase_starts']
            )

            # Combined loss
            total_loss = (
                self.remaining_weight * loss_remaining +
                self.future_weight * loss_future
            )

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.remaining_loss_tracker.update_state(loss_remaining)
        self.future_loss_tracker.update_state(loss_future)
        self.mae_remaining.update_state(
            y['remaining_phase'],
            predictions['remaining_phase']
        )

        return {
            'loss': self.loss_tracker.result(),
            'remaining_loss': self.remaining_loss_tracker.result(),
            'future_loss': self.future_loss_tracker.result(),
            'mae_remaining': self.mae_remaining.result()
        }

    def test_step(self, data):
        x, y = data
        predictions = self(x, training=False)

        # Compute losses
        loss_remaining = weighted_huber_loss(
            y['remaining_phase'],
            predictions['remaining_phase']
        )
        loss_future = future_phase_loss(
            y['future_phase_starts'],
            predictions['future_phase_starts']
        )
        total_loss = (
            self.remaining_weight * loss_remaining +
            self.future_weight * loss_future
        )

        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.remaining_loss_tracker.update_state(loss_remaining)
        self.future_loss_tracker.update_state(loss_future)
        self.mae_remaining.update_state(
            y['remaining_phase'],
            predictions['remaining_phase']
        )

        return {
            'loss': self.loss_tracker.result(),
            'remaining_loss': self.remaining_loss_tracker.result(),
            'future_loss': self.future_loss_tracker.result(),
            'mae_remaining': self.mae_remaining.result()
        }

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.remaining_loss_tracker,
            self.future_loss_tracker,
            self.mae_remaining
        ]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_batch_for_time_model(batch, sequence_length=64):
    """
    Prepare a batch from the dataloader for the time prediction model.

    The time predictor expects sequences of frames, but the dataloader
    provides individual frames. This function creates pseudo-sequences
    from individual frames (for single-frame prediction mode).

    For full sequence training, you would need to modify the dataloader
    to return actual video sequences.

    Args:
        batch: Batch from dataloader
        sequence_length: Expected sequence length (set to 1 for frame-by-frame)

    Returns:
        (inputs_dict, outputs_dict) for model training
    """
    # For now, we treat each frame as a sequence of length 1
    # This simplifies training but loses temporal context
    # TODO: Implement proper sequence batching for better results

    batch_size = tf.shape(batch['frame'])[0]

    # Add sequence dimension (batch, H, W, C) -> (batch, 1, H, W, C)
    frames = tf.expand_dims(batch['frame'], axis=1)

    # Prepare timing inputs
    elapsed = tf.cast(batch['elapsed_phase'], tf.float32)
    elapsed = tf.expand_dims(tf.expand_dims(elapsed, -1), 1)  # (batch, 1, 1)

    phases = tf.expand_dims(batch['phase'], axis=1)  # (batch, 1)

    inputs = {
        'frames': frames,
        'elapsed_phase': elapsed,
        'phase': phases
    }

    # Prepare outputs
    remaining_phase = tf.cast(batch['remaining_phase'], tf.float32)
    remaining_phase = tf.expand_dims(tf.expand_dims(remaining_phase, -1), 1)  # (batch, 1, 1)

    future_starts = tf.cast(batch['future_phase_starts'], tf.float32)
    future_starts = tf.expand_dims(future_starts[:, 1:], axis=1)  # (batch, 1, 6) - exclude phase 0

    outputs = {
        'remaining_phase': remaining_phase,
        'future_phase_starts': future_starts
    }

    return inputs, outputs


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_optimizer(config):
    """Create optimizer with learning rate schedule."""
    initial_lr = config['training']['learning_rate']

    # Cosine decay schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=config['training']['epochs'] * 1000,  # Approximate steps
        alpha=0.1  # Final LR = 10% of initial
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )

    return optimizer


def create_callbacks(config, output_dir):
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoints', 'time_model_{epoch:03d}.keras')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_mae_remaining',
        mode='min',
        verbose=1
    ))

    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_mae_remaining',
        patience=config['training'].get('early_stopping_patience', 10),
        mode='min',
        verbose=1,
        restore_best_weights=True
    ))

    # Learning rate reduction
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae_remaining',
        factor=0.5,
        patience=5,
        mode='min',
        verbose=1
    ))

    # TensorBoard
    log_dir = os.path.join(output_dir, 'logs', 'time_predictor')
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    ))

    return callbacks


def train_time_predictor(config):
    """Main training function."""
    print("=" * 60)
    print("Training Time Predictor (Task A)")
    print("=" * 60)

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['paths']['output_dir'], f'time_predictor_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"\nOutput directory: {output_dir}")

    # ========== DATA ==========
    print("\n1. Loading datasets...")

    batch_size = config['training']['batch_size']
    timing_labels_path = config['paths']['timing_labels_path']
    config_path = config['paths'].get('cholec80_config_path')

    # Get datasets
    train_ds = get_train_dataset(batch_size, timing_labels_path, config_path)
    val_ds = get_val_dataset(batch_size, timing_labels_path, config_path)

    # Apply augmentation to training data
    augment_fn = get_augmentation_fn(training=True)
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare batches for time model
    train_ds = train_ds.map(
        lambda batch: prepare_batch_for_time_model(batch, sequence_length=1),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda batch: prepare_batch_for_time_model(batch, sequence_length=1),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("   Datasets loaded successfully")

    # ========== MODEL ==========
    print("\n2. Creating model...")

    model_config = config['model']['time_predictor']

    base_model = create_time_predictor(
        sequence_length=1,  # Single frame for now
        d_model=model_config.get('d_model', 256),
        d_state=model_config.get('d_state', 64),
        n_ssm_blocks=model_config.get('n_ssm_blocks', 2),
        phase_embedding_dim=model_config.get('phase_embedding_dim', 16),
        dropout_rate=config['training']['dropout_rate'],
        backbone_trainable_layers=model_config.get('backbone_trainable_layers', 1),
        input_shape=(480, 854, 3)
    )

    # Wrap with custom training
    model = TimePredictor(
        base_model,
        remaining_weight=1.0,
        future_weight=0.5
    )

    # Compile
    optimizer = create_optimizer(config)
    model.compile(optimizer=optimizer)

    print("   Model created successfully")

    # Print summary
    print("\n   Model Architecture:")
    base_model.summary(print_fn=lambda x: print(f"   {x}"))

    # ========== TRAINING ==========
    print("\n3. Starting training...")

    callbacks = create_callbacks(config, output_dir)

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
    final_model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"   Model saved to: {final_model_path}")

    # Save training history
    history_path = os.path.join(output_dir, 'history.json')
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"   History saved to: {history_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val MAE: {min(history.history['val_mae_remaining']):.2f} frames")
    print("=" * 60)

    return model, history


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Time Predictor (Task A)')
    parser.add_argument('--config', type=str, default='mphy0043_cw/config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Train
    train_time_predictor(config)


if __name__ == '__main__':
    main()
