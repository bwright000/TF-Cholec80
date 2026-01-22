"""
Training script for Tool Detector Baseline (Task B - Visual Only).

Trains the baseline tool detection model that uses ONLY visual input.
This model will be compared against the timed version to measure
the improvement from adding timing information.

Usage:
    python training/train_tools.py --config config.yaml
    python training/train_tools.py --config config.yaml --epochs 50
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
from models.tool_detector import (
    create_tool_detector,
    FocalLoss,
    focal_loss,
    compute_tool_metrics,
    NUM_TOOLS,
    TOOL_NAMES
)


# ============================================================================
# METRICS
# ============================================================================

class MultiLabelAUC(tf.keras.metrics.AUC):
    """AUC metric for multi-label classification."""

    def __init__(self, num_labels=7, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class MeanAveragePrecision(tf.keras.metrics.Metric):
    """
    Mean Average Precision (mAP) for multi-label classification.

    Computes AP for each tool and averages them.
    """

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

def prepare_batch_for_tool_model(batch):
    """
    Prepare a batch from the dataloader for the tool detection model.

    Args:
        batch: Batch from dataloader

    Returns:
        (inputs, outputs) tuple for model training
    """
    inputs = batch['frame']
    outputs = tf.cast(batch['instruments'], tf.float32)

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
        decay_steps=config['training']['epochs'] * 1000,
        alpha=0.1
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    return optimizer


def create_callbacks(config, output_dir):
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
    checkpoint_path = os.path.join(
        output_dir, 'checkpoints', 'tool_baseline_{epoch:03d}.keras'
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_mean_average_precision',
        mode='max',
        verbose=1
    ))

    # Early stopping
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_average_precision',
        patience=config['training'].get('early_stopping_patience', 10),
        mode='max',
        verbose=1,
        restore_best_weights=True
    ))

    # Learning rate reduction
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mean_average_precision',
        factor=0.5,
        patience=5,
        mode='max',
        verbose=1
    ))

    # TensorBoard
    log_dir = os.path.join(output_dir, 'logs', 'tool_baseline')
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    ))

    return callbacks


def train_tool_detector(config):
    """Main training function for baseline tool detector."""
    print("=" * 60)
    print("Training Tool Detector Baseline (Task B - Visual Only)")
    print("=" * 60)

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(config['paths']['output_dir'], f'tool_baseline_{timestamp}')
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

    # Prepare batches for tool model
    train_ds = train_ds.map(
        prepare_batch_for_tool_model,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        prepare_batch_for_tool_model,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("   Datasets loaded successfully")

    # ========== MODEL ==========
    print("\n2. Creating model...")

    model_config = config['model']['tool_detector']

    model = create_tool_detector(
        hidden_dim=model_config.get('hidden_dim', 512),
        dropout_rate=config['training']['dropout_rate'],
        backbone_trainable_layers=model_config.get('backbone_trainable_layers', 1),
        input_shape=(480, 854, 3)
    )

    # Compile
    optimizer = create_optimizer(config)
    loss = FocalLoss(
        gamma=config['training'].get('focal_gamma', 2.0),
        alpha=config['training'].get('focal_alpha', 0.25)
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            MeanAveragePrecision(num_classes=NUM_TOOLS, name='mean_average_precision')
        ]
    )

    print("   Model created successfully")

    # Print summary
    print("\n   Model Architecture:")
    model.summary(print_fn=lambda x: print(f"   {x}"))

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

    # Evaluate on validation set and save per-tool metrics
    print("\n5. Computing per-tool metrics...")
    all_y_true = []
    all_y_pred = []

    for x, y in val_ds:
        preds = model(x, training=False)
        all_y_true.append(y.numpy())
        all_y_pred.append(preds.numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    metrics = compute_tool_metrics(
        tf.constant(all_y_true),
        tf.constant(all_y_pred)
    )

    # Print metrics
    print("\n   Per-tool metrics (threshold=0.5):")
    print(f"   {'Tool':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("   " + "-" * 47)
    for tool in TOOL_NAMES:
        m = metrics[tool]
        print(f"   {tool:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print("   " + "-" * 47)
    print(f"   {'Mean':<15} {metrics['overall']['mean_precision']:>10.3f} "
          f"{metrics['overall']['mean_recall']:>10.3f} {metrics['overall']['mean_f1']:>10.3f}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'tool_metrics.json')
    # Convert numpy values to Python types for JSON serialization
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            metrics_serializable[k] = {kk: float(vv) for kk, vv in v.items()}
        else:
            metrics_serializable[k] = float(v)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n   Metrics saved to: {metrics_path}")

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
    parser = argparse.ArgumentParser(description='Train Tool Detector Baseline (Task B)')
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
    train_tool_detector(config)


if __name__ == '__main__':
    main()
