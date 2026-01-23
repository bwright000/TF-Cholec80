"""
Tool Detector Model for MPHY0043 Coursework (Task B - Baseline).

This is the BASELINE model that detects surgical tools from visual input ONLY.
It does NOT use any timing information.

Architecture:
    RGB Frame → ResNet-50 → GAP → Dense(512) → Dense(7, sigmoid)

This baseline will be compared against the timed version (timed_tool_detector.py)
to evaluate whether timing information improves tool detection.

Tools detected (7 binary classifications):
    0: Grasper
    1: Bipolar
    2: Hook
    3: Scissors
    4: Clipper
    5: Irrigator
    6: SpecimenBag
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation
)

from .backbone import create_backbone, get_backbone_output_dim


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_TOOLS = 7
TOOL_NAMES = [
    'Grasper',
    'Bipolar',
    'Hook',
    'Scissors',
    'Clipper',
    'Irrigator',
    'SpecimenBag'
]


# ============================================================================
# TOOL DETECTOR MODEL (BASELINE - Visual Only)
# ============================================================================

def create_tool_detector(
    hidden_dim=512,
    dropout_rate=0.3,
    backbone_trainable_layers=1,
    input_shape=(480, 854, 3)
):
    """
    Create the baseline tool detection model (visual input only).

    This is a simple CNN classifier:
        Frame → ResNet-50 → GAP (2048-d) → Dense → Sigmoid (7 tools)

    Args:
        hidden_dim: Dimension of hidden dense layer
        dropout_rate: Dropout rate for regularization
        backbone_trainable_layers: Number of backbone layers to fine-tune
        input_shape: Shape of input images (H, W, C)

    Returns:
        Keras Model with:
            Input: frame (H, W, C)
            Output: tool_probabilities (7,) - probability for each tool
    """
    # ========== INPUT ==========
    frame_input = Input(shape=input_shape, name='frame', dtype=tf.uint8)

    # ========== BACKBONE ==========
    backbone = create_backbone(
        trainable_layers=backbone_trainable_layers,
        input_shape=input_shape
    )

    # Preprocess and extract features
    frame_float = tf.cast(frame_input, tf.float32)
    frame_preprocessed = tf.keras.applications.resnet50.preprocess_input(frame_float)
    visual_features = backbone(frame_preprocessed)  # (batch, 2048)

    # ========== CLASSIFICATION HEAD ==========
    # Hidden layer
    x = Dense(hidden_dim, name='hidden')(visual_features)
    x = BatchNormalization(name='hidden_bn')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    # Output layer - sigmoid for multi-label classification
    # (multiple tools can be present simultaneously)
    tool_output = Dense(
        NUM_TOOLS,
        activation='sigmoid',
        name='tool_predictions'
    )(x)

    # ========== BUILD MODEL ==========
    model = Model(
        inputs=frame_input,
        outputs=tool_output,
        name='tool_detector_baseline'
    )

    return model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss for handling class imbalance in tool detection.

    Focal loss down-weights easy examples and focuses on hard ones:
        FL(p) = -alpha * (1-p)^gamma * log(p)     for y=1
        FL(p) = -(1-alpha) * p^gamma * log(1-p)   for y=0

    Args:
        y_true: Ground truth labels (batch, num_tools)
        y_pred: Predicted probabilities (batch, num_tools)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for positive class

    Returns:
        Focal loss value
    """
    # Clip predictions to avoid log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Compute focal loss
    y_true = tf.cast(y_true, tf.float32)

    # For positive examples (y=1)
    pos_loss = -alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)

    # For negative examples (y=0)
    neg_loss = -(1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)

    # Combine based on ground truth
    loss = y_true * pos_loss + (1 - y_true) * neg_loss

    return tf.reduce_mean(loss)


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0):
    """
    Weighted binary cross-entropy for handling class imbalance.

    Gives higher weight to positive examples (tool present).

    Args:
        y_true: Ground truth labels (batch, num_tools)
        y_pred: Predicted probabilities (batch, num_tools)
        pos_weight: Weight multiplier for positive class

    Returns:
        Weighted BCE loss value
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_true = tf.cast(y_true, tf.float32)

    # Binary cross-entropy
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # Apply weights
    weights = y_true * pos_weight + (1 - y_true) * 1.0
    weighted_bce = weights * bce

    return tf.reduce_mean(weighted_bce)


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss as a Keras Loss class for easy use with model.compile().

    Args:
        gamma: Focusing parameter
        alpha: Weighting factor for positive class
    """

    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, gamma=self.gamma, alpha=self.alpha)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config


# ============================================================================
# METRICS
# ============================================================================

def compute_tool_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute per-tool and overall metrics.

    Args:
        y_true: Ground truth labels (batch, num_tools)
        y_pred: Predicted probabilities (batch, num_tools)
        threshold: Classification threshold

    Returns:
        Dictionary with precision, recall, f1, and accuracy per tool
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)

    metrics = {}

    for i, tool_name in enumerate(TOOL_NAMES):
        true_i = y_true[:, i]
        pred_i = y_pred_binary[:, i]

        # True positives, false positives, false negatives
        tp = tf.reduce_sum(true_i * pred_i)
        fp = tf.reduce_sum((1 - true_i) * pred_i)
        fn = tf.reduce_sum(true_i * (1 - pred_i))
        tn = tf.reduce_sum((1 - true_i) * (1 - pred_i))

        # Precision, Recall, F1
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        metrics[tool_name] = {
            'precision': precision.numpy(),
            'recall': recall.numpy(),
            'f1': f1.numpy(),
            'accuracy': accuracy.numpy()
        }

    # Overall metrics (micro-average)
    total_tp = sum(m['precision'] * 100 for m in metrics.values())  # Placeholder
    metrics['overall'] = {
        'mean_precision': sum(m['precision'] for m in metrics.values()) / NUM_TOOLS,
        'mean_recall': sum(m['recall'] for m in metrics.values()) / NUM_TOOLS,
        'mean_f1': sum(m['f1'] for m in metrics.values()) / NUM_TOOLS,
        'mean_accuracy': sum(m['accuracy'] for m in metrics.values()) / NUM_TOOLS
    }

    return metrics


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing Tool Detector (Baseline)...")
    print("=" * 60)

    # Create model with smaller input for testing
    print("\n1. Creating model...")
    model = create_tool_detector(
        hidden_dim=256,
        dropout_rate=0.3,
        backbone_trainable_layers=0,  # Frozen for speed
        input_shape=(224, 224, 3)  # Smaller for testing
    )

    # Model summary
    model.summary()

    # Test forward pass
    print("\n2. Testing forward pass...")
    dummy_input = tf.random.uniform(
        (4, 224, 224, 3),
        minval=0, maxval=255,
        dtype=tf.int32
    )
    dummy_input = tf.cast(dummy_input, tf.uint8)

    outputs = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output range: [{outputs.numpy().min():.3f}, {outputs.numpy().max():.3f}]")

    # Test loss functions
    print("\n3. Testing loss functions...")
    y_true = tf.constant([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0]
    ], dtype=tf.float32)

    y_pred = tf.constant([
        [0.9, 0.1, 0.8, 0.2, 0.1, 0.1, 0.1],
        [0.2, 0.7, 0.9, 0.1, 0.1, 0.1, 0.8],
        [0.8, 0.2, 0.3, 0.1, 0.7, 0.1, 0.2],
        [0.1, 0.1, 0.6, 0.2, 0.1, 0.9, 0.1]
    ], dtype=tf.float32)

    fl = focal_loss(y_true, y_pred)
    wbce = weighted_binary_crossentropy(y_true, y_pred)
    print(f"   Focal Loss: {fl.numpy():.4f}")
    print(f"   Weighted BCE: {wbce.numpy():.4f}")

    # Test metrics
    print("\n4. Testing metrics...")
    metrics = compute_tool_metrics(y_true, y_pred, threshold=0.5)
    print(f"   Overall Mean F1: {metrics['overall']['mean_f1']:.4f}")
    print(f"   Per-tool F1:")
    for tool in TOOL_NAMES[:3]:  # Show first 3
        print(f"      {tool}: {metrics[tool]['f1']:.4f}")

    # Parameter count
    print("\n5. Parameter count...")
    trainable = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    non_trainable = sum([tf.reduce_prod(v.shape).numpy() for v in model.non_trainable_variables])
    print(f"   Trainable: {trainable:,}")
    print(f"   Non-trainable: {non_trainable:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
