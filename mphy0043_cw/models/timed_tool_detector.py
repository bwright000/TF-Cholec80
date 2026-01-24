"""
Timed Tool Detector Model for MPHY0043 Coursework (Task B - With Timing).

This model detects surgical tools using BOTH visual input AND timing information.
The timing information comes from:
    1. Ground-truth timing labels (Oracle mode - upper bound)
    2. Predicted timing from Task A model (Realistic mode - actual deployment)

Architecture:
    Visual Branch:   Frame → ResNet-50 → 2048-d
    Timing Branch:   [remaining_time, phase_progress, phase_onehot] → Dense(64)
                        ↘              ↙
                           Concatenate
                                ↓
                    Dense(512) → Dense(7, sigmoid)

This will be compared against the baseline (tool_detector.py) to measure
the improvement from adding timing information.

Research Question: Does knowing where we are in the surgery help predict
which tools are being used?
"""

import tensorflow as tf
import keras
from keras import Model, ops
from keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Activation, Concatenate, Lambda
)

from .backbone import create_backbone, get_backbone_output_dim
from .tool_detector import NUM_TOOLS, TOOL_NAMES, focal_loss, FocalLoss


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_PHASES = 7


# ============================================================================
# TIMED TOOL DETECTOR MODEL
# ============================================================================

def create_timed_tool_detector(
    visual_hidden_dim=512,
    timing_hidden_dim=64,
    combined_hidden_dim=512,
    dropout_rate=0.3,
    backbone_trainable_layers=1,
    input_shape=(480, 854, 3),
    max_remaining_phase=5000.0,
    max_remaining_surgery=30000.0
):
    """
    Create the timed tool detection model (with timing information).

    Architecture:
        Visual Branch:
            Frame → ResNet-50 → 2048-d → Dense(visual_hidden_dim)

        Timing Branch:
            [remaining_phase, remaining_surgery, phase_progress, phase_onehot]
            → Dense(timing_hidden_dim)

        Combined:
            Concat(visual, timing) → Dense(combined_hidden_dim) → Dense(7, sigmoid)

    Args:
        visual_hidden_dim: Dimension for visual feature processing
        timing_hidden_dim: Dimension for timing feature processing
        combined_hidden_dim: Dimension for combined features
        dropout_rate: Dropout rate for regularization
        backbone_trainable_layers: Number of backbone layers to fine-tune
        input_shape: Shape of input images (H, W, C)
        max_remaining_phase: Normalization constant for remaining phase frames
            (from dataset statistics, ~5000 frames typical)
        max_remaining_surgery: Normalization constant for remaining surgery frames
            (from dataset statistics, ~30000 frames typical)

    Returns:
        Keras Model with inputs:
            - frame: (H, W, C) - the surgical image
            - remaining_phase: (1,) - frames left in current phase
            - remaining_surgery: (1,) - frames left in surgery
            - phase_progress: (1,) - progress through current phase (0-1)
            - phase: () - current phase ID (0-6)
        And output:
            - tool_predictions: (7,) - probability for each tool
    """
    # ========== INPUTS ==========
    # Visual input
    frame_input = Input(shape=input_shape, name='frame', dtype=tf.uint8)

    # Timing inputs
    remaining_phase_input = Input(shape=(1,), name='remaining_phase')
    remaining_surgery_input = Input(shape=(1,), name='remaining_surgery')
    phase_progress_input = Input(shape=(1,), name='phase_progress')
    phase_input = Input(shape=(), name='phase', dtype=tf.int32)

    # ========== VISUAL BRANCH ==========
    backbone = create_backbone(
        trainable_layers=backbone_trainable_layers,
        input_shape=input_shape
    )

    # Preprocess and extract features (Keras 3 compatible)
    frame_float = ops.cast(frame_input, 'float32')
    frame_preprocessed = Lambda(
        lambda x: tf.keras.applications.resnet50.preprocess_input(x)
    )(frame_float)
    visual_features = backbone(frame_preprocessed)  # (batch, 2048)

    # Process visual features
    visual_processed = Dense(visual_hidden_dim, name='visual_dense')(visual_features)
    visual_processed = BatchNormalization(name='visual_bn')(visual_processed)
    visual_processed = Activation('relu')(visual_processed)
    visual_processed = Dropout(dropout_rate)(visual_processed)

    # ========== TIMING BRANCH ==========
    # Normalize timing features using configurable constants from dataset statistics
    remaining_phase_norm = remaining_phase_input / max_remaining_phase
    remaining_surgery_norm = remaining_surgery_input / max_remaining_surgery

    # One-hot encode current phase (Keras 3 compatible)
    phase_onehot = ops.one_hot(phase_input, num_classes=NUM_PHASES)  # (batch, 7)

    # Concatenate all timing features
    timing_features = Concatenate(name='timing_concat')([
        remaining_phase_norm,
        remaining_surgery_norm,
        phase_progress_input,
        phase_onehot
    ])

    # Process timing features
    timing_processed = Dense(timing_hidden_dim, name='timing_dense1')(timing_features)
    timing_processed = BatchNormalization(name='timing_bn1')(timing_processed)
    timing_processed = Activation('relu')(timing_processed)
    timing_processed = Dropout(dropout_rate)(timing_processed)

    timing_processed = Dense(timing_hidden_dim, name='timing_dense2')(timing_processed)
    timing_processed = BatchNormalization(name='timing_bn2')(timing_processed)
    timing_processed = Activation('relu')(timing_processed)

    # ========== COMBINE BRANCHES ==========
    combined = Concatenate(name='feature_concat')([
        visual_processed,
        timing_processed
    ])

    # Combined processing
    combined = Dense(combined_hidden_dim, name='combined_dense')(combined)
    combined = BatchNormalization(name='combined_bn')(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(dropout_rate)(combined)

    # ========== OUTPUT ==========
    tool_output = Dense(
        NUM_TOOLS,
        activation='sigmoid',
        name='tool_predictions'
    )(combined)

    # ========== BUILD MODEL ==========
    model = Model(
        inputs={
            'frame': frame_input,
            'remaining_phase': remaining_phase_input,
            'remaining_surgery': remaining_surgery_input,
            'phase_progress': phase_progress_input,
            'phase': phase_input
        },
        outputs=tool_output,
        name='timed_tool_detector'
    )

    return model


# ============================================================================
# ATTENTION-BASED TIMING FUSION (Alternative Architecture)
# ============================================================================

def create_attention_timed_tool_detector(
    visual_hidden_dim=512,
    timing_hidden_dim=64,
    combined_hidden_dim=512,
    dropout_rate=0.3,
    backbone_trainable_layers=1,
    input_shape=(480, 854, 3),
    max_remaining_phase=5000.0,
    max_remaining_surgery=30000.0
):
    """
    Alternative architecture using attention to fuse timing and visual features.

    Instead of simple concatenation, this uses cross-attention to let
    timing information modulate visual features.

    Architecture:
        Visual → Query
        Timing → Key, Value
        Cross-Attention(Visual, Timing) → Fused Features → Classification
    """
    # ========== INPUTS ==========
    frame_input = Input(shape=input_shape, name='frame', dtype=tf.uint8)
    remaining_phase_input = Input(shape=(1,), name='remaining_phase')
    remaining_surgery_input = Input(shape=(1,), name='remaining_surgery')
    phase_progress_input = Input(shape=(1,), name='phase_progress')
    phase_input = Input(shape=(), name='phase', dtype=tf.int32)

    # ========== VISUAL BRANCH ==========
    backbone = create_backbone(
        trainable_layers=backbone_trainable_layers,
        input_shape=input_shape
    )

    frame_float = ops.cast(frame_input, 'float32')
    frame_preprocessed = Lambda(
        lambda x: tf.keras.applications.resnet50.preprocess_input(x)
    )(frame_float)
    visual_features = backbone(frame_preprocessed)

    # ========== TIMING BRANCH ==========
    # Normalize timing features using configurable constants from dataset statistics
    remaining_phase_norm = remaining_phase_input / max_remaining_phase
    remaining_surgery_norm = remaining_surgery_input / max_remaining_surgery
    phase_onehot = ops.one_hot(phase_input, num_classes=NUM_PHASES)

    timing_features = Concatenate()([
        remaining_phase_norm,
        remaining_surgery_norm,
        phase_progress_input,
        phase_onehot
    ])

    # Project timing to same dimension as visual_hidden_dim
    timing_projected = Dense(visual_hidden_dim, name='timing_proj')(timing_features)
    timing_projected = BatchNormalization()(timing_projected)
    timing_projected = Activation('relu')(timing_projected)

    # ========== ATTENTION FUSION ==========
    # Project visual features to visual_hidden_dim FIRST (to match timing_projected)
    # This ensures shapes are compatible for element-wise modulation
    # visual_features: (batch, 2048) -> visual_projected: (batch, visual_hidden_dim)
    visual_projected = Dense(visual_hidden_dim, name='visual_proj')(visual_features)
    visual_projected = Dropout(dropout_rate)(visual_projected)  # Regularization

    # Attention mechanism: simplified scalar attention for timing modulation
    # Uses element-wise product + sum instead of matmul(Q, K^T) for efficiency.
    # This produces a single attention weight per sample (not per position),
    # which is intentional for this fusion task where we want global timing
    # modulation rather than position-wise attention.
    visual_query = Dense(64, name='visual_query')(visual_projected)
    timing_key = Dense(64, name='timing_key')(timing_projected)

    # Scaled dot product attention (Keras 3 compatible)
    attention_scores = ops.sum(visual_query * timing_key, axis=-1, keepdims=True)
    attention_scores = attention_scores / 8.0  # sqrt(64) = 8
    attention_weights = Activation('sigmoid')(attention_scores)  # Soft gating [0, 1]

    # Bound timing contribution with tanh to prevent extreme modulation values
    # This keeps the multiplicative factor in range [1 - attention, 1 + attention]
    timing_bounded = Activation('tanh')(timing_projected)  # Bounded to [-1, 1]
    visual_modulated = visual_projected * (1.0 + attention_weights * timing_bounded)

    # ========== CLASSIFICATION HEAD ==========
    x = Dense(combined_hidden_dim)(visual_modulated)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    tool_output = Dense(NUM_TOOLS, activation='sigmoid', name='tool_predictions')(x)

    # ========== BUILD MODEL ==========
    model = Model(
        inputs={
            'frame': frame_input,
            'remaining_phase': remaining_phase_input,
            'remaining_surgery': remaining_surgery_input,
            'phase_progress': phase_progress_input,
            'phase': phase_input
        },
        outputs=tool_output,
        name='attention_timed_tool_detector'
    )

    return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_timing_inputs_from_predictions(time_predictor, frames, elapsed_phase, phases):
    """
    Get timing inputs for tool detector from Task A model predictions.

    This function bridges Task A (time prediction) and Task B (tool detection).
    In deployment, we use predicted timing rather than ground truth.

    Args:
        time_predictor: Trained Task A model
        frames: Input frames (batch, H, W, C)
        elapsed_phase: Elapsed time in phase (batch, 1)
        phases: Current phase IDs (batch,)

    Returns:
        Dictionary with timing inputs for timed_tool_detector
    """
    # Get predictions from time predictor
    # Note: Time predictor expects sequence input, so we add sequence dimension
    frames_seq = tf.expand_dims(frames, axis=1)  # (batch, 1, H, W, C)
    elapsed_seq = tf.expand_dims(elapsed_phase, axis=1)  # (batch, 1, 1)
    phases_seq = tf.expand_dims(phases, axis=1)  # (batch, 1)

    predictions = time_predictor({
        'frames': frames_seq,
        'elapsed_phase': elapsed_seq,
        'phase': phases_seq
    })

    # Extract predictions (remove sequence dimension)
    remaining_phase_pred = predictions['remaining_phase'][:, 0, :]  # (batch, 1)

    # Compute remaining surgery from remaining phase and future phases
    # This is an approximation; in practice you might predict this directly
    future_starts = predictions['future_phase_starts'][:, 0, :]  # (batch, 6)
    # Use max of future starts as estimate of remaining surgery
    remaining_surgery_pred = tf.reduce_max(future_starts, axis=-1, keepdims=True)
    remaining_surgery_pred = tf.maximum(remaining_surgery_pred, remaining_phase_pred)

    # Compute phase progress from elapsed and predicted remaining
    total_phase = elapsed_phase + remaining_phase_pred
    phase_progress_pred = elapsed_phase / (total_phase + 1.0)

    return {
        'remaining_phase': remaining_phase_pred,
        'remaining_surgery': remaining_surgery_pred,
        'phase_progress': phase_progress_pred,
        'phase': phases
    }


def prepare_timing_inputs_from_ground_truth(batch):
    """
    Extract timing inputs from ground truth batch data (Oracle mode).

    This is used for training and for computing the upper bound performance.

    Args:
        batch: Batch dictionary from dataloader with timing labels

    Returns:
        Dictionary with timing inputs for timed_tool_detector
    """
    return {
        'frame': batch['frame'],
        'remaining_phase': tf.cast(batch['remaining_phase'][:, tf.newaxis], tf.float32),
        'remaining_surgery': tf.cast(batch['remaining_surgery'][:, tf.newaxis], tf.float32),
        'phase_progress': batch['phase_progress'][:, tf.newaxis],
        'phase': batch['phase']
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing Timed Tool Detector...")
    print("=" * 60)

    # Create model with smaller input for testing
    print("\n1. Creating concatenation-based model...")
    model = create_timed_tool_detector(
        visual_hidden_dim=256,
        timing_hidden_dim=32,
        combined_hidden_dim=256,
        dropout_rate=0.3,
        backbone_trainable_layers=0,  # Frozen for speed
        input_shape=(224, 224, 3)  # Smaller for testing
    )

    model.summary()

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4

    dummy_frame = tf.random.uniform(
        (batch_size, 224, 224, 3),
        minval=0, maxval=255,
        dtype=tf.int32
    )
    dummy_frame = tf.cast(dummy_frame, tf.uint8)

    dummy_remaining_phase = tf.random.uniform((batch_size, 1), minval=0, maxval=5000)
    dummy_remaining_surgery = tf.random.uniform((batch_size, 1), minval=0, maxval=30000)
    dummy_phase_progress = tf.random.uniform((batch_size, 1), minval=0, maxval=1)
    dummy_phase = tf.random.uniform((batch_size,), minval=0, maxval=7, dtype=tf.int32)

    outputs = model({
        'frame': dummy_frame,
        'remaining_phase': dummy_remaining_phase,
        'remaining_surgery': dummy_remaining_surgery,
        'phase_progress': dummy_phase_progress,
        'phase': dummy_phase
    })

    print(f"   Output shape: {outputs.shape}")
    print(f"   Output range: [{outputs.numpy().min():.3f}, {outputs.numpy().max():.3f}]")

    # Test attention-based model
    print("\n3. Creating attention-based model...")
    attention_model = create_attention_timed_tool_detector(
        visual_hidden_dim=256,
        timing_hidden_dim=32,
        combined_hidden_dim=256,
        dropout_rate=0.3,
        backbone_trainable_layers=0,
        input_shape=(224, 224, 3)
    )

    attention_outputs = attention_model({
        'frame': dummy_frame,
        'remaining_phase': dummy_remaining_phase,
        'remaining_surgery': dummy_remaining_surgery,
        'phase_progress': dummy_phase_progress,
        'phase': dummy_phase
    })

    print(f"   Attention model output shape: {attention_outputs.shape}")

    # Compare parameter counts
    print("\n4. Parameter comparison...")
    concat_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    attention_params = sum([tf.reduce_prod(v.shape).numpy() for v in attention_model.trainable_variables])

    print(f"   Concatenation model: {concat_params:,} trainable params")
    print(f"   Attention model: {attention_params:,} trainable params")

    print("\n" + "=" * 60)
    print("All tests passed!")

# Terminal script to run this test: python -m mphy0043_cw.models.timed_tool_detector
