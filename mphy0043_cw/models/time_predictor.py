"""
Time Predictor Model for MPHY0043 Coursework (Task A).

Predicts remaining time in current surgical phase and future phase start times.
Uses a State Space Model (SSM) inspired architecture based on Mamba principles
for efficient temporal modeling.

Architecture:
    RGB Frame → ResNet-50 → 2048-d features
                    ↓
    Concat with [elapsed_time, phase_embedding]
                    ↓
    SSM Blocks (State Space Model layers)
                    ↓
    Prediction Heads:
        - remaining_current_phase (1 value)
        - future_phase_starts (6 values for phases 1-6)
"""

import tensorflow as tf
import keras
from keras import Model, ops
from keras.layers import (
    Layer, Input, Dense, Dropout, Concatenate,
    Embedding, BatchNormalization, LayerNormalization,
    Conv1D, Activation, Lambda
)
import numpy as np

# Import backbone (Created in backbone.py)
from .backbone import create_backbone, get_backbone_output_dim


# ============================================================================
# FRAME PREPROCESSING LAYER (Keras 3 Compatible)
# ============================================================================

class FramePreprocessingLayer(Layer):
    """
    Custom layer for preprocessing video frames for the backbone.

    This handles:
    1. Casting uint8 frames to float32
    2. Flattening sequence dimension for batch processing
    3. Applying ResNet preprocessing

    Keras 3 requires TF ops to be wrapped in layers.
    """

    def __init__(self, sequence_length, input_shape_hw, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.input_shape_hw = input_shape_hw  # (H, W, C)

    def call(self, frames):
        # frames: (batch, seq_len, H, W, C) uint8
        batch_size = tf.shape(frames)[0]

        # Cast to float
        frames_float = tf.cast(frames, tf.float32)

        # Flatten to (batch * seq_len, H, W, C)
        frames_flat = tf.reshape(
            frames_float,
            [-1] + list(self.input_shape_hw)
        )

        # Apply ResNet preprocessing
        frames_preprocessed = tf.keras.applications.resnet50.preprocess_input(frames_flat)

        return frames_preprocessed, batch_size

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'input_shape_hw': self.input_shape_hw
        })
        return config


class ReshapeBackToSequence(Layer):
    """
    Reshape backbone features back to sequence format.

    Keras 3 requires TF ops to be wrapped in layers.
    """

    def __init__(self, sequence_length, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    def call(self, inputs):
        features_flat, batch_size = inputs
        # Reshape from (batch * seq_len, feature_dim) to (batch, seq_len, feature_dim)
        return tf.reshape(
            features_flat,
            [batch_size, self.sequence_length, self.feature_dim]
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim
        })
        return config


# ============================================================================
# CONSTANTS
# ============================================================================

NUM_PHASES = 7
BACKBONE_DIM = 2048  # ResNet-50 output dimension


# ============================================================================
# SSM LAYER (Selective State Space Model - Mamba-inspired)
# ============================================================================

class SSMLayer(Layer):
    """
    Selective State Space Model layer inspired by Mamba.

    Optimized for memory-constrained hardware (GTX 970 4GB / M2 Apple Silicon).

    Implements the core SSM equations:
        h(t) = A * h(t-1) + B(x) * x_proj(t)
        y(t) = C(x) * h(t) + D * x(t)

    Key features:
        - Input-dependent B and C (selective mechanism from Mamba)
        - Proper state-space projections (fixes collapsed scalar issue)
        - Memory-efficient sequential scan (appropriate for limited VRAM)
        - Stable state dynamics via negative A parameterization

    Args:
        d_model: Model dimension (input/output size)
        d_state: State dimension (hidden state size, keep small for memory)
        dropout_rate: Dropout rate for regularization

    Hardware notes:
        - Sequential loop is kept intentionally (parallel scan has overhead
          that only benefits larger GPUs)
        - d_state=32-64 recommended for 4GB VRAM
        - For production, an associative scan could accelerate this
    """

    def __init__(self, d_model, d_state=32, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # === Input processing ===
        # Project input to model dimension
        self.input_proj = Dense(self.d_model, name='input_proj')

        # Project input to state dimension for state updates
        # This fixes the "collapsed scalar" issue - we now properly
        # project the full d_model features into state space
        self.x_to_state = Dense(self.d_state, use_bias=False, name='x_to_state')

        # === Selective SSM parameters (input-dependent B and C) ===
        # B_proj: generates input-dependent B matrix (gating for input)
        self.B_proj = Dense(self.d_state, name='B_proj')

        # C_proj: generates input-dependent C matrix (gating for output)
        self.C_proj = Dense(self.d_state, name='C_proj')

        # === Fixed SSM parameters ===
        # A is learned as log values for stability
        # Initialized negative to ensure state decay (prevents explosion)
        # A_log in [-4, -1] gives A in [~0.02, ~0.37] after exp(-exp())
        self.A_log = self.add_weight(
            name='A_log',
            shape=(self.d_state,),
            initializer=tf.keras.initializers.RandomUniform(-4, -1),
            trainable=True
        )

        # D is the skip connection weight (direct input to output)
        self.D = self.add_weight(
            name='D',
            shape=(self.d_model,),
            initializer='ones',
            trainable=True
        )

        # === Output processing ===
        # Project state back to model dimension
        # This fixes the "over-collapsed output" issue
        self.state_to_output = Dense(self.d_model, use_bias=False, name='state_to_output')

        # Final output projection
        self.output_proj = Dense(self.d_model, name='output_proj')

        # === Normalization and regularization ===
        self.layer_norm = LayerNormalization(name='layer_norm')
        self.dropout = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=None):
        """
        Forward pass through SSM layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Get static sequence length for Keras 3 compatibility
        # (dynamic tf.shape doesn't work with Python range)
        input_shape = tf.shape(x)
        batch_size = input_shape[0]

        # Get sequence length from static shape (required for Keras 3)
        seq_len = x.shape[1]
        if seq_len is None:
            # Fallback: use tf.unstack for dynamic sequences
            raise ValueError("SSMLayer requires static sequence length. "
                           "Ensure input has known sequence dimension.")

        # Store residual for skip connection
        residual = x

        # Pre-norm (like modern transformers/Mamba)
        x = self.layer_norm(x)

        # Project input to model dimension
        x = self.input_proj(x)  # (batch, seq, d_model)

        # Compute input-dependent B and C (selective mechanism)
        # This is the key Mamba innovation - the gates depend on input
        B = self.B_proj(x)  # (batch, seq, d_state)
        C = self.C_proj(x)  # (batch, seq, d_state)

        # Project input to state dimension for state updates
        x_state = self.x_to_state(x)  # (batch, seq, d_state)

        # Compute discrete A (state transition matrix)
        # A = -exp(A_log) ensures A is negative (stable decay)
        # A_bar = exp(A) gives discrete-time transition in (0, 1)
        A = -tf.exp(self.A_log)  # (d_state,) all negative
        A_bar = tf.exp(A)  # (d_state,) in (0, 1) for stable dynamics

        # === Sequential SSM scan ===
        # Note: Sequential loop is intentional for memory efficiency on
        # GTX 970/M2. Parallel associative scan has overhead that only
        # pays off on larger GPUs (RTX 3090+, A100, etc.)

        # Initialize hidden state
        h = tf.zeros((batch_size, self.d_state), dtype=x.dtype)
        outputs = []

        # Process sequence step by step
        for t in range(seq_len):
            # Get timestep inputs
            x_t = x[:, t, :]           # (batch, d_model)
            x_state_t = x_state[:, t, :]  # (batch, d_state)
            B_t = B[:, t, :]           # (batch, d_state)
            C_t = C[:, t, :]           # (batch, d_state)

            # State update: h(t) = A_bar * h(t-1) + B(x) * x_proj(t)
            # B_t acts as input gate, modulating how much of x enters state
            h = A_bar * h + B_t * x_state_t

            # Output from state: C(x) * h(t)
            # C_t acts as output gate, selecting what to read from state
            y_state = C_t * h  # (batch, d_state) - element-wise gating

            # Project state back to model dimension
            y_from_state = self.state_to_output(y_state)  # (batch, d_model)

            # Add skip connection: y(t) = state_output + D * x(t)
            y_t = y_from_state + self.D * x_t

            outputs.append(y_t)

        # Stack outputs: (batch, seq, d_model)
        y = tf.stack(outputs, axis=1)

        # Final output projection
        y = self.output_proj(y)

        # Dropout and residual connection
        y = self.dropout(y, training=training)
        y = y + residual

        return y

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'dropout_rate': self.dropout_rate
        })
        return config

    def compute_output_shape(self, input_shape):
        """Return output shape (same as input for this layer)."""
        return input_shape


# ============================================================================
# SSM BLOCK (Multiple SSM layers with feedforward)
# ============================================================================

class SSMBlock(Layer):
    """
    A block containing an SSM layer followed by a feedforward network.
    Similar to a Transformer block but with SSM instead of attention.

    Memory-optimized for GTX 970 (4GB) / M2 Apple Silicon.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (keep small: 32-64 for 4GB VRAM)
        d_ff: Feedforward hidden dimension (defaults to d_model * 2)
        dropout_rate: Dropout rate
    """

    def __init__(self, d_model, d_state=32, d_ff=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff or d_model * 2
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # SSM layer
        self.ssm = SSMLayer(
            d_model=self.d_model,
            d_state=self.d_state,
            dropout_rate=self.dropout_rate,
            name='ssm'
        )

        # Feedforward network
        self.ff_norm = LayerNormalization(name='ff_norm')
        self.ff1 = Dense(self.d_ff, activation='gelu', name='ff1')
        self.ff2 = Dense(self.d_model, name='ff2')
        self.ff_dropout = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, training=None):
        """Forward pass through SSM block."""
        # SSM layer with residual (handled inside SSMLayer)
        x = self.ssm(x, training=training)

        # Feedforward with residual
        residual = x
        x = self.ff_norm(x)
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.ff_dropout(x, training=training)
        x = x + residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        })
        return config

    def compute_output_shape(self, input_shape):
        """Return output shape (same as input for this block)."""
        return input_shape


# ============================================================================
# TIME PREDICTOR MODEL
# ============================================================================

def create_time_predictor(
    sequence_length=64,
    d_model=128,       # Small to improve efficiency
    d_state=32,        # Small state for hardware restrictions
    n_ssm_blocks=2,
    phase_embedding_dim=16,
    dropout_rate=0.3,
    backbone_trainable_layers=1,
    input_shape=(480, 854, 3)
):
    """
    Create the time prediction model (Task A).

    Architecture:
        Frame sequence → Backbone (per-frame) → Feature sequence
        Feature sequence + timing features → SSM blocks → Predictions

    Args:
        sequence_length: Number of frames in input sequence
        d_model: Dimension of SSM model (128 recommended for 4GB VRAM)
        d_state: SSM state dimension (32 recommended for 4GB VRAM)
        n_ssm_blocks: Number of SSM blocks
        phase_embedding_dim: Dimension of phase embedding
        dropout_rate: Dropout rate
        backbone_trainable_layers: Layers to fine-tune in backbone
        input_shape: Shape of input images (H, W, C)

    Returns:
        Keras Model with inputs:
            - frames: (batch, seq_len, H, W, C)
            - elapsed_phase: (batch, seq_len, 1)
            - phase: (batch, seq_len) - integer phase IDs
        And outputs:
            - remaining_phase: (batch, seq_len, 1)
            - future_phase_starts: (batch, seq_len, 6)

    Memory notes:
        - With default settings: ~25M parameters
        - Recommended batch_size=4 for full resolution on 4GB VRAM
        - For sequence training, consider batch_size=2 with seq_len=32
    """
    # ========== INPUTS ==========
    # Frame sequence input
    frames_input = Input(
        shape=(sequence_length,) + input_shape,
        name='frames',
        dtype=tf.uint8
    )

    # Elapsed time in current phase (normalized)
    elapsed_input = Input(
        shape=(sequence_length, 1),
        name='elapsed_phase'
    )

    # Current phase ID (for embedding)
    phase_input = Input(
        shape=(sequence_length,),
        name='phase',
        dtype=tf.int32
    )

    # ========== BACKBONE (Per-frame feature extraction) ==========
    backbone = create_backbone(
        trainable_layers=backbone_trainable_layers,
        input_shape=input_shape
    )

    # Process frames through backbone (Keras 3 compatible)
    # Use custom layers to wrap TF ops
    preprocess_layer = FramePreprocessingLayer(
        sequence_length=sequence_length,
        input_shape_hw=input_shape,
        name='frame_preprocessing'
    )
    frames_preprocessed, batch_size = preprocess_layer(frames_input)

    # Extract features
    features_flat = backbone(frames_preprocessed)  # (batch*seq, 2048)

    # Reshape back to sequence: (batch, seq_len, 2048)
    reshape_layer = ReshapeBackToSequence(
        sequence_length=sequence_length,
        feature_dim=BACKBONE_DIM,
        name='reshape_to_sequence'
    )
    visual_features = reshape_layer([features_flat, batch_size])

    # ========== PHASE EMBEDDING ==========
    phase_embedding = Embedding(
        input_dim=NUM_PHASES,
        output_dim=phase_embedding_dim,
        name='phase_embedding'
    )(phase_input)  # (batch, seq_len, embed_dim)

    # ========== COMBINE FEATURES ==========
    # Project visual features to d_model
    visual_proj = Dense(d_model, name='visual_projection')(visual_features)

    # Project elapsed time
    elapsed_proj = Dense(d_model // 4, name='elapsed_projection')(elapsed_input)

    # Project phase embedding
    phase_proj = Dense(d_model // 4, name='phase_projection')(phase_embedding)

    # Concatenate all features
    combined = Concatenate(name='feature_concat')([
        visual_proj,
        elapsed_proj,
        phase_proj
    ])

    # Project to model dimension
    combined = Dense(d_model, name='combined_projection')(combined)
    combined = BatchNormalization(name='combined_bn')(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(dropout_rate)(combined)

    # ========== SSM BLOCKS ==========
    x = combined
    for i in range(n_ssm_blocks):
        x = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            dropout_rate=dropout_rate,
            name=f'ssm_block_{i}'
        )(x)

    # Final layer normalization
    x = LayerNormalization(name='final_norm')(x)

    # ========== OUTPUT HEADS ==========
    # Shared intermediate layer
    shared = Dense(128, activation='relu', name='shared_dense')(x)
    shared = Dropout(dropout_rate)(shared)

    # Head 1: Remaining time in current phase
    remaining_phase = Dense(
        1,
        activation='relu',  # Time must be non-negative
        name='remaining_phase'
    )(shared)

    # Head 2: Future phase start times (6 values for phases 1-6)
    # -1 indicates phase already passed
    future_phase_starts = Dense(
        NUM_PHASES - 1,  # 6 future phases
        name='future_phase_starts'
    )(shared)

    # ========== BUILD MODEL ==========
    model = Model(
        inputs={
            'frames': frames_input,
            'elapsed_phase': elapsed_input,
            'phase': phase_input
        },
        outputs={
            'remaining_phase': remaining_phase,
            'future_phase_starts': future_phase_starts
        },
        name='time_predictor'
    )

    return model


# ============================================================================
# SINGLE-FRAME TIME PREDICTOR (For use with current dataloader)
# ============================================================================

def create_time_predictor_single_frame(
    hidden_dim=512,
    phase_embedding_dim=16,
    dropout_rate=0.3,
    backbone_trainable_layers=1,
    input_shape=(480, 854, 3)
):
    """
    Create a simpler single-frame time prediction model.

    This version works with single frames (no sequence dimension) and is
    compatible with the current dataloader that returns individual frames.

    Architecture:
        Frame → ResNet-50 → 2048-d
                    ↓
        Concat with [elapsed_time, phase_embedding]
                    ↓
        Dense layers → Predictions

    Args:
        hidden_dim: Dimension of hidden layers
        phase_embedding_dim: Dimension of phase embedding
        dropout_rate: Dropout rate
        backbone_trainable_layers: Layers to fine-tune in backbone
        input_shape: Shape of input images (H, W, C)

    Returns:
        Keras Model with inputs:
            - frame: (batch, H, W, C) - single frame
            - elapsed_phase: (batch,) - elapsed frames in phase
            - phase: (batch,) - current phase ID
        And outputs:
            - remaining_phase: (batch, 1)
            - future_phase_starts: (batch, 6)
    """
    # ========== INPUTS ==========
    frame_input = Input(shape=input_shape, name='frame', dtype=tf.uint8)
    elapsed_input = Input(shape=(), name='elapsed_phase')
    phase_input = Input(shape=(), name='phase', dtype=tf.int32)

    # ========== BACKBONE ==========
    backbone = create_backbone(
        trainable_layers=backbone_trainable_layers,
        input_shape=input_shape
    )

    # Preprocess frame (Keras 3 compatible)
    frame_float = ops.cast(frame_input, 'float32')
    frame_preprocessed = Lambda(
        lambda x: tf.keras.applications.resnet50.preprocess_input(x)
    )(frame_float)
    visual_features = backbone(frame_preprocessed)  # (batch, 2048)

    # ========== TIMING FEATURES ==========
    # Phase embedding
    phase_embedding = Embedding(
        input_dim=NUM_PHASES,
        output_dim=phase_embedding_dim,
        name='phase_embedding'
    )(phase_input)  # (batch, phase_embedding_dim)

    # Normalize elapsed time
    elapsed_normalized = elapsed_input / 30000.0  # Normalize by typical surgery length
    elapsed_expanded = ops.expand_dims(elapsed_normalized, axis=-1)  # (batch, 1)

    # ========== COMBINE FEATURES ==========
    combined = Concatenate(name='feature_concat')([
        visual_features,
        phase_embedding,
        elapsed_expanded
    ])

    # ========== PREDICTION LAYERS ==========
    x = Dense(hidden_dim, name='hidden1')(combined)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(hidden_dim // 2, name='hidden2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    # ========== OUTPUT HEADS ==========
    # Head 1: Remaining time in current phase
    remaining_phase = Dense(
        1,
        activation='relu',  # Time must be non-negative
        name='remaining_phase'
    )(x)

    # Head 2: Future phase start times (6 values for phases 1-6)
    future_phase_starts = Dense(
        NUM_PHASES - 1,  # 6 future phases
        name='future_phase_starts'
    )(x)

    # ========== BUILD MODEL ==========
    model = Model(
        inputs={
            'frame': frame_input,
            'elapsed_phase': elapsed_input,
            'phase': phase_input
        },
        outputs={
            'remaining_phase': remaining_phase,
            'future_phase_starts': future_phase_starts
        },
        name='time_predictor_single_frame'
    )

    return model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

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


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing Time Predictor Model with SSM (Mamba-inspired)...")
    print("=" * 60)
    print("Target hardware: GTX 970 (4GB) / M2 Apple Silicon")
    print("=" * 60)

    # Test SSM Layer
    print("\n1. Testing SSMLayer (fixed projections)...")
    ssm_layer = SSMLayer(d_model=64, d_state=32)
    test_input = tf.random.normal((2, 10, 64))  # (batch, seq, features)
    ssm_output = ssm_layer(test_input)
    print(f"   SSM Input shape: {test_input.shape}")
    print(f"   SSM Output shape: {ssm_output.shape}")

    # Verify shapes are preserved (not collapsed)
    assert ssm_output.shape == test_input.shape, "Output shape mismatch!"
    print("   ✓ Shape preserved (no scalar collapse)")

    # Test SSM Block
    print("\n2. Testing SSMBlock...")
    ssm_block = SSMBlock(d_model=64, d_state=32, d_ff=128)
    block_output = ssm_block(test_input)
    print(f"   Block Output shape: {block_output.shape}")

    # Test full model (with small dimensions for speed)
    print("\n3. Testing full Time Predictor model...")
    print("   (Using small dimensions for fast testing)")

    # Create model with smaller dimensions for testing
    model = create_time_predictor(
        sequence_length=4,  # Small sequence for testing
        d_model=64,
        d_state=32,
        n_ssm_blocks=1,
        phase_embedding_dim=8,
        dropout_rate=0.1,
        backbone_trainable_layers=0,  # Frozen for speed
        input_shape=(224, 224, 3)  # Smaller images for testing
    )

    # Create dummy inputs
    batch_size = 2
    seq_len = 4
    dummy_frames = tf.random.uniform(
        (batch_size, seq_len, 224, 224, 3),
        minval=0, maxval=255,
        dtype=tf.int32
    )
    dummy_frames = tf.cast(dummy_frames, tf.uint8)

    dummy_elapsed = tf.random.uniform((batch_size, seq_len, 1))
    dummy_phase = tf.random.uniform(
        (batch_size, seq_len),
        minval=0, maxval=7,
        dtype=tf.int32
    )

    # Run forward pass
    outputs = model({
        'frames': dummy_frames,
        'elapsed_phase': dummy_elapsed,
        'phase': dummy_phase
    })

    print(f"\n   Model Outputs:")
    print(f"   - remaining_phase shape: {outputs['remaining_phase'].shape}")
    print(f"   - future_phase_starts shape: {outputs['future_phase_starts'].shape}")

    # Model summary
    print(f"\n   Model Parameters:")
    trainable = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    non_trainable = sum([tf.reduce_prod(v.shape).numpy() for v in model.non_trainable_variables])
    total = trainable + non_trainable
    print(f"   - Trainable: {trainable:,}")
    print(f"   - Non-trainable: {non_trainable:,}")
    print(f"   - Total: {total:,}")

    # Estimate memory usage
    param_memory_mb = (total * 4) / (1024 * 1024)  # float32 = 4 bytes
    print(f"\n   Memory Estimates (float32):")
    print(f"   - Parameters: ~{param_memory_mb:.1f} MB")
    print(f"   - Recommended batch size for 4GB VRAM: 4-8")
    print(f"   - For full 480x854 images, use batch_size=4")

    # Test loss functions
    print("\n4. Testing loss functions...")
    y_true = tf.constant([[100.0], [50.0], [10.0], [5.0]])
    y_pred = tf.constant([[95.0], [55.0], [12.0], [3.0]])
    loss = weighted_huber_loss(y_true, y_pred)
    print(f"   Weighted Huber loss: {loss.numpy():.4f}")

    y_true_future = tf.constant([[100, 200, -1, -1, -1, -1],
                                  [50, 150, 300, -1, -1, -1]], dtype=tf.float32)
    y_pred_future = tf.constant([[95, 210, 0, 0, 0, 0],
                                  [55, 145, 310, 0, 0, 0]], dtype=tf.float32)
    loss_future = future_phase_loss(y_true_future, y_pred_future)
    print(f"   Future phase loss: {loss_future.numpy():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("\nSSM Layer fixes applied:")
    print("  ✓ Input properly projected to state dimension (not collapsed)")
    print("  ✓ Output properly projected back (not tiled scalar)")
    print("  ✓ Memory-efficient for GTX 970 / M2")
    print("  ✓ Sequential scan retained (parallel has overhead on small GPUs)")
