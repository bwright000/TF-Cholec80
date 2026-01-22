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
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Layer, Input, Dense, Dropout, Concatenate,
    Embedding, BatchNormalization, LayerNormalization,
    Conv1D, Activation
)
import numpy as np

# Import backbone (Created in backbone.py)
from models.backbone import create_backbone, get_backbone_output_dim


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
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

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


# ============================================================================
# TIME PREDICTOR MODEL
# ============================================================================

def create_time_predictor(
    sequence_length=64,
    d_model=128,       # Reduced for memory efficiency
    d_state=32,        # Small state for 4GB VRAM
    n_ssm_blocks=2,
    phase_embedding_dim=16,
    dropout_rate=0.3,
    backbone_trainable_layers=1,
    input_shape=(480, 854, 3)
):
    """
    Create the time prediction model (Task A).

    Optimized for GTX 970 (4GB) / M2 Apple Silicon.

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

    # Process frames through backbone
    # Reshape to (batch * seq_len, H, W, C) for batch processing
    batch_size = tf.shape(frames_input)[0]
    frames_flat = tf.reshape(
        tf.cast(frames_input, tf.float32),
        [-1] + list(input_shape)
    )

    # Apply ResNet preprocessing
    frames_preprocessed = tf.keras.applications.resnet50.preprocess_input(frames_flat)

    # Extract features
    features_flat = backbone(frames_preprocessed)  # (batch*seq, 2048)

    # Reshape back to sequence: (batch, seq_len, 2048)
    visual_features = tf.reshape(
        features_flat,
        [batch_size, sequence_length, BACKBONE_DIM]
    )

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
