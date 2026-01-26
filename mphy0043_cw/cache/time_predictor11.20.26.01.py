"""
Time Predictor Model for MPHY0043 Coursework (Task A).

Predicts remaining time in current surgical phase and future phase start times.
Uses a State Space Model (SSM) inspired architecture based on Mamba for efficient temporal modeling.

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
# FRAME PREPROCESSING LAYER
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
    Parallel Selective State Space Model (SSM) Layer.
    
    This layer implements a Mamba-inspired State Space Model optimized for 
    high-throughput training on GPU clusters. It reformulates the 
    traditionally sequential SSM recurrence as a parallelizable convolution 
    using the Fast Fourier Transform (FFT).

    Mathematical Formulation:
        Recurrence: h(t) = A*h(t-1) + B(x)*x(t)
        Convolutional Form: y = x * K, where K = [B, AB, A^2B, ..., A^{L-1}B]
    
    Key Optimizations for GPU Clusters:
        1. FFT Convolution: Replaces O(L) sequential loops with O(L log L) 
           parallel operations, maximizing CUDA core utilization.
        2. Stable A-Parameterization: Learns A in log-space to ensure 
           system stability (negative real eigenvalues).
        3. Selective Mechanism: Uses input-dependent projections for B and C, 
           allowing the model to "forget" irrelevant frames or "focus" on 
           critical surgical transitions.

    Args:
        d_model (int): The number of expected features in the input (channel dimension).
        d_state (int): The dimension of the hidden state (latent space). 
            Higher values (e.g., 128-256) are recommended for high-end GPUs.
        dropout_rate (float): Dropout probability applied to the output projection.
        **kwargs: Standard Keras layer keyword arguments.

    Input shape:
        3D tensor with shape: `(batch, sequence_length, d_model)`.

    Output shape:
        3D tensor with shape: `(batch, sequence_length, d_model)`.
    """
    def __init__(self, d_model, d_state=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.seq_len = input_shape[1]

        if self.seq_len is None:
            raise ValueError("SSMLayer requires a fixed sequence length.")

        # FFT length = next power of 2 ≥ (2L − 1)
        self.n_fft = 2 ** int(np.ceil(np.log2(2 * self.seq_len - 1)))

        # Projections
        self.input_proj = Dense(self.d_model, name='input_proj')
        self.x_to_state = Dense(self.d_state, use_bias=False)
        self.B_proj = Dense(self.d_state, name='B_proj')
        self.C_proj = Dense(self.d_state, name='C_proj')

        self.A_log = self.add_weight(
            name='A_log',
            shape=(self.d_state,),
            initializer=tf.keras.initializers.RandomUniform(-4, -1),
            trainable=True
        )

        self.D = self.add_weight(
            name='D',
            shape=(self.d_model,),
            initializer='ones'
        )

        self.state_to_output = Dense(self.d_model, use_bias=False)
        self.output_proj = Dense(self.d_model)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

        super().build(input_shape)
    def call(self, x, training=None):
        # Store input dtype for mixed precision compatibility
        # TensorFlow FFT requires float32, so we cast as needed
        input_dtype = x.dtype

        residual = x
        x = self.layer_norm(x)

        z = self.input_proj(x)
        x_state = self.x_to_state(z)

        # SSM state computation (done in float32 for numerical stability)
        A = -tf.exp(tf.cast(self.A_log, tf.float32))
        timesteps = tf.range(self.seq_len, dtype=tf.float32)
        A_powers = tf.exp(A[None, :] * timesteps[:, None])
        
        def ssm_convolution(u, kernel):
            # u: (batch, seq_len, d_state)
            # kernel: (seq_len, d_state)

            # Cast to float32 for FFT (TensorFlow FFT requires float32)
            u_f32 = tf.cast(u, tf.float32)
            kernel_f32 = tf.cast(kernel, tf.float32)

            # Move time to last axis
            u_t = tf.transpose(u_f32, [0, 2, 1])        # (batch, d_state, seq)
            k_t = tf.transpose(kernel_f32, [1, 0])      # (d_state, seq)

            # FFT along time
            u_fft = tf.signal.rfft(u_t, fft_length=[self.n_fft])
            k_fft = tf.signal.rfft(k_t, fft_length=[self.n_fft])

            # Broadcast kernel across batch
            y_fft = u_fft * k_fft[None, :, :]

            # Inverse FFT
            y = tf.signal.irfft(y_fft, fft_length=[self.n_fft])

            # Trim and restore shape
            y = y[:, :, :self.seq_len]
            y = tf.transpose(y, [0, 2, 1])       # (batch, seq, d_state)

            # Cast back to input dtype for mixed precision
            return tf.cast(y, input_dtype)

        B = self.B_proj(z)
        C = self.C_proj(z)

        y_state = ssm_convolution(x_state * B, A_powers)
        y_state = y_state * C

        y = self.state_to_output(y_state)
        # Cast D to input dtype for mixed precision compatibility
        y = y + tf.cast(self.D, input_dtype) * z
        y = self.output_proj(y)
        y = self.dropout(y, training=training)

        return y + residual


# ============================================================================
# SSM BLOCK (Multiple SSM layers with feedforward)
# ============================================================================

class SSMBlock(Layer):
    """
    A block containing an SSM layer followed by a feedforward network.
    Similar to a Transformer block but with SSM instead of attention.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
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
    backbone_trainable_layers=0,
    input_shape=(480, 854, 3)
):
    """
    Create the time prediction model (Task A).

    Architecture:
        Frame sequence → Backbone (per-frame) → Feature sequence
        Feature sequence + timing features → SSM blocks → Predictions

    Args:
        sequence_length: Number of frames in input sequence
        d_model: Dimension of SSM model 
        d_state: SSM state dimension
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
    """
    # ========== INPUTS ==========
    # VIDEO FRAMES ONLY - No phase, elapsed time, or progress inputs
    # This ensures the model learns purely from visual cues (Task A requirement)
    frames_input = Input(
        shape=(sequence_length,) + input_shape,
        name='frames',
        dtype=tf.uint8
    )

    # ========== BACKBONE (Per-frame feature extraction) ==========
    backbone = create_backbone(
        trainable_layers=backbone_trainable_layers,
        input_shape=input_shape
    )

    # Process frames through backbone (Keras 3 compatible)
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

    # ========== PROJECT VISUAL FEATURES ==========
    # Project to model dimension (no other inputs - video only)
    combined = Dense(d_model, name='visual_projection')(visual_features)

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
    # VIDEO-ONLY MODEL: Predict timing directly from visual features
    # No elapsed time input, so we predict remaining_phase directly

    # Shared intermediate layer
    shared = Dense(128, activation='relu', name='shared_dense')(x)
    shared = Dropout(dropout_rate)(shared)

    # Head 1: Remaining time in current phase (direct prediction)
    # Using softplus to ensure positive output
    remaining_phase = Dense(
        1,
        activation='softplus',
        name='remaining_phase'
    )(shared)

    # Head 2: Future phase deltas (Recommendation 2 still applies)
    # Predict positive deltas between consecutive phases
    future_phase_deltas = Dense(
        NUM_PHASES - 1,  # 6 deltas (phase 0→1, 1→2, ..., 5→6)
        activation='softplus',  # Positive deltas enforce ordering
        name='future_phase_deltas'
    )(shared)

    # DERIVE absolute start times via cumsum
    # Starting from remaining_phase (time until current phase ends)
    future_phase_starts = ops.cumsum(future_phase_deltas, axis=-1)
    # Add offset: first future phase starts after current phase ends
    future_phase_starts = future_phase_starts + remaining_phase
    future_phase_starts = tf.keras.layers.Lambda(
        lambda x: x, name='future_phase_starts'
    )(future_phase_starts)

    # ========== BUILD MODEL ==========
    # VIDEO FRAMES ONLY - Pure visual learning for Task A
    model = Model(
        inputs={
        'frames': batch['frames'],
        'phase': batch['phase'],
        'tools': batch['tools'],
        },
        outputs={
        'remaining_phase': batch['remaining_phase'],
        'future_phase_starts': batch['future_phase_starts'],
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
    input_shape=(480, 854, 3),
    n_phases=7,
    n_tools=7,
    n_future_phases=6,
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
            - phase: (batch,) - current phase ID
        And outputs:
            - remaining_phase: (batch, 1)
            - future_phase_starts: (batch, 6)
    """
    # ========== INPUTS ==========
    frame_input = Input(shape=input_shape, name='frame', dtype=tf.uint8)
    phase = Input(shape=(sequence_length,),
                    dtype=tf.int32,
                    name="phase"),
    tools = Input(shape=(sequence_length, n_tools), name="tools")
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
    tool_embedding = Embedding(
        input_dim=NUM_TOOLS,
        output_dim=tool_embedding_dim,
        name='tool_embedding'
    )
    # Normalize elapsed time
    elapsed_normalized = elapsed_input / 30000.0  # Normalize by typical surgery length
    elapsed_expanded = ops.expand_dims(elapsed_normalized, axis=-1)  # (batch, 1)

    # ========== COMBINE FEATURES ==========
    combined = Concatenate(name='feature_concat')([
        visual_features,
        phase_embedding,
        tool_embedding,
        elapsed_expanded,
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
    # Cast to float32 for numerical stability (handles mixed precision)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

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
    # Cast to float32 for numerical stability (handles mixed precision)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

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


def total_duration_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss for total phase duration prediction.

    Total duration is a more stationary target than remaining time,
    leading to more stable gradients during training.

    Args:
        y_true: Ground truth total phase durations
        y_pred: Predicted total phase durations
        delta: Huber loss delta parameter

    Returns:
        Huber loss value
    """
    # Cast to float32 for numerical stability (handles mixed precision)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Standard Huber loss
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    return tf.reduce_mean(huber)


def delta_loss(y_true, y_pred, delta=1.0):
    """
    Loss for future phase delta predictions.

    Handles -1 values (phase already passed) by masking them out.
    Deltas are the time intervals between consecutive phase starts.

    Args:
        y_true: Ground truth phase deltas, -1 if phase already passed
        y_pred: Predicted phase deltas (always positive due to softplus)

    Returns:
        Masked Huber loss
    """
    # Cast to float32 for numerical stability (handles mixed precision)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Create mask for valid predictions (not -1)
    mask = tf.cast(y_true >= 0, tf.float32)

    # Huber loss on valid predictions only
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + delta * linear

    # Apply mask
    masked_loss = huber * mask

    # Average over valid predictions only
    n_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(masked_loss) / n_valid


def combined_time_loss_v2(y_true_total, y_pred_total,
                          y_true_deltas, y_pred_deltas,
                          duration_weight=1.0, delta_weight=0.5):
    """
    Combined loss for the new prediction approach
    Args:
        y_true_total: Ground truth total phase durations
        y_pred_total: Predicted total phase durations
        y_true_deltas: Ground truth future phase deltas
        y_pred_deltas: Predicted future phase deltas
        duration_weight: Weight for total duration loss
        delta_weight: Weight for delta loss

    Returns:
        Combined weighted loss
    """
    loss_duration = total_duration_loss(y_true_total, y_pred_total)
    loss_deltas = delta_loss(y_true_deltas, y_pred_deltas)

    return duration_weight * loss_duration + delta_weight * loss_deltas


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing Time Predictor Model with SSM...")
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
