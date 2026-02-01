"""
Time Predictor Model for MPHY0043 Coursework (Task A).

Predicts remaining time in current surgical phase and future phase start times.
Uses an S4-style Linear State Space Model (SSM) for efficient temporal modeling.

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

S4 Enhancements:
    - HiPPO-LegS initialization for state matrix A
    - Learnable discretization step Δ per channel
    - Bilinear (Tustin) discretization method
"""

import tensorflow as tf
import keras
from keras import Model, ops
from keras.layers import (
    Layer, Input, Dense, Dropout, Concatenate,
    Embedding, BatchNormalization, LayerNormalization,
    Activation
)
import numpy as np

# Import backbone (Created in backbone.py)
from .backbone import create_backbone, get_backbone_output_dim

# Import dataset constants (single source of truth)
from mphy0043_cw.data.dataset import NUM_PHASES

# Import loss functions from losses.py (single source of truth)
# Re-exported here for backward compatibility with train_time.py imports
from .losses import (
    weighted_huber_loss, future_phase_loss, combined_time_loss,
    huber_loss, normalize_targets, denormalize_predictions,
    MAX_REMAINING_PHASE, MAX_REMAINING_SURGERY, MAX_FUTURE_PHASE_START
)


# ============================================================================
# FRAME PREPROCESSING LAYERS
# Note: Keras 3 requires TF ops to be wrapped in custom layers.
# ============================================================================

class FramePreprocessingLayer(Layer):
    """Preprocess video frames: cast to float32, flatten sequence, apply ResNet preprocessing."""

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
    """Reshape backbone features back to sequence format."""

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


class BackboneWrapper(Layer):
    """
    Wrapper layer for backbone that handles training mode correctly.

    When the backbone is frozen (trainable_layers=0), this wrapper ensures
    the backbone is called with training=False to avoid BatchNormalization
    momentum updates, which significantly slows down training.
    """

    def __init__(self, backbone, trainable_layers, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.trainable_layers = trainable_layers

    def call(self, x, training=None):
        # Use inference mode if backbone is completely frozen
        # This prevents BN layers from updating running statistics
        if self.trainable_layers == 0:
            return self.backbone(x, training=False)
        return self.backbone(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'trainable_layers': self.trainable_layers
        })
        return config


# ============================================================================
# CONSTANTS
# ============================================================================

# NUM_PHASES imported from mphy0043_cw.data.dataset
BACKBONE_DIM = 2048  # ResNet-50 output dimension


# ============================================================================
# HiPPO INITIALIZATION
# ============================================================================

def hippo_legs_initializer(N):
    """
    HiPPO-LegS (Legendre Scaled) diagonal initialization for S4D-style SSMs.

    The full HiPPO-LegS matrix has diagonal elements -(n+1) for n = 0, ..., N-1,
    giving [-1, -2, -3, ..., -N]. Since we parameterize A = -exp(A_log),
    we store A_log = log(n+1) so that A = -exp(log(n+1)) = -(n+1).

    This initialization provides:
    - Provably optimal history compression via Legendre polynomials
    - Stable eigenvalues (negative real parts)
    - Better gradient flow over long sequences

    Args:
        N: State dimension (d_state)

    Returns:
        Log-space diagonal values, shape (N,)
    """
    # HiPPO-LegS diagonal magnitudes: [1, 2, 3, ..., N]
    # When recovered as A = -exp(A_log), gives A = [-1, -2, -3, ..., -N]
    diag_magnitudes = np.arange(1, N + 1, dtype=np.float32)
    return np.log(diag_magnitudes)


# ============================================================================
# SSM LAYER (S4-style Linear State Space Model)
# ============================================================================

class SSMLayer(Layer):
    """
    S4-style Linear State Space Model (SSM) Layer.

    This layer implements an S4-inspired State Space Model optimized for
    high-throughput training on GPU clusters. It uses FFT-based convolution
    to efficiently compute the SSM recurrence in parallel.

    Mathematical Formulation:
        Continuous-time:  h'(t) = A*h(t) + B*x(t),  y(t) = C*h(t) + D*x(t)

        After bilinear (Tustin) discretization with step Δ:
        Discrete-time:    h[k] = Ā*h[k-1] + B̄*x[k],  y[k] = C*h[k] + D*x[k]

        where for diagonal A:
            Ā = (1 + Δ·A/2) / (1 - Δ·A/2)
            B̄ = Δ / (1 - Δ·A/2) · B

    Convolutional Form:
        y = x * K, where K = [C̄B̄, C̄ĀB̄, C̄Ā²B̄, ..., C̄Ā^{L-1}B̄]

    Key Features:
        1. HiPPO Initialization: State matrix A initialized with HiPPO-LegS
           for optimal long-range memory.
        2. Learnable Δ: Per-channel discretization step enables multi-scale
           temporal modeling - critical for surgical videos with varying
           phase durations.
        3. Bilinear Discretization: Tustin method provides better numerical
           stability than Euler discretization.
        4. FFT Convolution: O(L log L) parallel operations for sequence length L.

    Args:
        d_model (int): The number of expected features in the input (channel dimension).
        d_state (int): The dimension of the hidden state (latent space).
        dropout_rate (float): Dropout probability applied to the output projection.
        use_hippo_init (bool): Whether to use HiPPO initialization for A matrix.
        delta_min (float): Minimum value for learnable Δ initialization.
        delta_max (float): Maximum value for learnable Δ initialization.
        **kwargs: Standard Keras layer keyword arguments.

    Input shape:
        3D tensor with shape: `(batch, sequence_length, d_model)`.

    Output shape:
        3D tensor with shape: `(batch, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        dropout_rate=0.1,
        use_hippo_init=True,
        delta_min=0.001,
        delta_max=None,  # Auto-computed for stability if None
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.dropout_rate = dropout_rate
        self.use_hippo_init = use_hippo_init
        self.delta_min = delta_min

        # Stability constraint for bilinear discretization with HiPPO:
        # Bilinear transform requires |Δ·A/2| < 1 for stability.
        # With HiPPO, max|A| = d_state, so delta_max < 2/d_state.
        # We use 0.8 * (2/d_state) for safety margin.
        if delta_max is None:
            if use_hippo_init:
                # Constrained for HiPPO stability
                self.delta_max = 0.8 * (2.0 / d_state)
            else:
                # More relaxed for random init (smaller |A| values)
                self.delta_max = 0.1
        else:
            self.delta_max = delta_max

    def build(self, input_shape):
        self.seq_len = input_shape[1]

        if self.seq_len is None:
            raise ValueError("SSMLayer requires a fixed sequence length.")

        # FFT length = next power of 2 ≥ (2L − 1)
        self.n_fft = 2 ** int(np.ceil(np.log2(2 * self.seq_len - 1)))

        # Input projection
        self.input_proj = Dense(self.d_model, name='input_proj')
        self.x_to_state = Dense(self.d_state, use_bias=False)

        # ================================================================
        # SSM Parameters (constant, not input-dependent for LTI system)
        # ================================================================

        # A: State transition matrix (learned in log-space for stability)
        # Using HiPPO-LegS initialization for optimal long-range memory
        if self.use_hippo_init:
            hippo_A_log = hippo_legs_initializer(self.d_state)
            A_initializer = tf.keras.initializers.Constant(hippo_A_log)
        else:
            # Fallback to random initialization
            A_initializer = tf.keras.initializers.RandomUniform(-4, -1)

        self.A_log = self.add_weight(
            name='A_log',
            shape=(self.d_state,),
            initializer=A_initializer,
            trainable=True
        )

        # Δ (Delta): Learnable discretization step per channel
        # Enables multi-scale temporal modeling - different channels can
        # learn different timescales (critical for surgical phases with
        # varying durations from seconds to tens of minutes)
        self.log_delta = self.add_weight(
            name='log_delta',
            shape=(self.d_state,),
            initializer=tf.keras.initializers.RandomUniform(
                np.log(self.delta_min),
                np.log(self.delta_max)
            ),
            trainable=True
        )

        # B: Input-to-state matrix (constant, enables valid FFT convolution)
        # S4D paper: "B = 1 then trained" - uniform input-to-state mapping initially
        self.B = self.add_weight(
            name='B',
            shape=(self.d_state,),
            initializer='ones',
            trainable=True
        )

        # C: State-to-output matrix (constant, enables valid FFT convolution)
        # S4D paper: "C random with std=1 then trained" - proper gradient scale
        self.C = self.add_weight(
            name='C',
            shape=(self.d_state,),
            initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
            trainable=True
        )

        # D: Direct feedthrough (skip connection)
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'dropout_rate': self.dropout_rate,
            'use_hippo_init': self.use_hippo_init,
            'delta_min': self.delta_min,
            'delta_max': self.delta_max,
        })
        return config

    def call(self, x, training=None):
        # Store input dtype for mixed precision compatibility
        # TensorFlow FFT requires float32, so we cast as needed
        input_dtype = x.dtype

        residual = x
        x = self.layer_norm(x)

        z = self.input_proj(x)
        x_state = self.x_to_state(z)

        # ================================================================
        # SSM State Computation (Casted to float32 for TensorFlow FFT
        # ================================================================

        # Get continuous-time A (negative for stability)
        A = -tf.exp(tf.cast(self.A_log, tf.float32))  # (d_state,)

        # Get learnable discretization step Δ (positive via softplus)
        delta = tf.nn.softplus(tf.cast(self.log_delta, tf.float32))  # (d_state,)

        # ================================================================
        # Bilinear (Tustin) Discretization
        # For diagonal A, the bilinear discretization simplifies to:
        #   Ā = (1 + Δ·A/2) / (1 - Δ·A/2)
        #   B̄ = Δ / (1 - Δ·A/2) · B
        # ================================================================

        # Compute discretization terms
        delta_A_half = delta * A / 2.0  # (d_state,)

        # Discretized A: Ā = (1 + Δ·A/2) / (1 - Δ·A/2)
        A_bar = (1.0 + delta_A_half) / (1.0 - delta_A_half)  # (d_state,)

        # Discretized B: B̄ = Δ / (1 - Δ·A/2) · B
        B_bar = (delta / (1.0 - delta_A_half)) * tf.cast(self.B, tf.float32)  # (d_state,)

        # ================================================================
        # Compute SSM Convolution Kernel
        # K[t] = Ā^t · B̄ for t = 0, 1, ..., L-1
        # ================================================================

        timesteps = tf.range(self.seq_len, dtype=tf.float32)  # (seq_len,)

        # Compute Ā^t for each timestep
        # A_bar: (d_state,), timesteps: (seq_len,)
        # Result: (seq_len, d_state)
        A_bar_powers = tf.pow(
            A_bar[None, :],      # (1, d_state)
            timesteps[:, None]   # (seq_len, 1)
        )  # (seq_len, d_state)

        # Compute kernel: K[t] = Ā^t · B̄
        kernel = A_bar_powers * B_bar[None, :]  # (seq_len, d_state)

        def ssm_convolution(u, kernel):
            """
            Compute SSM convolution using FFT for O(L log L) complexity.

            Args:
                u: Input tensor (batch, seq_len, d_state)
                kernel: SSM kernel (seq_len, d_state)

            Returns:
                Convolved output (batch, seq_len, d_state)
            """
            # Cast to float32 for FFT (TensorFlow FFT requires float32)
            u_f32 = tf.cast(u, tf.float32)
            kernel_f32 = tf.cast(kernel, tf.float32)

            # Move time to last axis for FFT
            u_t = tf.transpose(u_f32, [0, 2, 1])        # (batch, d_state, seq)
            k_t = tf.transpose(kernel_f32, [1, 0])      # (d_state, seq)

            # FFT along time dimension
            u_fft = tf.signal.rfft(u_t, fft_length=[self.n_fft])
            k_fft = tf.signal.rfft(k_t, fft_length=[self.n_fft])

            # Pointwise multiplication in frequency domain (convolution theorem)
            y_fft = u_fft * k_fft[None, :, :]

            # Inverse FFT to get convolution result
            y = tf.signal.irfft(y_fft, fft_length=[self.n_fft])

            # Trim to original sequence length and restore shape
            y = y[:, :, :self.seq_len]
            y = tf.transpose(y, [0, 2, 1])  # (batch, seq, d_state)

            # Cast back to input dtype for mixed precision compatibility
            return tf.cast(y, input_dtype)

        # Convolve input state with kernel
        y_state = ssm_convolution(x_state, kernel)

        # Apply C as state-to-output projection
        # C: (d_state,) -> broadcast to (batch, seq, d_state)
        y_state = y_state * tf.cast(self.C[None, None, :], input_dtype)

        # Project back to model dimension
        y = self.state_to_output(y_state)

        # Add skip connection (D term)
        y = y + tf.cast(self.D, input_dtype) * z

        # Output projection and dropout
        y = self.output_proj(y)
        y = self.dropout(y, training=training)

        # Residual connection
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
        use_hippo_init: Whether to use HiPPO initialization for SSM
    """

    def __init__(
        self,
        d_model,
        d_state=32,
        d_ff=None,
        dropout_rate=0.1,
        use_hippo_init=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff or d_model * 2
        self.dropout_rate = dropout_rate
        self.use_hippo_init = use_hippo_init

    def build(self, input_shape):
        # SSM layer
        self.ssm = SSMLayer(
            d_model=self.d_model,
            d_state=self.d_state,
            dropout_rate=self.dropout_rate,
            use_hippo_init=self.use_hippo_init,
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
            'dropout_rate': self.dropout_rate,
            'use_hippo_init': self.use_hippo_init,
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
    input_shape=(480, 854, 3),
    use_hippo_init=True
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
        use_hippo_init: Whether to use HiPPO initialization for SSM

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

    # Wrap backbone to handle training mode correctly when frozen
    # This prevents BN momentum updates when backbone_trainable_layers=0
    backbone_wrapper = BackboneWrapper(
        backbone=backbone,
        trainable_layers=backbone_trainable_layers,
        name='backbone_wrapper'
    )

    # Extract features
    features_flat = backbone_wrapper(frames_preprocessed)  # (batch*seq, 2048)

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
            use_hippo_init=use_hippo_init,
            name=f'ssm_block_{i}'
        )(x)

    # Final layer normalization
    x = LayerNormalization(name='final_norm')(x)

    # ========== OUTPUT HEADS ==========
    # Shared intermediate layer
    shared = Dense(128, activation='relu', name='shared_dense')(x)
    shared = Dropout(dropout_rate)(shared)

    # Head 1: Remaining time in current phase (raw frame counts)
    # Using softplus to ensure non-negative outputs while allowing unbounded predictions
    # Softplus: f(x) = log(1 + exp(x)), always positive, smooth gradients
    remaining_phase = Dense(
        1,
        activation='softplus',  # Outputs raw frame counts (non-negative)
        name='remaining_phase'
    )(shared)

    # Head 2: Future phase start times (6 values for phases 1-6)
    # Raw frame counts; -1 in targets indicates phase already passed (masked in loss)
    # Using softplus for non-negative outputs
    future_phase_starts = Dense(
        NUM_PHASES - 1,  # 6 future phases
        activation='softplus',  # Outputs raw frame counts (non-negative)
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
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing Time Predictor Model with S4-style SSM...")
    print("=" * 60)

    # Test HiPPO initialization
    print("\n1. Testing HiPPO Initialization...")
    hippo_vals = hippo_legs_initializer(32)
    print(f"   HiPPO A_log shape: {hippo_vals.shape}")
    print(f"   HiPPO A_log range: [{hippo_vals.min():.4f}, {hippo_vals.max():.4f}]")
    print(f"   Expected: log(1) to log(32) = [0.0, 3.47]")

    # Verify the values recover correctly
    A_recovered = -np.exp(hippo_vals)
    print(f"   Recovered A diagonal: [{A_recovered[0]:.1f}, {A_recovered[1]:.1f}, ..., {A_recovered[-1]:.1f}]")
    print(f"   Expected: [-1, -2, ..., -32]")
    assert np.allclose(A_recovered, -np.arange(1, 33)), "HiPPO initialization incorrect!"
    print("   [OK] HiPPO initialization verified")

    # Test SSM Layer
    print("\n2. Testing SSMLayer (with HiPPO + learnable Δ + bilinear)...")
    ssm_layer = SSMLayer(d_model=64, d_state=32, use_hippo_init=True)
    test_input = tf.random.normal((2, 10, 64))  # (batch, seq, features)
    ssm_output = ssm_layer(test_input)
    print(f"   SSM Input shape: {test_input.shape}")
    print(f"   SSM Output shape: {ssm_output.shape}")

    # Verify shapes are preserved (not collapsed)
    assert ssm_output.shape == test_input.shape, "Output shape mismatch!"
    print("   [OK] Shape preserved (no scalar collapse)")

    # Check learnable delta is created
    delta_weight = [w for w in ssm_layer.weights if 'log_delta' in w.name]
    assert len(delta_weight) == 1, "log_delta weight not found!"
    print(f"   [OK] Learnable Δ created with shape {delta_weight[0].shape}")

    # Test SSM Block
    print("\n3. Testing SSMBlock...")
    ssm_block = SSMBlock(d_model=64, d_state=32, d_ff=128, use_hippo_init=True)
    block_output = ssm_block(test_input)
    print(f"   Block Output shape: {block_output.shape}")

    # Test full model (with small dimensions for speed)
    print("\n4. Testing full Time Predictor model...")
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
        input_shape=(224, 224, 3),  # Smaller images for testing
        use_hippo_init=True
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

    # Test loss functions (with normalized values in [0, 1])
    print("\n5. Testing loss functions (normalized targets)...")
    # Normalized targets: values in [0, 1] range
    y_true = tf.constant([[0.8], [0.5], [0.1], [0.05]])  # Normalized remaining times
    y_pred = tf.constant([[0.75], [0.55], [0.12], [0.03]])
    loss = huber_loss(y_true, y_pred, delta=0.1)
    print(f"   Huber loss (normalized): {loss.numpy():.4f}")

    # Future phase starts (normalized)
    y_true_future = tf.constant([[0.1, 0.2, -1, -1, -1, -1],
                                  [0.05, 0.15, 0.3, -1, -1, -1]], dtype=tf.float32)
    y_pred_future = tf.constant([[0.095, 0.21, 0.5, 0.5, 0.5, 0.5],
                                  [0.055, 0.145, 0.31, 0.5, 0.5, 0.5]], dtype=tf.float32)
    loss_future = future_phase_loss(y_true_future, y_pred_future, delta=0.1)
    print(f"   Future phase loss (normalized): {loss_future.numpy():.4f}")
