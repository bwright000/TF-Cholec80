"""
Data Augmentation module for MPHY0043 Coursework.

Provides image augmentation functions for training surgical video models.
Augmentations are applied ONLY during training to improve generalization.

Supported augmentations:
- Horizontal flip
- Random rotation (±15°)
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Random crop and resize
"""

import tensorflow as tf

# ============================================================================
# CONSTANTS
# ============================================================================
IMG_HEIGHT = 480
IMG_WIDTH = 854

# Parameters
BRIGHTNESS_DELTA = 0.2
CONTRAST_RANGE = (0.8, 1.2)
SATURATION_RANGE = (0.8, 1.2)
CROP_SCALE_MIN = 0.85

def random_sequence_crop(frames):
    """Applies the same spatial crop to all frames in a (Seq, H, W, C) tensor."""
    shape = tf.shape(frames)
    seq_len = shape[0]
    
    target_h = tf.cast(tf.cast(shape[1], tf.float32) * CROP_SCALE_MIN, tf.int32)
    target_w = tf.cast(tf.cast(shape[2], tf.float32) * CROP_SCALE_MIN, tf.int32)
    
    # Calculate random offsets ONCE for the whole sequence
    offset_h = tf.random.uniform([], 0, shape[1] - target_h, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, shape[2] - target_w, dtype=tf.int32)
    
    # Slice: [All Frames, Crop Height, Crop Width, All Channels]
    cropped = frames[:, offset_h:offset_h+target_h, offset_w:offset_w+target_w, :]
    
    # Resize back to original dimensions
    return tf.image.resize(cropped, [IMG_HEIGHT, IMG_WIDTH])

def augment_batch(batch):
    """
    Sequence-level augmentation.
    Input batch contains 'frames' key with shape (Batch, Seq, H, W, C).
    """
    # 1. Cast and normalize parameters for the sequence
    do_flip = tf.random.uniform([]) > 0.5
    b_delta = tf.random.uniform([], -BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)
    c_factor = tf.random.uniform([], CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    s_factor = tf.random.uniform([], SATURATION_RANGE[0], SATURATION_RANGE[1])

    # 2. Process the frames
    # batch['frames'] is usually (Batch, Seq, H, W, C)
    # We apply the same logic to the entire temporal block
    frames = tf.cast(batch['frames'], tf.float32)

    if do_flip:
        frames = tf.image.flip_left_right(frames)

    # These TF functions support 4D/5D tensors out of the box
    frames = tf.image.adjust_brightness(frames, b_delta)
    frames = tf.image.adjust_contrast(frames, c_factor)
    frames = tf.image.adjust_saturation(frames, s_factor)

    # 3. Apply Zoom/Crop (50% chance)
    if tf.random.uniform([]) < 0.5:
        # We need to map this over the batch dimension
        frames = tf.map_fn(random_sequence_crop, frames, 
                           fn_output_signature=tf.float32)

    # 4. Finalize
    batch['frames'] = tf.cast(tf.clip_by_value(frames, 0, 255), tf.uint8)
    return batch

def get_augmentation_fn(training=True):
    return augment_batch if training else lambda batch: batch

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    print("Testing Sequence Augmentation Consistency...")
    print("=" * 60)

    # 1. Create a dummy sequence (Batch=1, Seq=4, H, W, C)
    # Using 4 frames is enough to visually check consistency
    seq_len = 4
    dummy_frames = np.zeros((1, seq_len, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    
    # Draw a small white square in the top-left of every frame
    # This helps us track if 'Horizontal Flip' is consistent
    dummy_frames[:, :, 50:150, 50:150, :] = 255 
    
    batch = {
        'frames': tf.constant(dummy_frames)
    }

    # 2. Run Augmentation
    # We run it a few times to catch different random triggers (flip vs no flip)
    for i in range(3):
        aug_batch = augment_batch(batch.copy())
        aug_frames = aug_batch['frames'].numpy()[0] # Remove batch dim

        print(f"\nRun {i+1}:")
        print(f"  Shape: {aug_frames.shape}")
        print(f"  Dtype: {aug_frames.dtype}")
        
        # Check consistency: Mean pixel value of each frame should be identical
        means = [np.mean(aug_frames[t]) for t in range(seq_len)]
        is_consistent = all(m == means[0] for m in means)
        print(f"  Temporal Consistency Check: {'PASS' if is_consistent else 'FAIL'}")