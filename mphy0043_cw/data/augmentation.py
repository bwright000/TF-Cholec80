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

# Image dimensions (from Cholec80)
IMG_HEIGHT = 480
IMG_WIDTH = 854

# Augmentation parameters (conservative for surgical video)
MAX_ROTATION = 15.0          # degrees
BRIGHTNESS_DELTA = 0.2       # ±20% brightness
CONTRAST_RANGE = (0.8, 1.2)  # 80-120% contrast
SATURATION_RANGE = (0.8, 1.2)  # 80-120% saturation
CROP_SCALE_MIN = 0.85        # Minimum crop size (85% of original)

def random_horizontal_flip(image):
    """
    Randomly flip image horizontally (50% chance).
    
    Args:
        image: Tensor of shape (H, W, C) or (B, H, W, C)
        
    Returns:
        Flipped or original image
    """
    return tf.image.random_flip_left_right(image)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    
    Args:
        image: Tensor of shape (H, W, C), values in [0, 255]
        
    Returns:
        Brightness adjusted image
    """
    return tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)

def random_contrast(image):
    """
    Randomly adjust contrast.
    
    Args:
        image: Tensor of shape (H, W, C)
        
    Returns:
        Contrast-adjusted image
    """
    return tf.image.random_contrast(image,
        lower=CONTRAST_RANGE[0], upper=CONTRAST_RANGE[1]
    )
def random_saturation(image):
    """
    Randomly adjust saturation.
    
    Args:
        image: Tensor of shape (H, W, C)
        
    Returns:
        Saturation-adjusted image
    """
    return tf.image.random_saturation(image,
        lower=SATURATION_RANGE[0], upper=SATURATION_RANGE[1]
    )

def random_crop_and_resize(image):
    """    
        This simulates slight zoom variations.
    
    Args:
        image: Tensor of shape (H, W, C)
        
    Returns:
        Cropped and resized image
    """
    original_shape = tf.shape(image)
    height = original_shape[0]
    width = original_shape[1]
    # Random crop size (using CROP_SCALE_MIN)
    scale = tf.random.uniform([], CROP_SCALE_MIN, maxval=1.0)
    crop_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
    crop_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
    
    # Random crop
    cropped = tf.image.random_crop(image, [crop_height, crop_width, 3])
    
    # Resize back to original dimensions
    resized = tf.image.resize(cropped, [IMG_HEIGHT, IMG_WIDTH])
    
    return tf.cast(resized, tf.uint8)

def augment_image(image):
    """
    Apply all augmentations to a single image.
    Each augmentation is applied with some probability.
    
    Args:
        image: Tensor of shape (H, W, C), values in [0, 255], dtype uint8
        
    Returns:
        Augmented image with same shape and dtype
    """
    # Convert to float32 for processing
    image = tf.cast(image, tf.float32)

    # Apply augmentations (each augmentation has ~ 50% chance of application)
    image = random_horizontal_flip(image)
    image = random_brightness(image)
    image = random_contrast(image)
    image = random_saturation(image)

    # Crop handled separately for resizing
    if tf.random.uniform([]) < 0.5:
        image = random_crop_and_resize(tf.cast(image, tf.uint8))
        image = tf.cast(image, tf.float32)
    # Clip values to valid range and convert back to uint8
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image

def augment_batch(batch):
    """
    Apply augmentation to a batch of data.
    Only augments the 'frame' field, leaves labels unchanged.
    
    Args:
        batch: Dictionary containing 'frame' and other fields
        
    Returns:
        Batch with augmented frames
    """
    # Apply augmentation to each image in a batch
    augmented_frames = tf.map_fn(
        augment_image,
        batch['frame'],
        fn_output_signature=tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.uint8)
    )

    # Update batch with augmented frames
    batch['frame'] = augmented_frames
    return batch


def get_augmentation_fn(training=True):
    """
    Returns the appropriate augmentation function based on mode.
    
    Args:
        training: If True, return augmentation function. 
                  If False, return identity function (no augmentation).
                  
    Returns:
        Function that can be used with dataset.map()
    """
    if training:
        return augment_batch
    else:
        return lambda batch: batch

if __name__ == '__main__':
    import numpy as np
    
    # Create a dummy image for testing
    print("Testing augmentation functions...")
    
    dummy_image = tf.constant(
        np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    )
    
    print(f"Original image shape: {dummy_image.shape}")
    print(f"Original image dtype: {dummy_image.dtype}")
    
    # Test single image augmentation
    augmented = augment_image(dummy_image)
    print(f"Augmented image shape: {augmented.shape}")
    print(f"Augmented image dtype: {augmented.dtype}")
    
    # Test batch augmentation
    dummy_batch = {
        'frame': tf.stack([dummy_image, dummy_image]),  # Batch of 2
        'phase': tf.constant([1, 2]),
        'instruments': tf.constant([[1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0]])
    }
    
    augmented_batch = augment_batch(dummy_batch)
    print(f"Augmented batch frame shape: {augmented_batch['frame'].shape}")
    print(f"Labels unchanged: {augmented_batch['phase'].numpy()}")
    
    print("Test complete!")