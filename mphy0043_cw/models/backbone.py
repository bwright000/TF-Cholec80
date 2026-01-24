"""
Backbone module for MPHY0043 Coursework.

Provides the ResNet-50 feature extractor used by all models.
The backbone is pretrained on ImageNet and can be fine-tuned.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model


def create_backbone(trainable_layers=1, input_shape=(480, 854, 3)):
    """
    Create a ResNet-50 backbone for feature extraction.
    
    Args:
        trainable_layers: Number of layer blocks to fine-tune from the end.
                          0 = completely frozen (only use pretrained features)
                          1 = fine-tune last block (conv5)
                          2 = fine-tune last 2 blocks (conv4 + conv5)
        input_shape: Shape of input images (H, W, C)
        
    Returns:
        Keras Model that outputs 2048-d feature vectors
    """
    # Load pretrained ResNet-50 without the classification head
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,  # Remove the 1000-class classification layer
        input_shape=input_shape
    )
    
    # Freeze layers based on trainable_layers parameter
    if trainable_layers == 0:
        # Freeze entire backbone
        base_model.trainable = False
    else:
        # Freeze early layers, allow fine-tuning of later layers
        base_model.trainable = True
        
        # ResNet-50 has 5 conv blocks: conv1, conv2, conv3, conv4, conv5
        # Freeze everything except the last `trainable_layers` blocks
        # Layer names in ResNet50: conv1_, conv2_, conv3_, conv4_, conv5_
        freeze_until = 5 - trainable_layers  # e.g., trainable_layers=1 → freeze until conv4
        
        for layer in base_model.layers:
            # Check which block this layer belongs to
            layer_name = layer.name
            should_freeze = True
            
            for block_num in range(freeze_until + 1, 6):  # Blocks to keep trainable
                if f'conv{block_num}' in layer_name:
                    should_freeze = False
                    break
            
            layer.trainable = not should_freeze
    
    # Add global average pooling to get fixed-size output
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # (batch, 7, 27, 2048) → (batch, 2048)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=x, name='resnet50_backbone')
    
    return model


def get_backbone_output_dim():
    """Returns the output dimension of the backbone (2048 for ResNet-50)."""
    return 2048


if __name__ == '__main__':
    # Test the backbone
    print("Testing backbone...")
    
    backbone = create_backbone(trainable_layers=1)
    backbone.summary()
    
    # Test with dummy input
    dummy_input = tf.random.uniform((2, 480, 854, 3), minval=0, maxval=255)
    output = backbone(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dim: {get_backbone_output_dim()}")
    
    # Count trainable vs non-trainable parameters
    trainable = sum([tf.reduce_prod(v.shape).numpy() for v in backbone.trainable_variables])
    non_trainable = sum([tf.reduce_prod(v.shape).numpy() for v in backbone.non_trainable_variables])
    print(f"\nTrainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
    
    print("\nTest complete!")