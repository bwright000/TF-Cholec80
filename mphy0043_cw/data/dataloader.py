"""
DataLoader for MPHY0043 Coursework.
Provides training, validation, and test datasets using raw Cholec80 data.

This module wraps dataset.py and preprocessing.py to provide convenient
functions for loading datasets with timing labels.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import tensorflow as tf
import numpy as np

# Import from our dataset module
from mphy0043_cw.data.augmentation import augment_batch
from mphy0043_cw.data.dataset import (
    create_dataset,
    create_sequence_dataset,
    load_video_data,
    TRAIN_VIDEO_IDS,
    VAL_VIDEO_IDS,
    TEST_VIDEO_IDS,
    NUM_PHASES,
    NUM_TOOLS,
    FRAME_HEIGHT,
    FRAME_WIDTH
)
from mphy0043_cw.data.preprocessing import load_timing_labels


def get_train_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None,
    shuffle: bool = True,
    augment: bool = True,
    video_ids: Optional[List[int]] = None
) -> tf.data.Dataset:
    """
    Get training dataset (videos 01-32 by default).

    Args:
        data_dir: Path to Cholec80 data directory
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        video_ids: Optional list of video IDs (1-indexed). Defaults to TRAIN_VIDEO_IDS.

    Returns:
        tf.data.Dataset with training data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    # Use provided video_ids or default to TRAIN_VIDEO_IDS
    ids_to_use = video_ids if video_ids is not None else TRAIN_VIDEO_IDS

    return create_dataset(
        data_dir=data_dir,
        video_ids=ids_to_use,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=augment,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_val_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None,
    video_ids: Optional[List[int]] = None
) -> tf.data.Dataset:
    """
    Get validation dataset (videos 33-40 by default).

    Args:
        data_dir: Path to Cholec80 data directory
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)
        video_ids: Optional list of video IDs (1-indexed). Defaults to VAL_VIDEO_IDS.

    Returns:
        tf.data.Dataset with validation data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    # Use provided video_ids or default to VAL_VIDEO_IDS
    ids_to_use = video_ids if video_ids is not None else VAL_VIDEO_IDS

    return create_dataset(
        data_dir=data_dir,
        video_ids=ids_to_use,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_test_dataset(
    data_dir: str,
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None,
    video_ids: Optional[List[int]] = None
) -> tf.data.Dataset:
    """
    Get test dataset (videos 41-80 by default).

    Args:
        data_dir: Path to Cholec80 data directory
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)
        video_ids: Optional list of video IDs (1-indexed). Defaults to TEST_VIDEO_IDS.

    Returns:
        tf.data.Dataset with test data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    # Use provided video_ids or default to TEST_VIDEO_IDS
    ids_to_use = video_ids if video_ids is not None else TEST_VIDEO_IDS

    return create_dataset(
        data_dir=data_dir,
        video_ids=ids_to_use,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_dataset_for_videos(
    data_dir: str,
    video_ids: List[int],
    batch_size: int = 8,
    timing_labels_path: Optional[str] = None,
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Get dataset for specific video IDs.

    Args:
        data_dir: Path to Cholec80 data directory
        video_ids: List of video IDs (1-indexed)
        batch_size: Batch size
        timing_labels_path: Path to timing_labels.npz (optional)
        shuffle: Whether to shuffle data

    Returns:
        tf.data.Dataset with specified videos
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    return create_dataset(
        data_dir=data_dir,
        video_ids=video_ids,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=False,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


# ============================================================================
# SEQUENCE DATASETS (For SSM Time Predictor)
# ============================================================================

def get_val_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 64,
    timing_labels_path: Optional[str] = None,
    video_ids: Optional[List[int]] = None
) -> tf.data.Dataset:
    """
    Get validation sequence dataset (videos 33-40 by default).

    Args:
        data_dir: Path to Cholec80 data directory
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        stride: Step size between sequence starts
        timing_labels_path: Path to timing_labels.npz
        video_ids: Optional list of video IDs (1-indexed). Defaults to VAL_VIDEO_IDS.

    Returns:
        tf.data.Dataset with sequence data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    # Use provided video_ids or default to VAL_VIDEO_IDS
    ids_to_use = video_ids if video_ids is not None else VAL_VIDEO_IDS

    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=ids_to_use,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        stride=stride,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )


def get_test_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 64,
    timing_labels_path: Optional[str] = None,
    video_ids: Optional[List[int]] = None
) -> tf.data.Dataset:
    """
    Get test sequence dataset (videos 41-80 by default).

    Args:
        data_dir: Path to Cholec80 data directory
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        stride: Step size between sequence starts
        timing_labels_path: Path to timing_labels.npz
        video_ids: Optional list of video IDs (1-indexed). Defaults to TEST_VIDEO_IDS.

    Returns:
        tf.data.Dataset with sequence data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    # Use provided video_ids or default to TEST_VIDEO_IDS
    ids_to_use = video_ids if video_ids is not None else TEST_VIDEO_IDS

    return create_sequence_dataset(
        data_dir=data_dir,
        video_ids=ids_to_use,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        stride=stride,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )

def get_train_sequence_dataset(
    data_dir: str,
    sequence_length: int = 64,
    batch_size: int = 4,
    stride: int = 32,
    timing_labels_path: Optional[str] = None,
    augment: bool = True,
    video_ids: Optional[List[int]] = None
) -> tf.data.Dataset:
    """
    Get training sequence dataset (videos 01-32 by default).

    Args:
        data_dir: Path to Cholec80 data directory
        sequence_length: Number of consecutive frames per sequence
        batch_size: Number of sequences per batch
        stride: Step size between sequence starts
        timing_labels_path: Path to timing_labels.npz
        augment: Whether to apply augmentation
        video_ids: Optional list of video IDs (1-indexed). Defaults to TRAIN_VIDEO_IDS.

    Returns:
        tf.data.Dataset with sequence data
    """
    timing_labels = None
    if timing_labels_path and os.path.exists(timing_labels_path):
        timing_labels = load_timing_labels(timing_labels_path)

    # Use provided video_ids or default to TRAIN_VIDEO_IDS
    ids_to_use = video_ids if video_ids is not None else TRAIN_VIDEO_IDS

    # 1. Create the base sequence dataset
    dataset = create_sequence_dataset(
        data_dir=data_dir,
        video_ids=ids_to_use,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True,
        stride=stride,
        include_timing=timing_labels is not None,
        timing_labels=timing_labels
    )

    # 2. Apply sequence-level augmentation if requested
    if augment:
        # We apply this AFTER batching in create_sequence_dataset
        # so the function receives (Batch, Seq, H, W, C)
        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # 3. Always prefetch for cluster performance
    return dataset.prefetch(tf.data.AUTOTUNE)

# ============================================================================
# TEST
# ============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Cholec80 directory')
    parser.add_argument('--timing_labels', type=str, default=None, help='Path to timing_labels.npz')
    args = parser.parse_args()

    print("Testing Dataloader Consistency...")
    print("=" * 60)

    # 1. Test Standard Dataset (Task B)
    print("\n[CHECK] Loading Single-Frame Dataset...")
    train_ds = get_train_dataset(args.data_dir, batch_size=2, timing_labels_path=args.timing_labels)
    for batch in train_ds.take(1):
        print(f"  - Single Frame Keys: {list(batch.keys())}")
        print(f"  - Image shape: {batch['frame'].shape}") # Task B uses 'frame'

    # 2. Test Sequence Dataset (Task A)
    print("\n[CHECK] Loading Sequence Dataset (SSM)...")
    seq_ds = get_train_sequence_dataset(
        args.data_dir, 
        sequence_length=128, 
        batch_size=2, 
        timing_labels_path=args.timing_labels,
        augment=True # Test the new augmentation!
    )
    
    for batch in seq_ds.take(1):
        print(f"  - Sequence Keys: {list(batch.keys())}")
        print(f"  - Frames shape: {batch['frames'].shape}") # Task A uses 'frames'
        print(f"  - Phase shape: {batch['phase'].shape}")
        
        # Check if augmentation worked (values should be uint8)
        print(f"  - Data Type: {batch['frames'].dtype}")
        
    print("\n" + "=" * 60)
    print("Dataloader test complete! Ready for Cluster.")
